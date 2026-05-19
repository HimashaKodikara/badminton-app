import subprocess
import json
import uuid
import sys
from pathlib import Path
from collections import defaultdict, deque, Counter

import cv2
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import imageio_ffmpeg

# =========================
# PATH CONFIG
# =========================
BASE = Path("D:/UOM/L4S1/research/new/badminton_app2")

TRACKNET_DIR = BASE / "TrackNetV3"
MODELS_DIR = BASE / "models"
UPLOADS_DIR = BASE / "uploads"
OUTPUTS_DIR = BASE / "outputs"
STATIC_DIR = BASE / "static"

PLAYER_MODEL_PATH = MODELS_DIR / "player_best.pt"
COURT_MODEL_PATH = MODELS_DIR / "court_best.pt"
TRACKNET_PT = TRACKNET_DIR / "ckpts" / "TrackNet_best.pt"
INPAINT_PT = TRACKNET_DIR / "ckpts" / "InpaintNet_best.pt"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Increase this when final running. For full video, set to None.
MAX_FRAMES = 600

# Court model is expensive on CPU. Detect court every N frames and reuse last court points.
COURT_DETECT_INTERVAL = 10

RUN_YOLO_ANALYSIS = True

app = FastAPI()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

SHOT_CLASSES = [
    "ReadyPosition",
    "Service",
    "Smash",
    "BackHand",
    "Lift",
    "NetShot",
    "ForeHand",
]

COURT_LINE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 7), (7, 10), (10, 12), (12, 15),
    (4, 6), (6, 9), (9, 11), (11, 14), (14, 16), (16, 21),
    (18, 19), (19, 20), (20, 21),
    (5, 17), (17, 18),
]

LEFT_COURT_POINTS = [17, 15, 12, 10, 7, 5, 0]
RIGHT_COURT_POINTS = [21, 16, 14, 11, 9, 6, 4]


def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


def is_shot_class(name):
    return name in SHOT_CLASSES


def select_near_camera(detections, frame_h):
    best = None
    best_score = -1

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cy = (y1 + y2) / 2

        # Select bottom / near-side player
        if cy < frame_h * 0.45:
            continue

        score = (cy / frame_h) * 0.7 + (((x2 - x1) * (y2 - y1)) / (frame_h * frame_h)) * 0.3

        if score > best_score:
            best_score = score
            best = det

    return best


def get_shuttle(shuttle_df, frame_no):
    idx = frame_no - 1

    if idx < 0 or idx >= len(shuttle_df):
        return None, None, 0

    row = shuttle_df.iloc[idx]
    cols = {c.lower(): c for c in shuttle_df.columns}

    x_col = cols.get("x")
    y_col = cols.get("y")
    v_col = cols.get("visibility")

    if not x_col or not y_col:
        return None, None, 0

    x = row[x_col]
    y = row[y_col]
    vis = row[v_col] if v_col else 1

    if pd.isna(x) or pd.isna(y) or vis == 0:
        return None, None, 0

    return int(x), int(y), int(vis)


def draw_court(frame, court_model):
    result = court_model.predict(
        source=frame,
        imgsz=960,
        conf=0.10,
        device="cpu",
        verbose=False,
    )

    court_points = None

    for r in result:
        if r.keypoints is None:
            continue

        for cp in r.keypoints.xy.cpu().numpy():
            court_points = cp

            for p1, p2 in COURT_LINE_PAIRS:
                if p1 >= len(cp) or p2 >= len(cp):
                    continue

                x1, y1 = cp[p1]
                x2, y2 = cp[p2]

                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            for x, y in cp:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    return frame, court_points


def draw_cached_court(frame, court_points):
    if court_points is None:
        return frame

    for p1, p2 in COURT_LINE_PAIRS:
        if p1 >= len(court_points) or p2 >= len(court_points):
            continue

        x1, y1 = court_points[p1]
        x2, y2 = court_points[p2]

        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    for x, y in court_points:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    return frame


def valid_side(points, indexes):
    valid_points = []

    if points is None:
        return valid_points

    for i in indexes:
        if i < len(points):
            x, y = points[i]
            if x > 0 and y > 0:
                valid_points.append((float(x), float(y)))

    return sorted(valid_points, key=lambda point: point[1])


def boundary_x(side_points, shuttle_y):
    if len(side_points) < 2:
        return None

    for i in range(len(side_points) - 1):
        x1, y1 = side_points[i]
        x2, y2 = side_points[i + 1]

        if min(y1, y2) <= shuttle_y <= max(y1, y2):
            if y2 == y1:
                return min(x1, x2)

            return x1 + (shuttle_y - y1) / (y2 - y1) * (x2 - x1)

    return min(side_points, key=lambda point: abs(point[1] - shuttle_y))[0]


def inside_court(x, y, court_points):
    if x is None or y is None or court_points is None:
        return None

    left_points = valid_side(court_points, LEFT_COURT_POINTS)
    right_points = valid_side(court_points, RIGHT_COURT_POINTS)

    if len(left_points) < 2 or len(right_points) < 2:
        return None

    left_x = boundary_x(left_points, y)
    right_x = boundary_x(right_points, y)

    if left_x is None or right_x is None:
        return None

    if left_x > right_x:
        left_x, right_x = right_x, left_x

    return not (x < left_x or x > right_x)


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def normalize_shot_name(name):
    name = str(name).lower().strip()
    mapping = {
        "smash": "smash",
        "drop": "drop",
        "netshot": "net",
        "net": "net",
        "service": "serve",
        "serve": "serve",
        "lift": "clear",
        "forehand": "clear",
        "backhand": "clear",
        "clear": "clear",
        "readyposition": "unknown",
    }
    return mapping.get(name, name)


def get_court_status(value):
    text = str(value).lower()
    if value is True or value == 1 or text == "true":
        return "in"
    if value is False or value == 0 or text == "false":
        return "out"
    return "unknown"


def get_weak_reason(row):
    reason = str(row.get("weak_shot_reason", "Normal"))
    try:
        is_weak = int(row.get("weak_shot_binary", 0)) == 1
    except Exception:
        is_weak = False

    if is_weak:
        if "outside" in reason.lower():
            return "shuttle_landed_outside_court"
        return reason

    return "normal"


def get_zone(row, shot_df):
    court_status = get_court_status(row.get("inside_court"))
    x = row.get("shuttle_x")

    if court_status == "in":
        return "inside_court"

    if pd.isna(x) or shot_df.empty or "shuttle_x" not in shot_df.columns:
        return "unknown_out"

    valid_x = shot_df["shuttle_x"].dropna()
    if valid_x.empty:
        return "unknown_out"

    median_x = valid_x.median()
    return "left_out" if x < median_x else "right_out"


def get_shot_trajectory(shuttle_df, shot_frame, window=3):
    if shuttle_df.empty:
        return []

    points = shuttle_df[
        (shuttle_df["frame"] >= shot_frame - window) &
        (shuttle_df["frame"] <= shot_frame + window)
    ]

    trajectory = []
    for _, row in points.iterrows():
        trajectory.append({
            "frame": int(row["frame"]),
            "x": clean_value(row.get("shuttle_x")),
            "y": clean_value(row.get("shuttle_y")),
        })

    return trajectory


def get_start_end_position(trajectory):
    valid_points = [p for p in trajectory if p["x"] is not None and p["y"] is not None]

    if not valid_points:
        return {"x": None, "y": None}, {"x": None, "y": None}

    return (
        {"x": valid_points[0]["x"], "y": valid_points[0]["y"]},
        {"x": valid_points[-1]["x"], "y": valid_points[-1]["y"]},
    )


def build_tactical_outputs(
    job_id,
    frame_records,
    shot_records,
    shuttle_records,
    transition_matrix,
    output_paths,
):
    frame_df = pd.DataFrame(frame_records)
    shot_df = pd.DataFrame(shot_records)
    shuttle_df = pd.DataFrame(shuttle_records)

    frame_df.to_csv(output_paths["frame_csv"], index=False)
    shot_df.to_csv(output_paths["shot_csv"], index=False)
    shuttle_df.to_csv(output_paths["shuttle_analysis_csv"], index=False)

    transition_df = pd.DataFrame(0, index=SHOT_CLASSES, columns=SHOT_CLASSES)
    for from_shot, next_dict in transition_matrix.items():
        for to_shot, count in next_dict.items():
            if from_shot in transition_df.index and to_shot in transition_df.columns:
                transition_df.loc[from_shot, to_shot] = count

    transition_df.to_csv(output_paths["transition_csv"])

    frame_df.to_json(output_paths["frame_json"], orient="records", indent=2)
    shot_df.to_json(output_paths["shot_json"], orient="records", indent=2)
    shuttle_df.to_json(output_paths["shuttle_json"], orient="records", indent=2)
    transition_df.to_json(output_paths["transition_json"], orient="records", indent=2)

    total_shots = len(shot_records)

    percentage_distribution = (
        {
            key: round(value / total_shots * 100, 1)
            for key, value in Counter(s["shot_type"] for s in shot_records).items()
        }
        if total_shots
        else {}
    )

    if shot_df.empty or "shot_type" not in shot_df.columns:
        shot_type_list = []
        shot_counts = Counter()
        average_rally_length = 0
    else:
        shot_type_list = [normalize_shot_name(x) for x in shot_df["shot_type"].fillna("unknown")]
        shot_counts = Counter([s for s in shot_type_list if s != "unknown"])
        average_rally_length = shot_df.groupby("rally_id").size().mean() if "rally_id" in shot_df.columns else 0
        if pd.isna(average_rally_length):
            average_rally_length = 0

    current_video_profile = {
        "shot_distribution": dict(shot_counts),
        "shot_distribution_percentage_raw_classes": percentage_distribution,
        "average_rally_length": round(float(average_rally_length), 2),
        "preferred_zones": [],
    }

    shots = []
    for i, row in shot_df.iterrows():
        shot_type = normalize_shot_name(row.get("shot_type", "unknown"))
        current_frequency = shot_counts.get(shot_type, 0)
        shot_frame = int(row.get("frame", 0))
        trajectory = get_shot_trajectory(shuttle_df, shot_frame)
        start_position, end_position = get_start_end_position(trajectory)
        is_weak = int(row.get("weak_shot_binary", 0)) == 1
        court_status = get_court_status(row.get("inside_court"))

        shots.append({
            "shot_id": i + 1,
            "rally_id": int(row.get("rally_id", 1)),
            "shot_type": shot_type,
            "frame": shot_frame,
            "timestamp": clean_value(row.get("time_sec")),
            "current_frequency": current_frequency,
            "start_position": start_position,
            "end_position": end_position,
            "trajectory": trajectory,
            "weak_shot": {
                "is_weak": is_weak,
                "weakness_score": 0.82 if is_weak else 0.0,
                "reason": get_weak_reason(row),
                "court_status": court_status,
                "zone": get_zone(row, shot_df),
            },
            "tactical_analysis": {
                "is_effective": not is_weak,
                "reason": "weak_shot_detected" if is_weak else "normal",
            },
        })

    rallies = []
    if not shot_df.empty and "rally_id" in shot_df.columns:
        for rally_id, group in shot_df.groupby("rally_id"):
            length = len(group)
            weak_count = int(group["weak_shot_binary"].fillna(0).sum()) if "weak_shot_binary" in group.columns else 0

            if length <= 5:
                tempo = "fast"
            elif length <= 10:
                tempo = "medium"
            else:
                tempo = "slow"

            rallies.append({
                "rally_id": int(rally_id),
                "length": length,
                "average_rally_length_from_video": round(float(average_rally_length), 2),
                "deviation_from_video_average": round(length - float(average_rally_length), 2),
                "tempo": tempo,
                "fatigue_impact": "high" if length >= 8 else "low",
                "weak_shots_count": weak_count,
            })

    current_transition_matrix = {}
    for from_shot, next_dict in transition_matrix.items():
        for to_shot, count in next_dict.items():
            key = f"{normalize_shot_name(from_shot)}_to_{normalize_shot_name(to_shot)}"
            current_transition_matrix[key] = current_transition_matrix.get(key, 0) + int(count)

    transition_values = list(current_transition_matrix.values())
    avg_transition = sum(transition_values) / len(transition_values) if transition_values else 0

    sequence_deviation = {}
    for key, value in current_transition_matrix.items():
        sequence_deviation[key] = {
            "current": value,
            "average_transition_frequency": round(avg_transition, 2),
            "deviation": round(value - avg_transition, 2),
        }

    rally_comparison = []
    for rally in rallies:
        rally_comparison.append({
            "rally_id": rally["rally_id"],
            "current_length": rally["length"],
            "average_rally_length_from_video": rally["average_rally_length_from_video"],
            "deviation_from_video_average": rally["deviation_from_video_average"],
            "tempo": rally["tempo"],
            "weak_shots_count": rally["weak_shots_count"],
        })

    dominant_shot = shot_counts.most_common(1)[0][0] if shot_counts else "unknown"
    total_transitions = sum(current_transition_matrix.values())
    predictability_score = 0.0
    if total_transitions > 0:
        predictability_score = round(max(current_transition_matrix.values()) / total_transitions, 2)

    main_output = {
        "match_id": "match_001",
        "player_id": "player_01",
        "job_id": job_id,
        "current_video_profile": current_video_profile,
        "shots": shots,
        "rallies": rallies,
        "tactical_patterns": {
            "dominant_shot": dominant_shot,
            "predictability_score": predictability_score,
        },
        "transition_matrix": current_transition_matrix,
        "sequence_deviation": sequence_deviation,
        "rally_comparison": rally_comparison,
    }

    with open(output_paths["main_json"], "w", encoding="utf-8") as f:
        json.dump(main_output, f, indent=2)

    m2_output = {
        "match_id": "match_001",
        "job_id": job_id,
        "processed_frames": len(frame_records),
        "max_frames_used": MAX_FRAMES,
        "total_shots": total_shots,
        "shot_distribution": percentage_distribution,
        "transition_matrix": {k: dict(v) for k, v in transition_matrix.items()},
        "shots": shot_records,
    }

    with open(output_paths["m2_json"], "w", encoding="utf-8") as f:
        json.dump(m2_output, f, indent=2)

    return {
        "total_shots": total_shots,
        "shot_distribution": percentage_distribution,
        "main_output_preview": main_output,
    }


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "FastAPI badminton analyzer is running."})


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    original_name = Path(video.filename).name
    video_path = UPLOADS_DIR / f"{job_id}_{original_name}"

    with open(video_path, "wb") as f:
        f.write(await video.read())

    cap_test = cv2.VideoCapture(str(video_path))
    if not cap_test.isOpened():
        return JSONResponse({"error": "Could not open video file. Please upload a valid MP4 video."}, status_code=400)
    cap_test.release()

    required_files = [TRACKNET_PT, INPAINT_PT]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        return JSONResponse({"error": "Required TrackNet checkpoint file missing.", "missing": missing}, status_code=500)

    pred_dir = TRACKNET_DIR / f"prediction_{job_id}"
    pred_dir.mkdir(exist_ok=True)

    print("Starting TrackNet prediction...", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            str(TRACKNET_DIR / "predict.py"),
            "--video_file", str(video_path),
            "--tracknet_file", str(TRACKNET_PT),
            "--inpaintnet_file", str(INPAINT_PT),
            "--save_dir", str(pred_dir),
            "--large_video",
            "--eval_mode", "nonoverlap",
            "--batch_size", "1",
        ],
        cwd=str(TRACKNET_DIR),
        capture_output=True,
        text=True,
    )

    print("TrackNet STDOUT:", result.stdout[-3000:], flush=True)
    print("TrackNet STDERR:", result.stderr[-3000:], flush=True)

    if result.returncode != 0:
        return JSONResponse(
            {
                "error": "TrackNet prediction failed.",
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-3000:],
            },
            status_code=500,
        )

    csv_files = list(pred_dir.glob("*_ball.csv"))
    if not csv_files:
        csv_files = list(TRACKNET_DIR.glob(f"**/*_ball.csv"))

    if not csv_files:
        return JSONResponse(
            {
                "error": "TrackNet CSV not found.",
                "prediction_folder": str(pred_dir),
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-3000:],
            },
            status_code=500,
        )

    shuttle_csv = csv_files[0]
    shuttle_df = pd.read_csv(shuttle_csv)

    output_shuttle_csv = OUTPUTS_DIR / f"{job_id}_shuttle.csv"
    shuttle_df.to_csv(output_shuttle_csv, index=False)

    if not RUN_YOLO_ANALYSIS:
        return JSONResponse(
            {
                "job_id": job_id,
                "message": "TrackNet completed successfully. YOLO analysis is disabled.",
                "shuttle_csv": f"/outputs/{job_id}_shuttle.csv",
                "tracknet_csv_path": str(shuttle_csv),
            }
        )

    required_yolo = [PLAYER_MODEL_PATH, COURT_MODEL_PATH]
    missing_yolo = [str(p) for p in required_yolo if not p.exists()]
    if missing_yolo:
        return JSONResponse(
            {
                "error": "TrackNet worked, but YOLO model file missing.",
                "missing": missing_yolo,
                "shuttle_csv": f"/outputs/{job_id}_shuttle.csv",
            },
            status_code=500,
        )

    print("Loading YOLO models...", flush=True)
    player_model = YOLO(str(PLAYER_MODEL_PATH))
    court_model = YOLO(str(COURT_MODEL_PATH))
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.7, max_cosine_distance=0.3)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        width = 1280
        height = 720

    print(f"Video Info -> FPS: {fps}, Width: {width}, Height: {height}", flush=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = OUTPUTS_DIR / f"{job_id}_output.mp4"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print("Writer opened:", writer.isOpened(), flush=True)

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_video}")

    print(f"Writing output video to: {output_video}", flush=True)

    frame_records = []
    shot_records = []
    shuttle_records = []
    shot_sequence = []
    transition_matrix = defaultdict(lambda: defaultdict(int))

    last_shot = None
    last_shot_frame = -999
    shot_cooldown = int(fps * 0.6)
    target_track_id = None
    shuttle_history = deque(maxlen=15)
    last_court_points = None
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        if MAX_FRAMES is not None and frame_no > MAX_FRAMES:
            print(f"Stopped early at {MAX_FRAMES} frames for CPU testing.", flush=True)
            break

        if frame_no % 10 == 0:
            print(f"Processing frame {frame_no}", flush=True)

        timestamp = frame_no / fps
        sx, sy, sv = get_shuttle(shuttle_df, frame_no)

        detected_court_points = None
        if frame_no == 1 or frame_no % COURT_DETECT_INTERVAL == 1:
            frame, detected_court_points = draw_court(frame, court_model)
            if detected_court_points is not None:
                last_court_points = detected_court_points
        else:
            frame = draw_cached_court(frame, last_court_points)

        court_points = last_court_points

        player_result = player_model.predict(frame, conf=0.35, device="cpu", verbose=False)[0]

        all_detections = []
        tracker_detections = []

        if player_result.boxes is not None:
            for box in player_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = player_model.names[cls_id]

                detection = {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
                all_detections.append(detection)

                if is_shot_class(cls_name):
                    tracker_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        candidates = [d for d in all_detections if is_shot_class(d["class_name"])]
        selected = select_near_camera(candidates, height)

        current_shot = "Unknown"
        current_conf = 0
        player_box = None

        if selected:
            current_shot = selected["class_name"]
            current_conf = selected["conf"]
            player_box = selected["box"]

        tracks = tracker.update_tracks(tracker_detections, frame=frame)

        if player_box:
            best_iou = 0
            selected_track = None
            for trk in tracks:
                if not trk.is_confirmed():
                    continue
                iou = compute_iou(list(trk.to_ltrb()), player_box)
                if iou > best_iou:
                    best_iou = iou
                    selected_track = trk

            if selected_track and best_iou > 0.25:
                target_track_id = selected_track.track_id

        is_inside = inside_court(sx, sy, court_points)
        weak = 1 if is_inside is False else 0
        reason = "Outside Court Left/Right" if is_inside is False else "Normal"

        shuttle_history.append({"frame": frame_no, "x": sx, "y": sy, "visibility": sv})

        if player_box:
            x1, y1, x2, y2 = map(int, player_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"ID:{target_track_id} | {current_shot} {current_conf:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if sx is not None and sy is not None:
            if is_inside is True:
                color = (0, 165, 255)
            elif is_inside is False:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            cv2.circle(frame, (sx, sy), 8, color, -1)
            cv2.putText(frame, "Shuttle", (sx + 10, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Frame: {frame_no}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Shot: {current_shot}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"Weak: {weak} ({reason})",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if weak else (0, 255, 0),
            2,
        )

        frame = cv2.resize(frame, (width, height))
        writer.write(frame)

        if current_shot != "Unknown" and current_shot != last_shot and frame_no - last_shot_frame > shot_cooldown:
            rally_id = len(shot_sequence) // 8 + 1
            shot_data = {
                "match_id": "match_001",
                "rally_id": rally_id,
                "shot_number": len(shot_sequence) + 1,
                "frame": frame_no,
                "time_sec": round(timestamp, 3),
                "player_track_id": target_track_id,
                "shot_type": current_shot,
                "confidence": round(current_conf, 3),
                "shuttle_x": sx,
                "shuttle_y": sy,
                "inside_court": is_inside,
                "weak_shot_binary": weak,
                "weak_shot_reason": reason,
            }
            shot_records.append(shot_data)

            if shot_sequence:
                transition_matrix[shot_sequence[-1]][current_shot] += 1

            shot_sequence.append(current_shot)
            last_shot = current_shot
            last_shot_frame = frame_no

        frame_records.append(
            {
                "frame": frame_no,
                "time_sec": round(timestamp, 3),
                "player_track_id": target_track_id,
                "detected_shot": current_shot,
                "shuttle_x": sx,
                "shuttle_y": sy,
                "inside_court": is_inside,
                "weak_shot_binary": weak,
                "weak_shot_reason": reason,
            }
        )

        shuttle_records.append(
            {
                "frame": frame_no,
                "time_sec": round(timestamp, 3),
                "shuttle_x": sx,
                "shuttle_y": sy,
                "inside_court": is_inside,
            }
        )

    cap.release()

    if writer is not None:
        writer.release()
        
        browser_video = OUTPUTS_DIR / f"{job_id}_output_browser.mp4"

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        ffmpeg_cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(output_video),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(browser_video),
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("FFmpeg conversion failed:", result.stderr, flush=True)
            browser_video = output_video

    cv2.destroyAllWindows()

    print(f"Saved output video: {output_video}", flush=True)

    if output_video.exists():
        print(f"Output video size: {output_video.stat().st_size} bytes", flush=True)

    output_paths = {
        "shot_csv": OUTPUTS_DIR / f"{job_id}_shots.csv",
        "frame_csv": OUTPUTS_DIR / f"{job_id}_frames.csv",
        "shuttle_analysis_csv": OUTPUTS_DIR / f"{job_id}_shuttle_analysis.csv",
        "transition_csv": OUTPUTS_DIR / f"{job_id}_tactical_transition_matrix.csv",
        "m2_json": OUTPUTS_DIR / f"{job_id}_m2.json",
        "main_json": OUTPUTS_DIR / f"{job_id}_main_tactical_output.json",
        "frame_json": OUTPUTS_DIR / f"{job_id}_frame_level_output.json",
        "shot_json": OUTPUTS_DIR / f"{job_id}_shot_level_output.json",
        "shuttle_json": OUTPUTS_DIR / f"{job_id}_shuttle_trajectory_final.json",
        "transition_json": OUTPUTS_DIR / f"{job_id}_tactical_transition_matrix.json",
    }

    tactical_summary = build_tactical_outputs(
        job_id=job_id,
        frame_records=frame_records,
        shot_records=shot_records,
        shuttle_records=shuttle_records,
        transition_matrix=transition_matrix,
        output_paths=output_paths,
    )

    return JSONResponse(
        {
            "job_id": job_id,
            "message": "Analysis completed.",
            "processed_frames": frame_no,
            "video": f"/outputs/{browser_video.name}",            
            "shot_csv": f"/outputs/{job_id}_shots.csv",
            "frame_csv": f"/outputs/{job_id}_frames.csv",
            "shuttle_analysis_csv": f"/outputs/{job_id}_shuttle_analysis.csv",
            "transition_csv": f"/outputs/{job_id}_tactical_transition_matrix.csv",
            "transition_matrix": {k: dict(v) for k, v in transition_matrix.items()},
            "m2_json": f"/outputs/{job_id}_m2.json",
            "main_json": f"/outputs/{job_id}_main_tactical_output.json",
            "frame_json": f"/outputs/{job_id}_frame_level_output.json",
            "shot_json": f"/outputs/{job_id}_shot_level_output.json",
            "shuttle_json": f"/outputs/{job_id}_shuttle_trajectory_final.json",
            "transition_json": f"/outputs/{job_id}_tactical_transition_matrix.json",
            "total_shots": tactical_summary["total_shots"],
            "shot_distribution": tactical_summary["shot_distribution"],
            "shots": shot_records[:20],
        }
    )
