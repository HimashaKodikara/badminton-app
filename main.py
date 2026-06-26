# ============================================================
# FULL FINAL COLAB CODE - BADMINTON SHOT + WEAK SHOT ANALYZER
# ============================================================

# ============================================================
# IMPORTS
# ============================================================
import os
# Force CPU-only execution for this FastAPI app and child processes.
# This prevents GPU usage which can cause conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
from fastapi import Form

# ============================================================
# PATH CONFIG - LOCAL CPU VERSION
# ============================================================
# Folder structure expected:
# project/
#   main.py
#   TrackNetV3/
#     predict.py
#     ckpts/TrackNet_best.pt
#     ckpts/InpaintNet_best.pt
#   models/
#     player_best.pt
#     court_best.pt
#   uploads/
#   outputs/
#   static/
#
# Optional: set BADMINTON_APP_BASE to your project root.
BASE = Path(os.getenv("BADMINTON_APP_BASE", Path(__file__).resolve().parent))

TRACKNET_DIR = BASE / "TrackNetV3"
MODELS_DIR = BASE / "models"
UPLOADS_DIR = BASE / "uploads"
OUTPUTS_DIR = BASE / "outputs"
STATIC_DIR = BASE / "static"

PLAYER_MODEL_PATH = Path(os.getenv("PLAYER_MODEL_PATH", MODELS_DIR / "player_best.pt"))
COURT_MODEL_PATH = Path(os.getenv("COURT_MODEL_PATH", MODELS_DIR / "court_best.pt"))
TRACKNET_PT = Path(os.getenv("TRACKNET_PT", TRACKNET_DIR / "ckpts" / "TrackNet_best.pt"))
INPAINT_PT = Path(os.getenv("INPAINT_PT", TRACKNET_DIR / "ckpts" / "InpaintNet_best.pt"))

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# CPU only. Do not change this to 0 unless you want GPU execution.
YOLO_DEVICE = "cpu"

# For full video, set MAX_FRAMES = None.
MAX_FRAMES = 600

# Court model is expensive on CPU. Detect court every N frames and reuse last court points.
COURT_DETECT_INTERVAL = 10
RUN_YOLO_ANALYSIS = True

app = FastAPI()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# ============================================================
# CONFIG - BADMINTON-SPECIFIC SETTINGS
# ============================================================
# All possible badminton shot types that can be detected
SHOT_CLASSES = [
    "BackHand",
    "ForeHand",
    "Lift",
    "NetShot",
    "ReadyPosition",
    "Service",
    "Smash",
]

# Shot types where the player actually hits the shuttle (excludes ReadyPosition)
HIT_SHOT_CLASSES = [
    "BackHand",
    "ForeHand",
    "Lift",
    "NetShot",
    "Service",
    "Smash",
]

TACTICAL_SHOT_CLASSES = [
    "BackHand",
    "ForeHand",
    "Lift",
    "NetShot",
    "Service",
    "Smash",
    "DropShot",
    "Clear",
]

# Minimum time between consecutive weak shot detections (in seconds)
WEAK_SHOT_COOLDOWN_SEC = 1.2
# Court keypoint indexes that indicate the net position
NET_KEYPOINT_INDEXES = [11,12]
# Maximum distance (pixels) shuttle can travel to be considered a weak shot
WEAK_MAX_TRAVEL_DISTANCE_PX = 160
# Tolerance zone around the net line (in pixels)
NET_TOLERANCE = 20

# Visual offset to adjust net line detection based on your court calibration
NET_Y_OFFSET = -260


# Pairs of court keypoints that form the court lines (for drawing the court)
COURT_LINE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 7), (7, 10), (10, 12), (12, 15),
    (4, 6), (6, 9), (9, 11), (11, 14), (14, 16), (16, 21),
    (18, 19), (19, 20), (20, 21),
    (5, 17), (17, 18),
]



# Keypoint indexes that define the left side boundary of the court
LEFT_COURT_POINTS = [17, 15, 12, 10, 7, 5, 0]
# Keypoint indexes that define the right side boundary of the court
RIGHT_COURT_POINTS = [21, 16, 14, 11, 9, 6, 4]

# ============================================================
# HELPER FUNCTIONS - UTILITY FUNCTIONS FOR DETECTION & TRACKING
# ============================================================
# Calculate Intersection over Union (IoU) between two bounding boxes
# This helps match detected players across frames for tracking
def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-6)

# Check if a detected class is a valid badminton shot
def is_shot_class(name):
    return normalize_shot_name(name) in SHOT_CLASSES


# Select the player on the opposite side of the net (usually the top/far player)
# We focus on the player on the other side because that's who we want to analyze
def select_other_side_player(detections, frame_h, court_points=None):
    best = None
    best_score = -1

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        # Filter to upper half of frame (far/opposite player)
        if cy > frame_h * 0.55:
            continue

        # If court points are available, confirm player center is inside court bounds
        if court_points is not None:
            left_x, right_x = court_left_right_bounds(cy, court_points)
            if left_x is not None and right_x is not None:
                if cx < left_x or cx > right_x:
                    continue  # Skip players outside court left/right bounds

        upper_score = 1 - (cy / frame_h)
        size_score = area / (frame_h * frame_h)
        score = upper_score * 0.7 + size_score * 0.3

        if score > best_score:
            best_score = score
            best = det

    return best

# Get shuttle (badminton) position for a given frame from TrackNet predictions
# Returns: x, y coordinates and visibility score (0=not visible, 1=visible)
def get_shuttle(shuttle_df, frame_no):
    idx = frame_no - 1

    # Return None if frame is out of range
    if idx < 0 or idx >= len(shuttle_df):
        return None, None, 0

    row = shuttle_df.iloc[idx]
    # Create case-insensitive column name mapping
    cols = {c.lower(): c for c in shuttle_df.columns}

    # Find position and visibility columns
    x_col = cols.get("x")
    y_col = cols.get("y")
    v_col = cols.get("visibility")

    # Ensure required columns exist
    if not x_col or not y_col:
        return None, None, 0

    x = row[x_col]
    y = row[y_col]
    vis = row[v_col] if v_col else 1

    # Return None if shuttle is not detected (visibility=0) or coordinates are missing
    if pd.isna(x) or pd.isna(y) or vis == 0:
        return None, None, 0

    return int(x), int(y), int(vis)


# Detect and draw the badminton court lines on the frame using YOLO model
# Returns: frame with court drawn + detected court keypoints
def draw_court(frame, court_model):
    # Run YOLO detection to find court keypoints
    result = court_model.predict(
        source=frame,
        imgsz=960,
        conf=0.10,  # Low confidence threshold to catch all court features
        device=YOLO_DEVICE,
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

def court_left_right_bounds(y, court_points):
    if y is None or court_points is None:
        return None, None

    left_points = valid_side(court_points, LEFT_COURT_POINTS)
    right_points = valid_side(court_points, RIGHT_COURT_POINTS)

    if len(left_points) < 2 or len(right_points) < 2:
        return None, None

    left_x = boundary_x(left_points, y)
    right_x = boundary_x(right_points, y)

    if left_x is None or right_x is None:
        return None, None

    if left_x > right_x:
        left_x, right_x = right_x, left_x

    return left_x, right_x

def court_mid_y(court_points):
    if court_points is None:
        return None

    valid_points = [(float(x), float(y)) for x, y in court_points if x > 0 and y > 0]

    if not valid_points:
        return None

    y_values = [p[1] for p in valid_points]
    return (min(y_values) + max(y_values)) / 2


# Calculate the Y position of the net line from court keypoints
# This helps determine which side of the court the shuttle is on
def get_net_y(court_points):
    if court_points is None:
        return None

    y_values = []

    # Get Y coordinates of net keypoints
    for i in NET_KEYPOINT_INDEXES:
        if i < len(court_points):
            x, y = court_points[i]
            if x > 0 and y > 0:
                y_values.append(float(y))

    # Use average of net keypoints, or court midpoint as fallback
    if len(y_values) < 2:
        net_y = court_mid_y(court_points)
    else:
        net_y = sum(y_values) / len(y_values)

    # Apply offset for visual calibration
    return net_y + NET_Y_OFFSET


# Determine which side of the court the shuttle is on (top/bottom/net area)
def get_shuttle_side(sx, sy, court_points):
    if sx is None or sy is None or court_points is None:
        return "unknown"
    net_y = get_net_y(court_points)
    if net_y is None:
        # Fallback: use raw frame height midpoint
        return "unknown"
    # "top" = smaller Y values (upper frame), "bottom" = larger Y
    if sy < net_y - NET_TOLERANCE:
        return "top"
    elif sy > net_y + NET_TOLERANCE:
        return "bottom"
    else:
        return "net_area"

def opposite_court_side(side):
    if side == "top":
        return "bottom"
    if side == "bottom":
        return "top"
    return "unknown"


# Calculate how far the shuttle has traveled since a hit attempt
# Used to detect weak shots (short travel distance)
def shuttle_travel_distance_from_attempt(sx, sy, attempt):
    if sx is None or sy is None or attempt is None:
        return None

    # Get the starting position where the player hit the shuttle
    start_x = attempt.get("start_x")
    start_y = attempt.get("start_y")

    if start_x is None or start_y is None:
        return None

    # Euclidean distance formula
    return ((sx - start_x) ** 2 + (sy - start_y) ** 2) ** 0.5



    # Shade the zone between net and clearance line (semi-transparent orange band)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y), (w, clearance_y), (0, 100, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    return frame
def draw_net_line(frame, court_points):
    net_y = get_net_y(court_points)

    if net_y is None:
        return frame

    h, w = frame.shape[:2]
    y = int(net_y)

    cv2.line(frame, (0, y), (w, y), (0, 255, 255), 3)

    cv2.putText(
        frame,
        "NET LINE",
        (20, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return frame



def is_shuttle_drop_event(shuttle_history):
    valid_points = [
        p for p in shuttle_history
        if p.get("x") is not None and p.get("y") is not None and p.get("visibility", 1) != 0
    ]

    if len(valid_points) < 5:
        return False

    last_points = valid_points[-5:]
    y_values = [p["y"] for p in last_points]

    downward_count = 0
    for i in range(1, len(y_values)):
        if y_values[i] > y_values[i - 1]:
            downward_count += 1

    total_drop = y_values[-1] - y_values[0]

    return downward_count >= 3 and total_drop > 25

def shuttle_landed_on_floor(shuttle_history, min_frames=5):
    valid_points = [
        p for p in shuttle_history
        if p.get("x") is not None and p.get("y") is not None and p.get("visibility", 1) != 0
    ]

    if len(valid_points) < min_frames:
        return False

    last_points = valid_points[-min_frames:]
    y_values = [p["y"] for p in last_points]

    downward_count = sum(
        1 for i in range(1, len(y_values))
        if y_values[i] > y_values[i - 1]
    )

    total_drop = y_values[-1] - y_values[0]

    x_move = abs(last_points[-1]["x"] - last_points[-2]["x"])
    y_move = abs(last_points[-1]["y"] - last_points[-2]["y"])

    return (
        downward_count >= 3
        and total_drop > 25
        and x_move < 8
        and y_move < 8
    )
def player_hit_shuttle(current_shot):
    return current_shot in HIT_SHOT_CLASSES

WEAK_SHOT_EXCLUDED_TYPES = ["Service", "ReadyPosition"]

def shuttle_near_net(sy, court_points, margin=60):
    """True only if shuttle is physically close to the net line."""
    if sy is None or court_points is None:
        return False
    net_y = get_net_y(court_points)
    if net_y is None:
        return False
    return abs(sy - net_y) < margin

def shuttle_near_player(sx, sy, player_box, max_distance=130):
    if sx is None or sy is None or player_box is None:
        return False

    x1, y1, x2, y2 = player_box
    px = (x1 + x2) / 2
    py = y1 + (y2 - y1) * 0.35

    distance = ((sx - px) ** 2 + (sy - py) ** 2) ** 0.5
    return distance < max_distance


def shuttle_direction_or_speed_changed(shuttle_history):
    valid_points = [
        p for p in shuttle_history
        if p.get("x") is not None and p.get("y") is not None and p.get("visibility", 1) != 0
    ]

    if len(valid_points) < 6:
        return False

    p1 = valid_points[-6]
    p2 = valid_points[-4]
    p3 = valid_points[-2]

    vx1 = p2["x"] - p1["x"]
    vy1 = p2["y"] - p1["y"]
    vx2 = p3["x"] - p2["x"]
    vy2 = p3["y"] - p2["y"]

    speed1 = (vx1 ** 2 + vy1 ** 2) ** 0.5
    speed2 = (vx2 ** 2 + vy2 ** 2) ** 0.5

    if speed1 < 5 or speed2 < 5:
        return False

    dot = vx1 * vx2 + vy1 * vy2
    cosine = dot / (speed1 * speed2 + 1e-6)
    speed_change = abs(speed2 - speed1)

    return cosine < 0.55 or speed_change > 20

# ============================================================
# SHOT CLASSIFICATION FIX - FROM INFERENCING POSE NOTEBOOK STYLE
# ============================================================
# In the notebook, the custom badminton model is used for frame-level
# shot classification, while the baseline pose model is used for cleaner
# keypoints/player tracking. These helpers make your app use the custom
# model output more safely.

SHOT_CONF_THRES = 0.35          # Same idea as notebook CONF_THRES
SHOT_SMOOTHING_WINDOW = 7       # Majority vote over recent frames
SHOT_EVENT_COOLDOWN_FRAMES = 12 # Prevent duplicate shot events

def extract_best_shot_from_result(result):
    """Return best valid shot class from a YOLO result.

    The notebook used custom_result.boxes.cls[0], but that can be wrong
    when YOLO returns multiple boxes. This version checks all boxes and
    picks the highest-confidence valid badminton shot class.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return "unknown", 0.0, None

    best_name = "unknown"
    best_conf = 0.0
    best_box = None

    boxes = result.boxes
    names = result.names

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        raw_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        shot_name = normalize_shot_name(raw_name)

        if shot_name not in SHOT_CLASSES:
            continue
        if conf < SHOT_CONF_THRES:
            continue

        if conf > best_conf:
            best_name = shot_name
            best_conf = conf
            best_box = boxes.xyxy[i].cpu().numpy().tolist()

    return best_name, best_conf, best_box


def predict_frame_shot(player_model, frame):
    """Run custom shot model like the notebook and return clean shot output."""
    result = player_model.predict(
        source=frame,
        conf=SHOT_CONF_THRES,
        device=YOLO_DEVICE,
        verbose=False,
    )[0]
    return extract_best_shot_from_result(result)


def smooth_shot_prediction(shot_history):
    """Majority vote to reduce frame-by-frame wrong labels."""
    valid = [s for s in shot_history if s in SHOT_CLASSES and s != "ReadyPosition"]
    if not valid:
        return "unknown"
    return Counter(valid).most_common(1)[0][0]


def valid_shot_event(current_shot, sx, sy, player_box, shuttle_history, shot_conf=0.0):
    """Accept a shot only when model confidence + shuttle-contact evidence agree.

    Your old version used only shot class + shuttle_near_player. This caused
    wrong repeated shots. This version requires confidence and either shuttle
    near player or shuttle direction/speed change.
    """
    current_shot = normalize_shot_name(current_shot)

    if current_shot not in HIT_SHOT_CLASSES:
        return False

    if shot_conf < SHOT_CONF_THRES:
        return False

    near_player = shuttle_near_player(sx, sy, player_box, max_distance=230)
    trajectory_changed = shuttle_direction_or_speed_changed(shuttle_history)

    return near_player or trajectory_changed

def should_record_new_shot(frame_no, last_shot_frame, current_shot, last_recorded_shot):
    """Avoid saving the same predicted shot in many consecutive frames."""
    if current_shot not in HIT_SHOT_CLASSES:
        return False

    if last_shot_frame is None:
        return True

    gap = frame_no - last_shot_frame

    if current_shot == last_recorded_shot and gap < SHOT_EVENT_COOLDOWN_FRAMES:
        return False

    return gap >= max(4, SHOT_EVENT_COOLDOWN_FRAMES // 2)

def hit_attempt_event(current_shot, sx, sy, player_box):
    # Used only for weak-shot detection.
    return (
        current_shot in HIT_SHOT_CLASSES
        and shuttle_near_player(sx, sy, player_box, max_distance=220)
    )


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


# Standardize shot names to match SHOT_CLASSES format
# Handles variations in naming (e.g., 'serve' -> 'Service')
def normalize_shot_name(name):
    raw = str(name).strip()

    key = (
        raw.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    mapping = {
        "readyposition": "ReadyPosition",
        "ready": "ReadyPosition",

        "service": "Service",
        "serve": "Service",
        "serviceshot": "Service",
        "serveshot": "Service",

        "smash": "Smash",
        "smashshot": "Smash",

        "backhand": "BackHand",
        "backhandshot": "BackHand",

        "forehand": "ForeHand",
        "forehandshot": "ForeHand",

        "lift": "Lift",
        "liftshot": "Lift",

        "netshot": "NetShot",
        "net": "NetShot",

        # These are not YOLO classes, only tactical output labels
        "dropshot": "DropShot",
        "drop": "DropShot",
        "clear": "Clear",
    }

    return mapping.get(key, raw)


def infer_tactical_shot(shot_type, trajectory):
    shot_type = normalize_shot_name(shot_type)

    # Only infer DropShot / Clear from ForeHand
    if shot_type != "ForeHand":
        return shot_type

    start_pos, end_pos = get_start_end_position(trajectory)

    if (
        start_pos["x"] is None or start_pos["y"] is None or
        end_pos["x"] is None or end_pos["y"] is None
    ):
        return "ForeHand"

    travel_y = end_pos["y"] - start_pos["y"]

    travel_distance = (
        (end_pos["x"] - start_pos["x"]) ** 2 +
        (end_pos["y"] - start_pos["y"]) ** 2
    ) ** 0.5

    landing_near_net = abs(travel_y) < 120 and travel_distance < 180
    landing_deep_court = travel_distance > 250

    if landing_near_net:
        return "DropShot"

    if landing_deep_court:
        return "Clear"

    return "ForeHand"

def get_court_status(value):
    text = str(value).lower()
    if value is True or value == 1 or text == "true":
        return "in"
    if value is False or value == 0 or text == "false":
        return "out"
    return "unknown"


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

# ============================================================
# OUTPUT BUILDER - GENERATE CSV AND JSON OUTPUT FILES
# ============================================================
# This function creates all analysis output files (CSVs and JSONs)
def build_tactical_outputs(
     job_id,
    frame_records,
    shot_records,
    weak_shot_records,
    shuttle_records,
    transition_matrix,
    rally_sequences,
    output_paths,
    match_id="match_001",
    player_id="player_01",
):
    frame_df = pd.DataFrame(frame_records)
    shot_df = pd.DataFrame(shot_records)
    weak_shot_df = pd.DataFrame(weak_shot_records, columns=[
    "match_id",
    "rally_id",
    "weak_shot_number",
    "frame",
    "time_sec",
    "player_track_id",
    "attempted_shot_type",
    "confidence",
    "shuttle_x",
    "shuttle_y",
    "inside_court",
    "net_y",
    "weak_shot_binary",
    "weak_shot_reason",
])
    shuttle_df = pd.DataFrame(shuttle_records)

    # Only produce JSON outputs — CSV outputs removed per request
    frame_df.to_json(output_paths["frame_json"], orient="records", indent=2)
    shot_df.to_json(output_paths["shot_json"], orient="records", indent=2)
    weak_shot_df.to_json(output_paths["weak_shot_json"], orient="records", indent=2)
    shuttle_df.to_json(output_paths["shuttle_json"], orient="records", indent=2)

    # Build transition matrix and write as JSON only
    transition_df = pd.DataFrame(0, index=TACTICAL_SHOT_CLASSES, columns=TACTICAL_SHOT_CLASSES)    
    for from_shot, next_dict in transition_matrix.items():
        for to_shot, count in next_dict.items():
            if from_shot in transition_df.index and to_shot in transition_df.columns:
                transition_df.loc[from_shot, to_shot] = count

    transition_df.to_json(output_paths["transition_json"], orient="records", indent=2)

    
    normal_total_shots = len(shot_records)
    weak_total_shots = len(weak_shot_records)

    if shot_df.empty or "shot_type" not in shot_df.columns:
        shot_counts = Counter()
        average_rally_length = 0
    else:
        tactical_shot_type_list = []

        for _, row in shot_df.iterrows():
            raw_type = row.get("shot_type", "unknown")
            shot_frame = int(row.get("frame", 0))
            trajectory = get_shot_trajectory(shuttle_df, shot_frame)
            tactical_type = infer_tactical_shot(raw_type, trajectory)

            if tactical_type != "unknown":
                tactical_shot_type_list.append(tactical_type)

        shot_counts = Counter(tactical_shot_type_list)

        average_rally_length = (
            shot_df.groupby("rally_id").size().mean()
            if "rally_id" in shot_df.columns
            else 0
        )

        if pd.isna(average_rally_length):
            average_rally_length = 0

    percentage_distribution = (
        {
            key: round(value / normal_total_shots * 100, 1)
            for key, value in shot_counts.items()
        }
        if normal_total_shots
        else {}
    )


    current_video_profile = {
        "normal_shot_distribution": dict(shot_counts),
        "normal_shot_distribution_percentage_raw_classes": percentage_distribution,
        "average_rally_length": round(float(average_rally_length), 2),
        "normal_total_shots": normal_total_shots,
        "weak_total_shots": weak_total_shots,
        "total_including_weak": normal_total_shots + weak_total_shots,
        "preferred_zones": [],
    }

    shots = []
    for i, row in shot_df.iterrows():
        raw_shot_type = row.get("shot_type", "unknown")

        shot_frame = int(row.get("frame", 0))
        trajectory = get_shot_trajectory(shuttle_df, shot_frame)
        # NEW
        start_position, end_position = get_start_end_position(trajectory)
        court_status = get_court_status(row.get("inside_court"))
        shot_type = infer_tactical_shot(raw_shot_type, trajectory)

        current_frequency = shot_counts.get(shot_type, 0)

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
                "is_weak": False,
                "weakness_score": 0.0,
                "reason": "normal",
                "court_status": court_status,
                "zone": get_zone(row, shot_df),
            },
            "tactical_analysis": {
                "is_effective": True,
                "reason": "normal",
            },
        })

    weak_shots = []
    for i, row in weak_shot_df.iterrows():
        weak_shots.append({
            "weak_shot_id": i + 1,
            "rally_id": int(row.get("rally_id", 1)),
            "attempted_shot_type": normalize_shot_name(row.get("attempted_shot_type", "unknown")),
            "frame": int(row.get("frame", 0)),
            "timestamp": clean_value(row.get("time_sec")),
            "shuttle_position": {
                "x": clean_value(row.get("shuttle_x")),
                "y": clean_value(row.get("shuttle_y")),
            },
            "weak_shot": {
                "is_weak": True,
                "weakness_score": 0.82,
                "reason": row.get("weak_shot_reason", "weak_shot_detected"),
                "court_status": get_court_status(row.get("inside_court")),
            },
        })

    rallies = []
    if not shot_df.empty and "rally_id" in shot_df.columns:
        for rally_id, group in shot_df.groupby("rally_id"):
            length = len(group)
            weak_count = 0
            if not weak_shot_df.empty and "rally_id" in weak_shot_df.columns:
                weak_count = int((weak_shot_df["rally_id"] == rally_id).sum())

            if length <= 5:
                tempo = "fast"
            elif length <= 10:
                tempo = "medium"
            else:
                tempo = "slow"

            rallies.append({
                "rally_id": int(rally_id),
                "normal_shot_length": length,
                "weak_shots_count": weak_count,
                "average_rally_length_from_video": round(float(average_rally_length), 2),
                "deviation_from_video_average": round(length - float(average_rally_length), 2),
                "tempo": tempo,
                "fatigue_impact": "high" if length >= 8 else "low",
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

    dominant_shot = shot_counts.most_common(1)[0][0] if shot_counts else "unknown"
    total_transitions = sum(current_transition_matrix.values())
    predictability_score = 0.0
    if total_transitions > 0:
        predictability_score = round(max(current_transition_matrix.values()) / total_transitions, 2)

    # ============================================================
    # RALLY-BY-RALLY SHOT PATTERN ANALYSIS
    # ============================================================
    # rally_sequences format: {1: ["Service", "ForeHand", "NetShot"], ...}
    # This keeps every rally separate, so transitions do not cross rally boundaries.
    if rally_sequences is None:
        rally_sequences = defaultdict(list)
        for shot in shot_records:
            derived_rally_id = int(shot.get("rally_id", 1))
            derived_shot_type = normalize_shot_name(shot.get("shot_type", "Unknown"))
            rally_sequences[derived_rally_id].append(derived_shot_type)

    rally_patterns = []

    for rally_id, sequence in sorted(rally_sequences.items(), key=lambda item: int(item[0])):
        normalized_sequence = [normalize_shot_name(s) for s in sequence if normalize_shot_name(s) != "Unknown"]

        rally_patterns.append({
            "rally_id": int(rally_id),
            "shot_pattern": normalized_sequence,
            "pattern_string": " → ".join(normalized_sequence),
            "pattern_length": len(normalized_sequence),
        })

    pattern_counts = Counter(
        " → ".join([normalize_shot_name(s) for s in sequence if normalize_shot_name(s) != "Unknown"])
        for sequence in rally_sequences.values()
        if len([s for s in sequence if normalize_shot_name(s) != "Unknown"]) > 1
    )

    total_patterns = sum(pattern_counts.values())

    pattern_scores = []
    for pattern, count in pattern_counts.items():
        pattern_scores.append({
            "pattern": pattern,
            "count": int(count),
            "score": round(count / total_patterns, 3) if total_patterns else 0,
        })

    pattern_scores = sorted(pattern_scores, key=lambda x: x["score"], reverse=True)

    main_output = {
        "match_id": match_id,
        "player_id": player_id,
        "job_id": job_id,
        "current_video_profile": current_video_profile,
        "normal_shots": shots,
        "weak_shots": weak_shots,
        "rallies": rallies,
        "rally_patterns": rally_patterns,
        "pattern_scores": pattern_scores,
        "tactical_patterns": {
            "dominant_shot": dominant_shot,
            "predictability_score": predictability_score,
        },
        "transition_matrix": current_transition_matrix,
        "sequence_deviation": sequence_deviation,
    }

    with open(output_paths["main_json"], "w", encoding="utf-8") as f:
        json.dump(main_output, f, indent=2)

    m2_output = {
        "match_id": match_id,
        "player_id": player_id,
        "job_id": job_id,
        "processed_frames": len(frame_records),
        "max_frames_used": MAX_FRAMES,
        "normal_total_shots": normal_total_shots,
        "weak_total_shots": weak_total_shots,
        "total_including_weak": normal_total_shots + weak_total_shots,
        "normal_shot_distribution": percentage_distribution,
        "transition_matrix": {k: dict(v) for k, v in transition_matrix.items()},
        "rally_patterns": rally_patterns,
        "pattern_scores": pattern_scores,
        "normal_shots": shot_records,
        "weak_shots": weak_shot_records,
    }

    with open(output_paths["m2_json"], "w", encoding="utf-8") as f:
        json.dump(m2_output, f, indent=2)

    return {
        "normal_total_shots": normal_total_shots,
        "weak_total_shots": weak_total_shots,
        "total_including_weak": normal_total_shots + weak_total_shots,
        "shot_distribution": percentage_distribution,
        "main_output_preview": main_output,
    }

# ============================================================
# FASTAPI ROUTES - API ENDPOINTS FOR THE WEB APPLICATION
# ============================================================
# Main page endpoint - serves the HTML interface
@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "FastAPI badminton analyzer is running."})


# Main analysis endpoint - processes uploaded video and returns analysis results
@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    match_id: str = Form("match_001"),
    player_name: str = Form("player_01"),
):
    # Generate unique ID for this analysis job
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
    tracknet_env = os.environ.copy()
    tracknet_env["CUDA_VISIBLE_DEVICES"] = "-1"

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
        env=tracknet_env,
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

    # Save TrackNet shuttle track as JSON for downstream analysis (CSV removed)
    output_shuttle_json = OUTPUTS_DIR / f"{job_id}_shuttle_trajectory_final.json"
    shuttle_df.to_json(output_shuttle_json, orient="records", indent=2)

    if not RUN_YOLO_ANALYSIS:
        return JSONResponse(
            {
                "job_id": job_id,
                "message": "TrackNet completed successfully. YOLO analysis is disabled.",
                "shuttle_json": f"/outputs/{job_id}_shuttle_trajectory_final.json",
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
                "shuttle_json": f"/outputs/{job_id}_shuttle_trajectory_final.json",
            },
            status_code=500,
        )

    # Load pre-trained YOLO models for player and court detection
    print("Loading YOLO models...", flush=True)

    player_model = YOLO(str(PLAYER_MODEL_PATH))  # Detects players and shot types
    court_model = YOLO(str(COURT_MODEL_PATH))    # Detects court keypoints

    print("Loaded player model path:", PLAYER_MODEL_PATH, flush=True)
    print("YOLO class names:", player_model.names, flush=True)
    # Initialize tracker for identifying the same player across frames
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.7, max_cosine_distance=0.3)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        width = 1280
        height = 720

    print(f"Video Info -> FPS: {fps}, Width: {width}, Height: {height}", flush=True)

    output_video = OUTPUTS_DIR / f"{job_id}_output.mp4"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    print("Writer opened:", writer.isOpened(), flush=True)

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_video}")

    # Data structures to store analysis results
    frame_records = []  # Every frame's data
    shot_records = []  # Normal shots detected
    weak_shot_records = []  # Shots classified as weak
    shuttle_records = []  # Shuttle position tracking
    shot_sequence = []  # Sequence of shots inside the current rally only
    rally_sequences = defaultdict(list)  # Rally-wise shot patterns: {rally_id: [shot1, shot2, ...]}
    transition_matrix = defaultdict(lambda: defaultdict(int))  # Shot transition counts inside rallies only

    # Shot detection variables
    last_shot = None
    last_shot_frame = -999
    shot_cooldown = int(fps * 0.4)  
    target_track_id = None  # ID of the player being tracked
    shuttle_history = deque(maxlen=15)  # Keep last 15 shuttle positions
    shot_class_history = deque(maxlen=7)  # Stabilize shot prediction with history
    last_court_points = None  # Cache court keypoints for efficiency
    frame_no = 0

    # Weak shot detection variables
    last_weak_frame = -999
    weak_shot_cooldown = int(fps * WEAK_SHOT_COOLDOWN_SEC)  
    pending_hit_attempts = []  # Track potential weak shot attempts
    last_hit_attempt_frame = -999
    hit_attempt_cooldown = int(fps * 2.5)  # Minimum time between hit attempts
    weak_evaluation_frames = int(fps * 0.9)   # ~27 frames at 30fps, was 75
    MAX_ATTEMPT_AGE_FRAMES = int(fps * 2.0)   # discard stale attempts

    # Rally tracking - groups consecutive shots between long pauses
    current_rally_id = 1


    # Output rally numbering
    recorded_rally_ids = {}
    next_recorded_rally_id = 1

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
        frame = draw_net_line(frame, court_points)


        player_result = player_model.predict(frame, conf=0.35, device=YOLO_DEVICE, verbose=False)[0]

        all_detections = []
        tracker_detections = []

        if player_result.boxes is not None:
            for box in player_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = normalize_shot_name(player_model.names[cls_id])

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
        selected = select_other_side_player(candidates, height, court_points=court_points)
        current_shot = "Unknown"
        current_conf = 0
        player_box = None

        if selected:
            current_shot = selected["class_name"]
            current_conf = selected["conf"]
            player_box = selected["box"]

        # Stabilize shot prediction using history to reduce false detections
        # Use the most common shot type in recent frames (vote-based approach)
        shot_class_history.append(current_shot)

        if len(shot_class_history) >= 3:
            shot_counter = Counter(shot_class_history)

            stable_shot = shot_counter.most_common(1)[0][0]
            stable_count = shot_counter[stable_shot]

            min_votes = 1 if stable_shot in ("Service", "BackHand") else 2

            if stable_count >= min_votes:
                current_shot = stable_shot
            else:
                current_shot = "Unknown"

        tracks = tracker.update_tracks(tracker_detections, frame=frame)

        # Draw bounding box around detected player and show their shot type
        if player_box is not None:
            x1, y1, x2, y2 = map(int, player_box)

            # Draw green rectangle around player
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Display shot type and confidence score
            label = f"{current_shot} {current_conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )



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

        # Check if shuttle is within court boundaries
        is_inside = inside_court(sx, sy, court_points)
        # Track shuttle position history for analyzing motion
        shuttle_history.append({"frame": frame_no, "x": sx, "y": sy, "visibility": sv})

        # Determine which side of the net the shuttle is on
        shuttle_side = get_shuttle_side(sx, sy, court_points)
        # Get Y position of the net line
        net_y = get_net_y(court_points)
        # Detect if shuttle is falling/dropping
        drop_event = is_shuttle_drop_event(shuttle_history)
        
        
        # We're tracking the top player (top side of court)
        focused_player_side = "top"
      
        RALLY_GAP_SECONDS = 2.0
        RALLY_GAP_FRAMES = int(fps * RALLY_GAP_SECONDS)
        # Rally finished when shuttle lands


        # Weak shot detection setup
        weak = 0  # Binary flag: is this a weak shot?
        weak_event = False  # Did we detect a weak shot this frame?
        weak_attempt = None  # The attempt that resulted in a weak shot
        reason = "Normal"  # Reason for weak shot classification

        WEAK_SHOT_EXCLUDED_TYPES = ["Service", "ReadyPosition"]
        # Register a potential weak shot attempt
        # This starts tracking when a player hits the shuttle near the net
        if (
            not pending_hit_attempts
            and hit_attempt_event(current_shot, sx, sy, player_box)
            and current_shot not in WEAK_SHOT_EXCLUDED_TYPES
            and (
                shuttle_side == focused_player_side
                or (shuttle_side == "net_area" and shuttle_near_net(sy, court_points, margin=60))
            )
            and net_y is not None
            and frame_no - last_hit_attempt_frame > hit_attempt_cooldown
        ):
            pending_hit_attempts.append({
                "frame": frame_no,
                "time_sec": round(timestamp, 3),
                "shot_type": current_shot,
                "confidence": round(current_conf, 3),
                "player_track_id": target_track_id,
                "hitter_side": focused_player_side,
                "opponent_side": opposite_court_side(focused_player_side),
                "start_x": sx,
                "start_y": sy,
                "crossed_net": False,
            })
            last_hit_attempt_frame = frame_no

        # Evaluate pending weak shot attempts to determine if they were actually weak shots
        remaining_attempts = []

        for attempt in pending_hit_attempts:
            age = frame_no - attempt["frame"]

            # Discard stale attempts unconditionally
            if age > MAX_ATTEMPT_AGE_FRAMES:
                continue  # drop it, don't append to remaining
            
            if shuttle_side == attempt["opponent_side"]:
                attempt["crossed_net"] = True

            if attempt["crossed_net"]:
                continue  # good shot, discard

            if age < weak_evaluation_frames:
                remaining_attempts.append(attempt)
                continue

            # Calculate how far shuttle has traveled since hit
            travel_distance = shuttle_travel_distance_from_attempt(sx, sy, attempt)

            # Check if travel distance is short (characteristic of weak shot)
            short_failed_travel = (
                travel_distance is None
                or travel_distance <= WEAK_MAX_TRAVEL_DISTANCE_PX
            )

            # A weak shot is when:
            # 1. Shuttle is still inside the court
            # 2. Shuttle didn't cross to opponent's side
            # 3. Shuttle didn't travel far (short distance)
            if (
                is_inside is True
                and shuttle_side == attempt["hitter_side"]  # Still on same side
                and short_failed_travel
                and frame_no - last_weak_frame > weak_shot_cooldown  # Avoid double-detection
            ):
                weak = 1
                weak_event = True
                weak_attempt = attempt
                reason = "Weak Shot: Shuttle did not pass the net line and landed inside player's side"
                last_weak_frame = frame_no
                continue

        # Keep only attempts that haven't been evaluated yet
        pending_hit_attempts = remaining_attempts
        # Draw shuttle on video with color coding for different states:
        # Orange = on player's side, Blue = passed net, Red = out of bounds
        if sx is not None and sy is not None:
            # Color definitions
            ORANGE = (0, 165, 255)  # Player side
            BLUE = (255, 0, 0)  # Passed net / opponent side
            RED = (0, 0, 255)  # Out of bounds
            WHITE = (255, 255, 255)  # Unknown

            shuttle_color = ORANGE
            shuttle_status = "PLAYER SIDE"

            # Get court left/right boundaries at shuttle's Y position
            left_x, right_x = court_left_right_bounds(sy, court_points)

            # Check if shuttle is outside left or right court boundary
            outside_left_right = (
                left_x is not None
                and right_x is not None
                and (sx < left_x or sx > right_x)
            )

            # Change color if shuttle passed net to opponent's side
            if shuttle_side == opposite_court_side(focused_player_side):
                shuttle_color = BLUE
                shuttle_status = "PASSED NET"

            # Out-of-bounds has highest priority in coloring
            if outside_left_right:
                shuttle_color = RED
                shuttle_status = "OUT LEFT/RIGHT"

            # White if court boundaries couldn't be calculated
            if left_x is None or right_x is None:
                shuttle_color = WHITE
                shuttle_status = "COURT UNKNOWN"

            # Draw shuttle circle
            cv2.circle(frame, (int(sx), int(sy)), 9, shuttle_color, -1)
            # Display shuttle status label
            cv2.putText(
                frame,
                shuttle_status,
                (int(sx) + 10, int(sy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                shuttle_color,
                2,
            )

        # Display analysis information on the frame (for visual feedback)
        cv2.putText(frame, f"Frame: {frame_no}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracked Player: TOP PLAYER | Shot: {current_shot}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # Show shuttle location info
        cv2.putText(
            frame,
            f"Shuttle Side: {shuttle_side} | Net Y: {net_y:.1f}" if net_y is not None else "Shuttle Side: unknown | Net Y: unknown",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        # Show weak shot status (red if weak, green if normal)
        cv2.putText(
            frame,
            f"Weak: {weak} ({reason})",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if weak else (0, 255, 0),  # Red for weak, green for normal
            2,
        )
        # Show current rally number
        cv2.putText(
            frame,
            f"Rally: {current_rally_id}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        frame = cv2.resize(frame, (width, height))
        writer.write(frame)

        # ====================================================
        # NORMAL SHOT DETECTION - INDEPENDENT FROM WEAK SHOTS
        # ====================================================
        # Check if a valid shot event occurred (player hit shuttle)
        shot_detected = valid_shot_event(
            current_shot=current_shot,
            sx=sx,
            sy=sy,
            player_box=player_box,
            shuttle_history=shuttle_history,
            shot_conf=current_conf,
        )

        # If a shot happens, the rally is still active
        if shot_detected:
            rally_waiting_to_end = False

        # Record the shot if conditions are met
        if (
    shot_detected
    and current_shot != "Unknown"
    and (
        frame_no - last_shot_frame > shot_cooldown
        or current_shot != last_shot
    )
):
             # ← ADD THIS at the top of this block
            if current_shot == "Service":
                pending_hit_attempts.clear()
                last_hit_attempt_frame = -999
            # Check if enough time has passed since the last shot for a new rally
            # REPLACE WITH:
            if shot_records:
                previous_shot_frame = shot_records[-1]["frame"]
                frame_gap = frame_no - previous_shot_frame
                gap_seconds = frame_gap / fps

                # Condition 1: large time gap (clear pause between rallies)
                large_gap = frame_gap > RALLY_GAP_FRAMES

                # Condition 2: shuttle landed AND moderate gap (rally ended naturally)
                shuttle_landed = shuttle_landed_on_floor(shuttle_history)
                natural_end = shuttle_landed and gap_seconds > 1.0

                if large_gap or natural_end:
                    current_rally_id += 1
                    pending_hit_attempts.clear()
                    shot_class_history.clear()   # ← prevent bleed
                    shot_sequence = []           # reset current-rally sequence
                    last_shot = None
                    last_shot_frame = -999
                    print(f"🏸 Rally ended (gap={gap_seconds:.1f}s, landed={shuttle_landed}). New Rally {current_rally_id}")
            if current_rally_id not in recorded_rally_ids:
                recorded_rally_ids[current_rally_id] = next_recorded_rally_id
                next_recorded_rally_id += 1

            rally_id = recorded_rally_ids[current_rally_id]

            shot_data = {
                "match_id": "match_001",
                "rally_id": rally_id,
                "shot_number": len(shot_records) + 1,
                "frame": frame_no,
                "time_sec": round(timestamp, 3),
                "player_track_id": target_track_id,
                "shot_type": current_shot,
                "confidence": round(current_conf, 3),
                "shuttle_x": sx,
                "shuttle_y": sy,
                "inside_court": is_inside,
                "net_y": round(net_y, 2) if net_y is not None else None,
                "weak_shot_binary": 0,
                "weak_shot_reason": "Normal",
            }

            shot_records.append(shot_data)

            # Save shot pattern rally-by-rally
            normalized_current_shot = normalize_shot_name(current_shot)
            rally_sequences[rally_id].append(normalized_current_shot)
            shot_sequence.append(normalized_current_shot)

            # Calculate transition only inside the same rally
            # Example: ForeHand → NetShot, NetShot → Lift
            if len(rally_sequences[rally_id]) >= 2:
                previous_shot = rally_sequences[rally_id][-2]
                transition_matrix[previous_shot][normalized_current_shot] += 1

            last_shot = current_shot
            last_shot_frame = frame_no
        # ====================================================
        # WEAK SHOT DETECTION - SEPARATE OUTPUT ONLY
        # ====================================================
        if weak_event:
            if current_rally_id not in recorded_rally_ids:
                recorded_rally_ids[current_rally_id] = next_recorded_rally_id
                next_recorded_rally_id += 1

            rally_id = recorded_rally_ids[current_rally_id]
            attempted_type = weak_attempt.get("shot_type", "WeakReturn") if weak_attempt else "WeakReturn"

            weak_data = {
                "match_id": "match_001",
                "rally_id": rally_id,
                "weak_shot_number": len(weak_shot_records) + 1,
                "frame": frame_no,
                "time_sec": round(timestamp, 3),
                "player_track_id": weak_attempt.get("player_track_id", target_track_id) if weak_attempt else target_track_id,
                "attempted_shot_type": attempted_type,
                "confidence": weak_attempt.get("confidence", 0.0) if weak_attempt else 0.0,
                "shuttle_x": sx,
                "shuttle_y": sy,
                "inside_court": is_inside,
                "net_y": round(net_y, 2) if net_y is not None else None,
                "weak_shot_binary": 1,
                "weak_shot_reason": reason,
            }

            weak_shot_records.append(weak_data)

        frame_records.append(
                {
                    "frame": frame_no,
                    "time_sec": round(timestamp, 3),
                    "player_track_id": target_track_id,
                    "detected_shot": current_shot,
                    "shuttle_x": sx,
                    "shuttle_y": sy,
                    "inside_court": is_inside,
                    "net_y": round(net_y, 2) if net_y is not None else None,
                    "shuttle_side": shuttle_side,
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
                    "net_y": round(net_y, 2) if net_y is not None else None,
                    "shuttle_side": shuttle_side,
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

    ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if ffmpeg_result.returncode != 0:
        print("FFmpeg conversion failed:", ffmpeg_result.stderr, flush=True)
        browser_video = output_video

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    print(f"Saved output video: {output_video}", flush=True)
    if output_video.exists():
        print(f"Output video size: {output_video.stat().st_size} bytes", flush=True)

    output_paths = {
        # JSON-only outputs (CSV files removed)
        "m2_json": OUTPUTS_DIR / f"{job_id}_m2.json",
        "main_json": OUTPUTS_DIR / f"{job_id}_main_tactical_output.json",
        "frame_json": OUTPUTS_DIR / f"{job_id}_frame_level_output.json",
        "shot_json": OUTPUTS_DIR / f"{job_id}_shot_level_output.json",
        "weak_shot_json": OUTPUTS_DIR / f"{job_id}_weak_shots.json",
        "shuttle_json": OUTPUTS_DIR / f"{job_id}_shuttle_trajectory_final.json",
        "transition_json": OUTPUTS_DIR / f"{job_id}_tactical_transition_matrix.json",
    }

    tactical_summary = build_tactical_outputs(
        job_id=job_id,
        frame_records=frame_records,
        shot_records=shot_records,
        weak_shot_records=weak_shot_records,
        shuttle_records=shuttle_records,
        transition_matrix=transition_matrix,
        rally_sequences=rally_sequences,
        output_paths=output_paths,
        match_id=match_id,
        player_id=player_name,
    )

    return JSONResponse(
        {
            "job_id": job_id,
            "message": "Analysis completed.",
            "processed_frames": frame_no,
            "video": f"/outputs/{browser_video.name}",

            # JSON outputs 
            "m2_json": f"/outputs/{job_id}_m2.json",
            "main_json": f"/outputs/{job_id}_main_tactical_output.json",
            "frame_json": f"/outputs/{job_id}_frame_level_output.json",
            "shot_json": f"/outputs/{job_id}_shot_level_output.json",
            "weak_shot_json": f"/outputs/{job_id}_weak_shots.json",
            "shuttle_json": f"/outputs/{job_id}_shuttle_trajectory_final.json",
            "transition_json": f"/outputs/{job_id}_tactical_transition_matrix.json",

            "transition_matrix": {k: dict(v) for k, v in transition_matrix.items()},
            "rally_patterns": tactical_summary["main_output_preview"].get("rally_patterns", []),
            "pattern_scores": tactical_summary["main_output_preview"].get("pattern_scores", []),
            "normal_total_shots": tactical_summary["normal_total_shots"],
            "weak_total_shots": tactical_summary["weak_total_shots"],
            "total_including_weak": tactical_summary["total_including_weak"],
            "shot_distribution": tactical_summary["shot_distribution"],
            "shots": shot_records[:20],
            "weak_shots": weak_shot_records[:20],
        }
    )


@app.post("/reanalyze")
async def reanalyze(job_id: str):
    """
    Re-runs only the output builder using previously saved JSON files.
    Skips TrackNet and YOLO entirely. Use this to tweak output logic
    without reprocessing the video.
    """
    # Load previously saved shot and weak shot records
    main_json_path = OUTPUTS_DIR / f"{job_id}_main_tactical_output.json"
    m2_json_path   = OUTPUTS_DIR / f"{job_id}_m2.json"

    if not m2_json_path.exists():
        return JSONResponse(
            {"error": f"No previous analysis found for job_id '{job_id}'. Run /analyze first."},
            status_code=404,
        )

    with open(m2_json_path, "r") as f:
        m2 = json.load(f)

    shot_records      = m2.get("normal_shots", [])
    weak_shot_records = m2.get("weak_shots", [])

    # Rebuild rally-wise sequences and transition matrix from shot records.
    # This prevents the last shot of one rally from connecting to the first shot of the next rally.
    rally_sequences = defaultdict(list)

    for shot in shot_records:
        rally_id = int(shot.get("rally_id", 1))
        shot_type = normalize_shot_name(shot.get("shot_type", "Unknown"))
        if shot_type != "Unknown":
            rally_sequences[rally_id].append(shot_type)

    transition_matrix = defaultdict(lambda: defaultdict(int))

    for rally_id, sequence in rally_sequences.items():
        for i in range(len(sequence) - 1):
            from_shot = sequence[i]
            to_shot = sequence[i + 1]
            transition_matrix[from_shot][to_shot] += 1

    # Load shuttle records if available
    shuttle_json_path = OUTPUTS_DIR / f"{job_id}_shuttle_trajectory_final.json"
    shuttle_records = []
    if shuttle_json_path.exists():
        with open(shuttle_json_path, "r") as f:
            shuttle_records = json.load(f)

    output_paths = {
        "m2_json":        OUTPUTS_DIR / f"{job_id}_m2.json",
        "main_json":      OUTPUTS_DIR / f"{job_id}_main_tactical_output.json",
        "frame_json":     OUTPUTS_DIR / f"{job_id}_frame_level_output.json",
        "shot_json":      OUTPUTS_DIR / f"{job_id}_shot_level_output.json",
        "weak_shot_json": OUTPUTS_DIR / f"{job_id}_weak_shots.json",
        "shuttle_json":   OUTPUTS_DIR / f"{job_id}_shuttle_trajectory_final.json",
        "transition_json":OUTPUTS_DIR / f"{job_id}_tactical_transition_matrix.json",
    }

    tactical_summary = build_tactical_outputs(
        job_id=job_id,
        frame_records=[],           # not needed for output builder
        shot_records=shot_records,
        weak_shot_records=weak_shot_records,
        shuttle_records=shuttle_records,
        transition_matrix=transition_matrix,
        rally_sequences=rally_sequences,
        output_paths=output_paths,
    )

    return JSONResponse({
        "job_id": job_id,
        "message": "Re-analysis completed using saved data.",
        "normal_total_shots":   tactical_summary["normal_total_shots"],
        "weak_total_shots":     tactical_summary["weak_total_shots"],
        "total_including_weak": tactical_summary["total_including_weak"],
        "shot_distribution":    tactical_summary["shot_distribution"],
        "main_json":      f"/outputs/{job_id}_main_tactical_output.json",
        "shot_json":      f"/outputs/{job_id}_shot_level_output.json",
        "weak_shot_json": f"/outputs/{job_id}_weak_shots.json",
        "transition_json":f"/outputs/{job_id}_tactical_transition_matrix.json",
        "rally_patterns": tactical_summary["main_output_preview"].get("rally_patterns", []),
        "pattern_scores": tactical_summary["main_output_preview"].get("pattern_scores", []),
    })

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
