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
import numpy as np
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
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "yolov8n-pose.pt")
TRACKNET_PT = Path(os.getenv("TRACKNET_PT", TRACKNET_DIR / "ckpts" / "TrackNet_best.pt"))
INPAINT_PT = Path(os.getenv("INPAINT_PT", TRACKNET_DIR / "ckpts" / "InpaintNet_best.pt"))

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# CPU only. Do not change this to 0 unless you want GPU execution.
YOLO_DEVICE = "cpu"

# For full video, set MAX_FRAMES = None.
MAX_FRAMES = 1200

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
# Court keypoints that define the net line.
# Point 12 is the left net endpoint and point 11 is the right net endpoint.
NET_KEYPOINT_INDEXES = [11, 12]

# Maximum distance (pixels) shuttle can travel to be considered a weak shot.
WEAK_MAX_TRAVEL_DISTANCE_PX = 240

# Tolerance around the detected net line in pixels.
NET_TOLERANCE = 20

# Optional vertical correction. Keep 0 to use the exact point 11-12 line.
# Negative values move the line upward; positive values move it downward.
NET_Y_OFFSET = 0

# Use several court detections to avoid locking onto one inaccurate frame.
NET_CALIBRATION_SAMPLES = 5

# Match-level fixed net calibration.
_NET_POINT_11_SAMPLES = []
_NET_POINT_12_SAMPLES = []
_FIXED_NET_POINT_11 = None
_FIXED_NET_POINT_12 = None
_NET_LINE_LOCKED = False


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

# Out-of-court confirmation settings
OUT_BOUNDARY_MARGIN_PX = 15
OUT_CONFIRM_FRAMES = 3
OUT_EVENT_COOLDOWN_SEC = 0.8

# Standard doubles-court dimensions used to convert model-detected
# pixel keypoints into real court coordinates for m2.json.
BADMINTON_COURT_WIDTH_M = 6.10
BADMINTON_COURT_LENGTH_M = 13.40

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

def get_left_right_court_status(shuttle_x, shuttle_y, court_points, margin=OUT_BOUNDARY_MARGIN_PX):
    result = {
        "inside_court": None,
        "out_direction": None,
        "left_bound_x": None,
        "right_bound_x": None,
    }
    if shuttle_x is None or shuttle_y is None or court_points is None:
        return result

    left_x, right_x = court_left_right_bounds(shuttle_y, court_points)
    if left_x is None or right_x is None:
        return result
    if left_x > right_x:
        left_x, right_x = right_x, left_x

    result["left_bound_x"] = float(left_x)
    result["right_bound_x"] = float(right_x)

    if shuttle_x < left_x - margin:
        result["inside_court"] = False
        result["out_direction"] = "left_out"
    elif shuttle_x > right_x + margin:
        result["inside_court"] = False
        result["out_direction"] = "right_out"
    else:
        result["inside_court"] = True
    return result


def get_court_homography(court_points):
    """Map court-model pixel keypoints to real court coordinates in metres."""
    if court_points is None:
        return None
    required = [0, 4, 18, 21]
    if any(i >= len(court_points) for i in required):
        return None

    src = []
    for i in required:
        x, y = court_points[i]
        if x <= 0 or y <= 0:
            return None
        src.append([float(x), float(y)])

    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray([
        [0.0, BADMINTON_COURT_LENGTH_M],
        [BADMINTON_COURT_WIDTH_M, BADMINTON_COURT_LENGTH_M],
        [0.0, 0.0],
        [BADMINTON_COURT_WIDTH_M, 0.0],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def pixel_to_court_position(x, y, homography):
    if x is None or y is None or homography is None:
        return {"x": None, "y": None}
    point = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography)[0][0]
    return {"x": round(float(transformed[0]), 4), "y": round(float(transformed[1]), 4)}



# ============================================================
# M2-ONLY COURT COORDINATE CONVERSION
# These helpers are used only while generating m2.json.
# They do not replace or modify the main pipeline court logic.
# ============================================================
M2_INSIDE_TOLERANCE_M = 0.35
M2_HALF_COURT_LENGTH_M = BADMINTON_COURT_LENGTH_M / 2.0


def get_m2_court_homography(court_points):
    """Build a validated pixel-to-court homography for m2.json only.

    Physical coordinate convention before semantic normalization:
      near/bottom baseline -> y = 0.0
      far/top baseline     -> y = 13.4
    """
    if court_points is None:
        return None

    required_indexes = [0, 4, 18, 21]
    if any(index >= len(court_points) for index in required_indexes):
        return None

    candidates = []
    for index in required_indexes:
        x, y = court_points[index]
        if x <= 0 or y <= 0:
            return None
        candidates.append([float(x), float(y)])

    candidates = np.asarray(candidates, dtype=np.float32)
    by_y = candidates[np.argsort(candidates[:, 1])]
    far_pair = by_y[:2]
    near_pair = by_y[2:]

    far_left = far_pair[np.argmin(far_pair[:, 0])]
    far_right = far_pair[np.argmax(far_pair[:, 0])]
    near_left = near_pair[np.argmin(near_pair[:, 0])]
    near_right = near_pair[np.argmax(near_pair[:, 0])]

    src = np.asarray(
        [far_left, far_right, near_left, near_right],
        dtype=np.float32,
    )

    far_width = float(np.linalg.norm(far_right - far_left))
    near_width = float(np.linalg.norm(near_right - near_left))
    left_length = float(np.linalg.norm(near_left - far_left))
    right_length = float(np.linalg.norm(near_right - far_right))

    if min(far_width, near_width, left_length, right_length) < 20.0:
        return None

    width_ratio = max(far_width, near_width) / max(
        1.0, min(far_width, near_width)
    )
    length_ratio = max(left_length, right_length) / max(
        1.0, min(left_length, right_length)
    )

    if width_ratio > 8.0 or length_ratio > 4.0:
        return None

    dst = np.asarray(
        [
            [0.0, BADMINTON_COURT_LENGTH_M],
            [BADMINTON_COURT_WIDTH_M, BADMINTON_COURT_LENGTH_M],
            [0.0, 0.0],
            [BADMINTON_COURT_WIDTH_M, 0.0],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(src, dst)
    if homography is None or not np.all(np.isfinite(homography)):
        return None

    if abs(float(np.linalg.det(homography))) < 1e-9:
        return None

    return homography




def reflect_value_into_range(value, minimum, maximum):
    """Reflect a slightly out-of-range value back inside without pinning it to a boundary."""
    value = float(value)
    minimum = float(minimum)
    maximum = float(maximum)
    span = maximum - minimum
    if span <= 0:
        return minimum

    # Reflection preserves the distance from the nearest boundary:
    # -0.30 -> 0.30, 13.70 -> 13.10 for a 0..13.40 range.
    shifted = (value - minimum) % (2.0 * span)
    if shifted > span:
        shifted = 2.0 * span - shifted
    return minimum + shifted

def get_m2_fallback_position(x, y, court_points):
    """Estimate a physical court position when homography is unavailable."""
    if x is None or y is None or court_points is None:
        return None

    required_indexes = [0, 4, 18, 21]
    if any(index >= len(court_points) for index in required_indexes):
        return None

    corners = []
    for index in required_indexes:
        px, py = court_points[index]
        if px <= 0 or py <= 0:
            return None
        corners.append([float(px), float(py)])

    corners = np.asarray(corners, dtype=np.float32)
    min_x, max_x = float(corners[:, 0].min()), float(corners[:, 0].max())
    min_y, max_y = float(corners[:, 1].min()), float(corners[:, 1].max())

    if max_x - min_x < 1.0 or max_y - min_y < 1.0:
        return None

    normalized_x = (float(x) - min_x) / (max_x - min_x)
    normalized_y = (float(y) - min_y) / (max_y - min_y)

    # Do not silently force a clearly invalid point onto a baseline.
    if not (-0.10 <= normalized_x <= 1.10 and -0.10 <= normalized_y <= 1.10):
        return None

    normalized_x = reflect_value_into_range(normalized_x, 0.0, 1.0)
    normalized_y = reflect_value_into_range(normalized_y, 0.0, 1.0)
    court_x = normalized_x * BADMINTON_COURT_WIDTH_M
    court_y = (1.0 - normalized_y) * BADMINTON_COURT_LENGTH_M
    return {"x": court_x, "y": court_y}


def m2_physical_to_semantic_position(position, selected_player_side="far"):
    """Convert physical near/far coordinates to the M2 player/opponent system.

    M2 semantic convention requested by the user:
      player side   -> y from 0.0 to 6.7
      opponent side -> y from 6.7 to 13.4
    """
    if position is None:
        return None

    court_x = float(position["x"])
    physical_y = float(position["y"])

    if selected_player_side == "far":
        semantic_y = BADMINTON_COURT_LENGTH_M - physical_y
    else:
        semantic_y = physical_y

    return {
        "x": court_x,
        "y": semantic_y,
    }


def m2_pixel_to_court_position(
    x,
    y,
    homography,
    inside_court_value=None,
    court_points=None,
    selected_player_side="far",
):
    """Convert a TrackNet pixel point to M2 court metres.

    This function never inserts a fixed half-court centre. A point is returned
    only from the video's TrackNet coordinates (or from the same pixel point
    projected with the outer-court fallback). Clearly broken transforms are
    rejected so they cannot become repeated 0.0 or 13.4 boundary values.
    """
    physical_position = None

    if x is not None and y is not None and homography is not None:
        point = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
        try:
            transformed = cv2.perspectiveTransform(point, homography)[0][0]
            court_x = float(transformed[0])
            court_y = float(transformed[1])
            if (
                np.isfinite(court_x)
                and np.isfinite(court_y)
                and -M2_INSIDE_TOLERANCE_M <= court_x <= BADMINTON_COURT_WIDTH_M + M2_INSIDE_TOLERANCE_M
                and -M2_INSIDE_TOLERANCE_M <= court_y <= BADMINTON_COURT_LENGTH_M + M2_INSIDE_TOLERANCE_M
            ):
                # Reflect small calibration overshoots back into the court.
                # This preserves boundary distance instead of turning many
                # positions into exactly 0.0 or 13.4.
                physical_position = {
                    "x": reflect_value_into_range(
                        court_x, 0.0, BADMINTON_COURT_WIDTH_M
                    ),
                    "y": reflect_value_into_range(
                        court_y, 0.0, BADMINTON_COURT_LENGTH_M
                    ),
                }
        except cv2.error:
            pass

    if physical_position is None:
        physical_position = get_m2_fallback_position(x, y, court_points)

    semantic = m2_physical_to_semantic_position(
        physical_position,
        selected_player_side=selected_player_side,
    )
    if semantic is None:
        return None

    return {
        "x": round(float(semantic["x"]), 4),
        "y": round(float(semantic["y"]), 4),
    }


def m2_position_matches_landing_side(position, landing_side, tolerance=0.25):
    if position is None:
        return False
    court_y = float(position["y"])
    if landing_side == "player_side":
        return -tolerance <= court_y <= M2_HALF_COURT_LENGTH_M + tolerance
    if landing_side == "opponent_side":
        return M2_HALF_COURT_LENGTH_M - tolerance <= court_y <= BADMINTON_COURT_LENGTH_M + tolerance
    return 0.0 <= court_y <= BADMINTON_COURT_LENGTH_M


def enforce_m2_landing_side(position, landing_side):
    """Keep the converted metre coordinate in the half named by landing_side.

    The position still comes from the same TrackNet pixel. If calibration or
    near/far orientation places it in the opposite semantic half, mirror it
    across the net so the coordinate and landing_side cannot contradict.
    """
    if position is None:
        return None

    court_x = reflect_value_into_range(
        float(position.get("x", BADMINTON_COURT_WIDTH_M / 2.0)),
        0.0,
        BADMINTON_COURT_WIDTH_M,
    )
    court_y = reflect_value_into_range(
        float(position.get("y", M2_HALF_COURT_LENGTH_M)),
        0.0,
        BADMINTON_COURT_LENGTH_M,
    )

    if landing_side == "player_side" and court_y > M2_HALF_COURT_LENGTH_M:
        court_y = BADMINTON_COURT_LENGTH_M - court_y
    elif landing_side == "opponent_side" and court_y < M2_HALF_COURT_LENGTH_M:
        court_y = BADMINTON_COURT_LENGTH_M - court_y

    # Protect against tiny floating-point errors around the net.
    if landing_side == "player_side":
        court_y = min(court_y, M2_HALF_COURT_LENGTH_M)
    elif landing_side == "opponent_side":
        court_y = max(court_y, M2_HALF_COURT_LENGTH_M)

    return {
        "x": round(court_x, 4),
        "y": round(court_y, 4),
    }


def _interpolate_tracknet_window(records, start_frame, end_frame, max_gap_frames):
    """Reconstruct short TrackNet dropouts from surrounding video detections."""
    by_frame = {int(r.get("frame", -1)): r for r in records}
    rows = []
    for frame in range(start_frame, end_frame + 1):
        r = by_frame.get(frame, {})
        rows.append({
            "frame": frame,
            "shuttle_x": r.get("shuttle_x"),
            "shuttle_y": r.get("shuttle_y"),
            "inside_court": r.get("inside_court"),
            "out_direction": r.get("out_direction"),
            "observed": r.get("shuttle_x") is not None and r.get("shuttle_y") is not None,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return []

    for col in ("shuttle_x", "shuttle_y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Interpolate only short internal gaps; do not invent a full trajectory.
        df[col] = df[col].interpolate(
            method="linear",
            limit=max_gap_frames,
            limit_area="inside",
        )

    output = []
    for row in df.to_dict("records"):
        if pd.isna(row["shuttle_x"]) or pd.isna(row["shuttle_y"]):
            continue
        row["shuttle_x"] = float(row["shuttle_x"])
        row["shuttle_y"] = float(row["shuttle_y"])
        row["position_source"] = "tracknet" if row["observed"] else "tracknet_interpolated"
        output.append(row)
    return output


def find_m2_shot_landing(
    shot,
    next_shot,
    shuttle_records,
    homography,
    court_points,
    selected_player_side="far",
    fps=30,
):
    """Estimate landing from the actual TrackNet trajectory for one shot.

    Priority:
      1. Observed TrackNet points in the expected receiving half.
      2. Short-gap interpolation between observed TrackNet points.
      3. Nearest observed TrackNet point around the shot window.

    No fixed centre coordinate is inserted and no invalid/null coordinate is
    written to M2 when at least one shuttle point exists in the video.
    """
    start_frame = int(shot.get("frame", 0))
    hitter = str(shot.get("hitter", "player")).strip().lower()
    landing_side = get_m2_landing_side(hitter)

    if next_shot is not None:
        next_frame = int(next_shot.get("frame", start_frame + int(fps * 2)))
        guard = max(2, int(round(float(fps) * 0.10)))
        end_frame = max(start_frame + 1, next_frame - guard)
    else:
        end_frame = start_frame + int(round(float(fps) * 2.5))

    # Include a few pre-contact frames because TrackNet and shot detection can
    # differ slightly in timing.
    search_start = max(1, start_frame - max(2, int(round(fps * 0.10))))
    max_gap = max(2, int(round(fps * 0.20)))
    trajectory = _interpolate_tracknet_window(
        shuttle_records, search_start, end_frame, max_gap_frames=max_gap
    )

    converted = []
    for record in trajectory:
        position = m2_pixel_to_court_position(
            record.get("shuttle_x"),
            record.get("shuttle_y"),
            homography,
            inside_court_value=record.get("inside_court"),
            court_points=court_points,
            selected_player_side=selected_player_side,
        )
        if position is None:
            continue
        item = dict(record)
        item["position"] = position
        converted.append(item)

    expected = [
        item for item in converted
        if m2_position_matches_landing_side(item["position"], landing_side)
    ]

    if expected:
        # Use the final trajectory point before the next contact. Prefer a real
        # TrackNet observation over an interpolated point at nearly the same time.
        observed_expected = [i for i in expected if i.get("position_source") == "tracknet"]
        pool = observed_expected if observed_expected else expected
        chosen = max(pool, key=lambda i: int(i["frame"]))
        return chosen, chosen["position"], chosen.get("position_source", "tracknet")

    if converted:
        # Timing or calibration may put all points just across the net. Choose
        # the real converted point nearest to the expected half; do not force it
        # to a baseline or substitute a centre coordinate.
        target_y = M2_HALF_COURT_LENGTH_M
        chosen = min(
            converted,
            key=lambda i: abs(float(i["position"]["y"]) - target_y),
        )
        return chosen, chosen["position"], chosen.get("position_source", "tracknet")

    # Last resort: locate the nearest visible TrackNet detection anywhere around
    # this shot. It is still taken from the video, not a fabricated coordinate.
    visible = [
        r for r in shuttle_records
        if r.get("shuttle_x") is not None and r.get("shuttle_y") is not None
    ]
    visible.sort(key=lambda r: abs(int(r.get("frame", 0)) - start_frame))
    for record in visible:
        position = m2_pixel_to_court_position(
            record.get("shuttle_x"), record.get("shuttle_y"), homography,
            court_points=court_points,
            selected_player_side=selected_player_side,
        )
        if position is not None:
            return record, position, "nearest_tracknet"

    # A video with no usable TrackNet coordinate cannot provide an exact landing.
    # Preserve a numeric value using the shot pixel only when it exists.
    position = m2_pixel_to_court_position(
        shot.get("shuttle_x"), shot.get("shuttle_y"), homography,
        court_points=court_points,
        selected_player_side=selected_player_side,
    )
    if position is not None:
        return shot, position, "shot_frame_tracknet"

    # This should occur only when TrackNet produced no coordinates at all.
    # Keep the JSON numeric while explicitly marking the source.
    return shot, {"x": round(BADMINTON_COURT_WIDTH_M / 2, 4), "y": round(M2_HALF_COURT_LENGTH_M, 4)}, "unavailable"


def merge_m2_selected_and_opponent_shots(selected_shots, both_player_events, selected_player_side, fps=30):
    """Keep every selected-player shot_level event and add opponent events.

    selected_shots is authoritative: no selected-player event recognized in
    shot_level_output.json is removed from m2.json.
    """
    merged = []
    selected_frames = []
    selected_side = "near" if selected_player_side == "near" else "far"

    for raw in selected_shots:
        item = dict(raw)
        item["hitter"] = "player"
        item["hitter_physical_side"] = selected_side
        item["m2_event_source"] = "shot_level_output"
        merged.append(item)
        selected_frames.append(int(item.get("frame", 0)))

    duplicate_window = max(3, int(round(float(fps) * 0.18)))
    for raw in both_player_events:
        item = dict(raw)
        hitter = str(item.get("hitter", "")).lower()
        frame = int(item.get("frame", 0))

        if hitter == "player":
            # Already represented by the authoritative shot-level output.
            if any(abs(frame - sf) <= duplicate_window for sf in selected_frames):
                continue
            item["m2_event_source"] = "both_player_detector"
            merged.append(item)
        elif hitter == "opponent":
            item["m2_event_source"] = "both_player_detector"
            merged.append(item)

    return sorted(merged, key=lambda r: int(r.get("frame", 0)))


def get_m2_landing_side(hitter):
    """Return the expected receiving side based on who hit the shot.

    M2 convention:
      - selected player hits -> shuttle goes to opponent_side
      - opponent hits        -> shuttle goes to player_side

    ``landing_position`` is still calculated separately from the TrackNet
    shuttle endpoint and stored unchanged in m2.json.
    """
    normalized_hitter = str(hitter).strip().lower()

    if normalized_hitter == "player":
        return "opponent_side"

    if normalized_hitter == "opponent":
        return "player_side"

    return "unknown"


def _m2_contact_position_for_event(
    shot,
    shuttle_records,
    homography,
    court_points,
    selected_player_side="far",
    fps=30,
):
    """Return the nearest valid TrackNet contact position for one M2 event."""
    frame = int(shot.get("frame", 0))
    radius = max(2, int(round(float(fps) * 0.18)))

    candidates = [
        record
        for record in shuttle_records
        if record.get("shuttle_x") is not None
        and record.get("shuttle_y") is not None
        and abs(int(record.get("frame", 0)) - frame) <= radius
    ]
    candidates.sort(key=lambda record: abs(int(record.get("frame", 0)) - frame))

    for record in candidates:
        position = m2_pixel_to_court_position(
            record.get("shuttle_x"),
            record.get("shuttle_y"),
            homography,
            court_points=court_points,
            selected_player_side=selected_player_side,
        )
        if position is not None:
            return record, position

    # Use the event's own TrackNet point when the surrounding trajectory has a gap.
    if shot.get("shuttle_x") is not None and shot.get("shuttle_y") is not None:
        position = m2_pixel_to_court_position(
            shot.get("shuttle_x"),
            shot.get("shuttle_y"),
            homography,
            court_points=court_points,
            selected_player_side=selected_player_side,
        )
        if position is not None:
            return shot, position

    return None, None


def _m2_hitter_from_contact_position(position, original_hitter, net_tolerance_m=0.45):
    """Infer the hitter from the shuttle's M2 court side at contact.

    M2 semantic convention:
      player side   -> y 0.0 to 6.7
      opponent side -> y 6.7 to 13.4

    Points very close to the net are ambiguous, so retain the detector label.
    """
    original = str(original_hitter).strip().lower()
    if original not in ("player", "opponent"):
        original = "player"

    if position is None:
        return original

    court_y = float(position.get("y", M2_HALF_COURT_LENGTH_M))
    if abs(court_y - M2_HALF_COURT_LENGTH_M) <= net_tolerance_m:
        return original

    return "player" if court_y < M2_HALF_COURT_LENGTH_M else "opponent"


def _m2_shuttle_crossed_between(
    start_frame,
    end_frame,
    shuttle_records,
    homography,
    court_points,
    selected_player_side="far",
):
    """Check whether TrackNet shows a clear net-side crossing between two events."""
    if end_frame <= start_frame:
        return False

    sides = []
    net_band_m = 0.35

    for record in shuttle_records:
        frame = int(record.get("frame", -1))
        if frame < start_frame or frame > end_frame:
            continue
        if record.get("shuttle_x") is None or record.get("shuttle_y") is None:
            continue

        position = m2_pixel_to_court_position(
            record.get("shuttle_x"),
            record.get("shuttle_y"),
            homography,
            court_points=court_points,
            selected_player_side=selected_player_side,
        )
        if position is None:
            continue

        court_y = float(position["y"])
        if court_y < M2_HALF_COURT_LENGTH_M - net_band_m:
            side = "player"
        elif court_y > M2_HALF_COURT_LENGTH_M + net_band_m:
            side = "opponent"
        else:
            continue

        if not sides or sides[-1] != side:
            sides.append(side)

    return len(sides) >= 2


def _m2_event_pixel_distance(first, second):
    x1, y1 = first.get("shuttle_x"), first.get("shuttle_y")
    x2, y2 = second.get("shuttle_x"), second.get("shuttle_y")
    if None in (x1, y1, x2, y2):
        return None
    return float(((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2) ** 0.5)


def clean_m2_video_detected_events(
    ordered_shots,
    shuttle_records,
    court_points,
    selected_player_side="far",
    fps=30,
):
    """Clean and group both-player M2 detections without changing other outputs.

    M2-only rules:
      1. Preserve far/near physical-player labels from the independent
         both-player detector; use TrackNet contact side only as fallback.
      2. Remove near-identical repeated detections.
      3. For consecutive events assigned to the same hitter, use TrackNet net-side
         crossings to distinguish a duplicate from a possibly missed return.
      4. Start a new M2 rally when a valid Service appears after existing shots.
      5. Also start a new rally after a long inactivity gap.
      6. Never blindly alternate hitter labels.
    """
    selected_player_side = "near" if selected_player_side == "near" else "far"
    homography = get_m2_court_homography(court_points)

    prepared = []
    for raw_shot in sorted(ordered_shots, key=lambda item: int(item.get("frame", 0))):
        shot = dict(raw_shot)
        frame = int(shot.get("frame", 0))
        shot_type = normalize_shot_name(shot.get("shot_type", "Unknown"))
        if shot_type not in HIT_SHOT_CLASSES:
            continue

        contact_record, contact_position = _m2_contact_position_for_event(
            shot,
            shuttle_records,
            homography,
            court_points,
            selected_player_side=selected_player_side,
            fps=fps,
        )

        shot["shot_type"] = shot_type

        # Preserve the physical-side label produced by the independent
        # both-player detector. Previously this value was always overwritten
        # from the shuttle contact coordinate, which could map both physical
        # players to "player" when the homography/contact frame was uncertain.
        physical_side = str(
            shot.get("hitter_physical_side", "")
        ).strip().lower()
        selected_side = (
            "near" if selected_player_side == "near" else "far"
        )

        if physical_side in ("far", "near"):
            shot["hitter"] = (
                "player"
                if physical_side == selected_side
                else "opponent"
            )
        else:
            # Use TrackNet contact-side inference only as a fallback when the
            # both-player detector did not provide a physical court side.
            shot["hitter"] = _m2_hitter_from_contact_position(
                contact_position,
                shot.get("hitter", "player"),
            )

        shot["_m2_contact_position"] = contact_position
        if contact_record is not None:
            shot["_m2_contact_frame"] = int(contact_record.get("frame", frame))
        else:
            shot["_m2_contact_frame"] = frame
        prepared.append(shot)

    cleaned = []
    duplicate_window_frames = max(4, int(round(float(fps) * 0.35)))
    same_hitter_review_frames = max(
        duplicate_window_frames,
        int(round(float(fps) * 0.85)),
    )

    for shot in prepared:
        if not cleaned:
            cleaned.append(shot)
            continue

        previous = cleaned[-1]
        frame = int(shot.get("frame", 0))
        previous_frame = int(previous.get("frame", 0))
        gap = max(0, frame - previous_frame)

        same_hitter = shot.get("hitter") == previous.get("hitter")
        same_type = shot.get("shot_type") == previous.get("shot_type")
        pixel_distance = _m2_event_pixel_distance(previous, shot)
        shuttle_nearly_same = pixel_distance is not None and pixel_distance <= 90.0

        crossed = _m2_shuttle_crossed_between(
            previous_frame,
            frame,
            shuttle_records,
            homography,
            court_points,
            selected_player_side=selected_player_side,
        )

        near_identical = (
            same_hitter
            and gap <= duplicate_window_frames
            and not crossed
            and (same_type or shuttle_nearly_same)
        )

        same_hitter_without_return = (
            same_hitter
            and gap <= same_hitter_review_frames
            and not crossed
        )

        if near_identical or same_hitter_without_return:
            current_conf = float(shot.get("confidence", 0.0) or 0.0)
            previous_conf = float(previous.get("confidence", 0.0) or 0.0)

            # Keep the stronger event as the most likely real contact.
            if current_conf > previous_conf:
                cleaned[-1] = shot
            continue

        # If the same hitter is seen after TrackNet crossed the net, keep both:
        # the opposite player's return may have been missed by the shot model.
        cleaned.append(shot)

    # Assign M2 rally IDs independently from the main pipeline.
    rally_id = 1
    previous_frame = None
    shots_in_current_rally = 0
    long_gap_frames = max(1, int(round(float(fps) * 4.0)))

    for shot in cleaned:
        frame = int(shot.get("frame", 0))
        shot_type = normalize_shot_name(shot.get("shot_type", "Unknown"))

        starts_with_service = shot_type == "Service" and shots_in_current_rally > 0
        starts_after_long_gap = (
            previous_frame is not None
            and frame - previous_frame > long_gap_frames
            and shots_in_current_rally > 0
        )

        if starts_with_service or starts_after_long_gap:
            rally_id += 1
            shots_in_current_rally = 0

        shot["rally_id"] = rally_id
        shots_in_current_rally += 1
        previous_frame = frame

        # Remove internal helper values before writing m2.json.
        shot.pop("_m2_contact_position", None)
        shot.pop("_m2_contact_frame", None)

    return cleaned


def build_separate_m2_output(
    match_id,
    player_id,
    shot_records,
    shuttle_records,
    court_points,
    output_path,
    fps=30,
    selected_player_side="far",
):
    """Build m2.json with shots from both the selected player and opponent."""
    homography = get_m2_court_homography(court_points)
    ordered = sorted(
        shot_records,
        key=lambda r: (
            int(r.get("rally_id", 1)),
            int(r.get("frame", 0)),
        ),
    )
    ordered = clean_m2_video_detected_events(
        ordered,
        shuttle_records=shuttle_records,
        court_points=court_points,
        selected_player_side=selected_player_side,
        fps=fps,
    )
    shots = []
    rally_ids = defaultdict(list)
    rally_times = defaultdict(list)

    selected_player_side = "near" if selected_player_side == "near" else "far"

    for i, shot in enumerate(ordered):
        next_shot = ordered[i + 1] if i + 1 < len(ordered) else None
        landing, position, position_source = find_m2_shot_landing(
            shot=shot,
            next_shot=next_shot,
            shuttle_records=shuttle_records,
            homography=homography,
            court_points=court_points,
            selected_player_side=selected_player_side,
            fps=fps,
        )
        shot_id = i + 1
        rally_id = int(shot.get("rally_id", 1))
        timestamp = round(float(shot.get("time_sec", 0)), 4)
        hitter = shot.get("hitter", "player")
        landing_side = get_m2_landing_side(hitter)
        position = enforce_m2_landing_side(position, landing_side)

        # Keep the original TrackNetV3 landing coordinate in video pixels.
        # The converted badminton-court coordinate is stored separately in metres.
        landing_pixel_position = {
            "x": round(float(landing.get("shuttle_x")), 4)
            if landing.get("shuttle_x") is not None else None,
            "y": round(float(landing.get("shuttle_y")), 4)
            if landing.get("shuttle_y") is not None else None,
        }

        shots.append({
            "shot_id": shot_id,
            "timestamp": timestamp,
            "hitter": hitter,
            "shot_type": normalize_shot_name(
                shot.get("shot_type", "Unknown")
            ),
            "landing_side": landing_side,
            "landing_position": landing_pixel_position,
            "landing_position_meters": position,
            "landing_frame": int(landing.get("frame", shot.get("frame", 0))),
            "landing_position_source": position_source,
        })
        rally_ids[rally_id].append(shot_id)
        rally_times[rally_id].append(timestamp)

    rallies = []
    for rally_id in sorted(rally_ids):
        times = rally_times[rally_id]
        duration = max(times) - min(times) if len(times) > 1 else 0.0
        rallies.append({
            "rally_id": rally_id,
            "shots": rally_ids[rally_id],
            "duration": round(duration, 2),
        })

    counts = Counter(s["shot_type"] for s in shots)
    total = len(shots)
    output = {
        "match_id": match_id,
        "player_id": player_id,
        "selected_player_side": selected_player_side,
        "total_shots": total,
        "total_rallies": len(rallies),
        "shots": shots,
        "rallies": rallies,
        "shot_distribution": {
            k: round(v / total * 100, 1) for k, v in counts.items()
        } if total else {},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output


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
def _valid_net_point(court_points, index):
    if court_points is None or index >= len(court_points):
        return None

    x, y = court_points[index]

    if x is None or y is None:
        return None

    x = float(x)
    y = float(y)

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    if x <= 0 or y <= 0:
        return None

    return x, y


def update_fixed_match_net(court_points):
    """Calibrate the match net from court points 11 and 12, then lock it.

    Point 12 is treated as the left endpoint and point 11 as the right endpoint.
    Several valid court detections are combined with a median so one inaccurate
    detection cannot place the net line incorrectly.
    """
    global _FIXED_NET_POINT_11
    global _FIXED_NET_POINT_12
    global _NET_LINE_LOCKED

    if _NET_LINE_LOCKED:
        return True

    point_11 = _valid_net_point(court_points, 11)
    point_12 = _valid_net_point(court_points, 12)

    if point_11 is None or point_12 is None:
        return False

    # Reject impossible endpoint ordering or a line that is too short.
    left_candidate = point_12
    right_candidate = point_11

    if left_candidate[0] > right_candidate[0]:
        left_candidate, right_candidate = right_candidate, left_candidate

    line_width = right_candidate[0] - left_candidate[0]

    if line_width < 80:
        return False

    _NET_POINT_11_SAMPLES.append(point_11)
    _NET_POINT_12_SAMPLES.append(point_12)

    sample_count = min(
        len(_NET_POINT_11_SAMPLES),
        len(_NET_POINT_12_SAMPLES),
    )

    # Use the running median during calibration.
    p11_array = np.asarray(_NET_POINT_11_SAMPLES, dtype=np.float32)
    p12_array = np.asarray(_NET_POINT_12_SAMPLES, dtype=np.float32)

    median_11 = np.median(p11_array, axis=0)
    median_12 = np.median(p12_array, axis=0)

    _FIXED_NET_POINT_11 = (
        float(median_11[0]),
        float(median_11[1]) + float(NET_Y_OFFSET),
    )
    _FIXED_NET_POINT_12 = (
        float(median_12[0]),
        float(median_12[1]) + float(NET_Y_OFFSET),
    )

    if sample_count >= NET_CALIBRATION_SAMPLES:
        _NET_LINE_LOCKED = True

        print(
            "✅ Fixed match net calibrated from court points 11 and 12:",
            {
                "point_12_left": [
                    round(_FIXED_NET_POINT_12[0], 2),
                    round(_FIXED_NET_POINT_12[1], 2),
                ],
                "point_11_right": [
                    round(_FIXED_NET_POINT_11[0], 2),
                    round(_FIXED_NET_POINT_11[1], 2),
                ],
                "samples": sample_count,
            },
        )

    return True


def get_fixed_net_points(court_points=None):
    """Return the fixed left/right net endpoints for the current match."""
    update_fixed_match_net(court_points)

    if _FIXED_NET_POINT_11 is None or _FIXED_NET_POINT_12 is None:
        return None, None

    point_a = _FIXED_NET_POINT_12
    point_b = _FIXED_NET_POINT_11

    if point_a[0] <= point_b[0]:
        return point_a, point_b

    return point_b, point_a


def get_net_y_at_x(x, court_points=None):
    """Return the Y coordinate of the sloped point-12-to-point-11 net at X."""
    left_point, right_point = get_fixed_net_points(court_points)

    if left_point is None or right_point is None:
        return None

    left_x, left_y = left_point
    right_x, right_y = right_point

    if abs(right_x - left_x) < 1e-6:
        return (left_y + right_y) / 2.0

    x = float(x)
    ratio = (x - left_x) / (right_x - left_x)

    # Extrapolation is allowed near the court edges but limited so a shuttle
    # far outside the image cannot produce an extreme net Y value.
    ratio = max(-0.25, min(1.25, ratio))

    return left_y + ratio * (right_y - left_y)


def get_net_y(court_points):
    """Backward-compatible centre Y value of the fixed 11-12 net line."""
    left_point, right_point = get_fixed_net_points(court_points)

    if left_point is None or right_point is None:
        return None

    center_x = (left_point[0] + right_point[0]) / 2.0
    return get_net_y_at_x(center_x, court_points)


# Determine which side of the court the shuttle is on (top/bottom/net area)
def get_shuttle_side(sx, sy, court_points):
    """Classify shuttle side using the sloped fixed line through points 12 and 11."""
    if sx is None or sy is None:
        return "unknown"

    net_y = get_net_y_at_x(sx, court_points)

    if net_y is None:
        return "unknown"

    if sy < net_y - NET_TOLERANCE:
        return "top"

    if sy > net_y + NET_TOLERANCE:
        return "bottom"

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
    """Draw the fixed sloped net line directly between court points 12 and 11."""
    left_point, right_point = get_fixed_net_points(court_points)

    if left_point is None or right_point is None:
        return frame

    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, min(frame_w - 1, int(round(left_point[0]))))
    y1 = max(0, min(frame_h - 1, int(round(left_point[1]))))
    x2 = max(0, min(frame_w - 1, int(round(right_point[0]))))
    y2 = max(0, min(frame_h - 1, int(round(right_point[1]))))

    cv2.line(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        3,
    )

    cv2.circle(frame, (x1, y1), 7, (0, 255, 255), -1)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 255), -1)

    label_y = max(25, min(y1, y2) - 10)

    cv2.putText(
        frame,
        "FIXED NET: POINT 12 TO POINT 11",
        (max(10, x1), label_y),
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


def valid_shot_event(
    current_shot,
    sx,
    sy,
    player_box,
    shuttle_history,
    shot_conf,
    shuttle_side,
    focused_player_side,
    court_points
):
    current_shot = normalize_shot_name(current_shot)

    if current_shot not in HIT_SHOT_CLASSES:
        return False

    if shot_conf < SHOT_CONF_THRES:
        return False

    # Shuttle must be on the selected player's side
    # far  player = top side
    # near player = bottom side
    same_side = shuttle_side == focused_player_side

    # Allow net-area shots only if shuttle is very close to net
    near_net = (
        shuttle_side == "net_area"
        and shuttle_near_net(sy, court_points, margin=70)
    )

    if not (same_side or near_net):
        return False


   # Service can be accepted when shuttle is on selected player's side
    # and close to selected player, even if trajectory change is not clear yet
    if current_shot == "Service" and shot_conf >= 0.50 :
        return True
    if current_shot == "NetShot" and shot_conf >= 0.50:
        return True
    if current_shot == "BackHand" and shot_conf >= 0.50:
        return True
    # Other shots need shuttle direction/speed change
    trajectory_changed = shuttle_direction_or_speed_changed(shuttle_history)

    if not trajectory_changed:
        return False

    return True

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
    confirmed_out_events=None,
):
    if confirmed_out_events is None:
        confirmed_out_events = []

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
    shots_out = sum(1 for shot in shot_records if shot.get("inside_court") is False)
    shots_in_court = sum(1 for shot in shot_records if shot.get("inside_court") is True)
    left_out_count = sum(1 for shot in shot_records if shot.get("out_direction") == "left_out")
    right_out_count = sum(1 for shot in shot_records if shot.get("out_direction") == "right_out")

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
        "shots_in_court": shots_in_court,
        "shots_out": shots_out,
        "left_out_count": left_out_count,
        "right_out_count": right_out_count,
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

    # Use confirmed, deduplicated left/right out events.
    out_of_court_events = confirmed_out_events

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
        "out_of_court_events": out_of_court_events,
    }

    with open(output_paths["main_json"], "w", encoding="utf-8") as f:
        json.dump(main_output, f, indent=2)

    # ============================================================
    # TACTICAL METRICS FOR combination/combine.json
    # ============================================================
    # Map internal tactical shot names to the requested output names.
    tactical_metric_name_map = {
        "Smash": "smash",
        "DropShot": "drop",
        "Clear": "clear",
        "NetShot": "net",
        "Service": "serve",
        "Lift": "lift",
        "ForeHand": "forehand",
        "BackHand": "backhand",
    }

    tactical_metric_distribution = {}
    for shot_name, percentage in percentage_distribution.items():
        output_name = tactical_metric_name_map.get(
            normalize_shot_name(shot_name),
            str(shot_name).strip().lower(),
        )
        tactical_metric_distribution[output_name] = round(float(percentage), 1)

    # Weak shots are treated as the current proxy for unforced errors,
    # according to the requested output definition.
    weak_shot_ratio = (
        weak_total_shots / normal_total_shots
        if normal_total_shots > 0
        else 0.0
    )

    # A true smash-success metric requires rally-winner information.
    # With the currently available fields, an in-court smash is used as
    # the measurable success proxy.
    smash_records = [
        shot for shot in shot_records
        if normalize_shot_name(shot.get("shot_type", "Unknown")) == "Smash"
    ]
    successful_smashes = sum(
        1 for shot in smash_records
        if shot.get("inside_court") is True
    )
    smash_success_rate = (
        successful_smashes / len(smash_records)
        if smash_records
        else 0.0
    )

    tactical_metrics = {
        "shot_distribution": tactical_metric_distribution,
        "smash_success_rate": round(float(smash_success_rate), 4),
        "unforced_error_rate": round(float(weak_shot_ratio), 4),
        "average_rally_length": round(float(average_rally_length), 2),
        "weak_shot_ratio": round(float(weak_shot_ratio), 4),
        "total_shots": normal_total_shots,
    }

    combined_output = {
        "match_id": match_id,
        "player_id": player_id,
        "job_id": job_id,
        "tactical_metrics": tactical_metrics,
        "summary": {
            "processed_frames": len(frame_records),
            "max_frames_used": MAX_FRAMES,
            "normal_total_shots": normal_total_shots,
            "weak_total_shots": weak_total_shots,
            "total_including_weak": normal_total_shots + weak_total_shots,
            "shots_in_court": shots_in_court,
            "shots_out": shots_out,
            "left_out_count": left_out_count,
            "right_out_count": right_out_count,
            "normal_shot_distribution": percentage_distribution,
            "total_rallies": len(rallies),
        },
        "main_tactical_output": main_output,
        "frame_level_output": frame_records,
        "shot_level_output": shot_records,
        "weak_shots": weak_shot_records,
        "shuttle_trajectory": shuttle_records,
        "out_of_court_events": out_of_court_events,
        "tactical_transition_matrix": transition_df.to_dict(orient="records"),
    }

    with open(output_paths["combine_json"], "w", encoding="utf-8") as f:
        json.dump(combined_output, f, indent=2, ensure_ascii=False)

    return {
        "normal_total_shots": normal_total_shots,
        "weak_total_shots": weak_total_shots,
        "total_including_weak": normal_total_shots + weak_total_shots,
        "shots_in_court": shots_in_court,
        "shots_out": shots_out,
        "left_out_count": left_out_count,
        "right_out_count": right_out_count,
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
    player_side: str = Form("far"),  # far, near, all
):
    # Generate unique ID for this analysis job
    global _FIXED_NET_POINT_11
    global _FIXED_NET_POINT_12
    global _NET_LINE_LOCKED
    _NET_POINT_11_SAMPLES.clear()
    _NET_POINT_12_SAMPLES.clear()
    _FIXED_NET_POINT_11 = None
    _FIXED_NET_POINT_12 = None
    _NET_LINE_LOCKED = False
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
    pose_model = YOLO(POSE_MODEL_PATH)           # Detects player body keypoints

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
    shot_records = []  # Selected-player normal shots (existing outputs)
    m2_shot_records = []  # Both players' shots for m2.json only
    weak_shot_records = []  # Shots classified as weak
    shuttle_records = []  # Shuttle position tracking
    shot_sequence = []  # Sequence of shots inside the current rally only
    rally_sequences = defaultdict(list)  # Rally-wise shot patterns: {rally_id: [shot1, shot2, ...]}
    transition_matrix = defaultdict(lambda: defaultdict(int))  # Shot transition counts inside rallies only

    # Shot detection variables
    last_shot = None
    last_shot_frame = -999
    shot_cooldown = int(fps * 0.2)  
    target_track_id = None  # ID of the player being tracked
    shuttle_history = deque(maxlen=15)  # Keep last 15 shuttle positions
    shot_class_history = deque(maxlen=7)  # Stabilize shot prediction with history
    last_court_points = None  # Cache court keypoints for efficiency
    frame_no = 0

    # ====================================================
    # USER-SELECTED PLAYER SIDE + HIT-MOMENT SHOT LOGIC
    # This uses the working Colab logic: shot model + pose wrist movement.
    # It does not change weak-shot/rally/output logic below.
    # ====================================================
    PLAYER_SIDE = str(player_side).lower().strip()
    if PLAYER_SIDE not in ["far", "near", "all"]:
        PLAYER_SIDE = "far"

    Y_SPLIT = 0.55
    SHOT_CONF = 0.12
    POSE_CONF = 0.25
    IMGSZ = 960

    CLASS_MOVEMENT_THRESHOLD = {
        "Service": 6,
        "NetShot": 7,
        "Lift": 10,
        "BackHand": 14,
        "ForeHand": 14,
        "Smash": 18,
        "ReadyPosition": 999,
    }

    STABLE_FRAMES = 2
    COOLDOWN_FRAMES = 8
    prev_wrist_center = None
    shot_buffer = deque(maxlen=STABLE_FRAMES)
    shot_logic_cooldown = 0

    # Independent recognition state for both court sides. This is used only
    # to build m2.json; the existing selected-player output remains unchanged.
    m2_side_state = {
        "far": {
            "prev_wrist": None,
            "buffer": deque(maxlen=STABLE_FRAMES),
            "cooldown": 0,
            "last_frame": -999,
            "last_shot": None,
        },
        "near": {
            "prev_wrist": None,
            "buffer": deque(maxlen=STABLE_FRAMES),
            "cooldown": 0,
            "last_frame": -999,
            "last_shot": None,
        },
    }

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

    outside_frame_count = 0
    outside_direction_candidate = None
    last_out_event_frame = -999
    out_event_cooldown_frames = max(1, int(fps * OUT_EVENT_COOLDOWN_SEC))
    confirmed_out_events = []

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


        player_result = player_model.predict(frame, conf=SHOT_CONF, iou=0.45, imgsz=IMGSZ, device=YOLO_DEVICE, verbose=False)[0]

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

        split_line = height * Y_SPLIT

        # ====================================================
        # WORKING SHOT RECOGNITION LOGIC
        # Keeps only far/near/all player, checks wrist movement,
        # then accepts the best shot from your trained model.
        # ====================================================
        current_wrist_center = None
        movement = 0
        wrist_centers_by_side = {"far": None, "near": None}
        movements_by_side = {"far": 0.0, "near": 0.0}

        pose_results = pose_model.predict(
            frame,
            conf=POSE_CONF,
            imgsz=IMGSZ,
            device=YOLO_DEVICE,
            verbose=False,
        )[0]

        if pose_results.keypoints is not None:
            keypoints = pose_results.keypoints.xy.cpu().numpy()

            # Keep the best/most central pose detected on each court half.
            pose_candidates = {"far": [], "near": []}
            for person_kpts in keypoints:
                valid_points = person_kpts[person_kpts[:, 1] > 0]
                if len(valid_points) == 0:
                    continue

                person_center_y = float(valid_points[:, 1].mean())
                side = "far" if person_center_y < split_line else "near"
                pose_candidates[side].append((person_center_y, person_kpts))

            for side in ("far", "near"):
                if not pose_candidates[side]:
                    continue

                # Far player: choose candidate nearest the far court center.
                # Near player: choose candidate nearest the near court center.
                expected_y = split_line * 0.65 if side == "far" else split_line + (height - split_line) * 0.55
                _, person_kpts = min(
                    pose_candidates[side],
                    key=lambda item: abs(item[0] - expected_y),
                )

                wrists = []
                for wrist_index in (9, 10):
                    wrist = person_kpts[wrist_index]
                    if wrist[0] > 0 and wrist[1] > 0:
                        wrists.append(wrist)

                if wrists:
                    wrist_centers_by_side[side] = sum(wrists) / len(wrists)

                selected_side_for_drawing = (
                    PLAYER_SIDE == side or PLAYER_SIDE == "all"
                )
                if selected_side_for_drawing:
                    for x, y in person_kpts:
                        if x > 0 and y > 0:
                            cv2.circle(
                                frame,
                                (int(x), int(y)),
                                4,
                                (0, 0, 255),
                                -1,
                            )

        for side in ("far", "near"):
            wrist = wrist_centers_by_side[side]
            previous = m2_side_state[side]["prev_wrist"]
            if wrist is not None and previous is not None:
                movements_by_side[side] = float(
                    ((wrist[0] - previous[0]) ** 2 +
                     (wrist[1] - previous[1]) ** 2) ** 0.5
                )
            m2_side_state[side]["prev_wrist"] = wrist

        selected_detection_side = "near" if PLAYER_SIDE == "near" else "far"
        current_wrist_center = wrist_centers_by_side[selected_detection_side]
        movement = movements_by_side[selected_detection_side]
        prev_wrist_center = current_wrist_center

        current_shot = "Unknown"
        current_conf = 0
        player_box = None

        best_shot = None
        best_box = None
        best_conf = 0

        for det in all_detections:
            shot_name = normalize_shot_name(det["class_name"])
            conf = det["conf"]

            if shot_name == "ReadyPosition":
                continue

            if shot_name not in HIT_SHOT_CLASSES:
                continue

            x1, y1, x2, y2 = det["box"]
            box_center_y = (y1 + y2) / 2

            if PLAYER_SIDE == "far" and box_center_y > split_line:
                continue
            elif PLAYER_SIDE == "near" and box_center_y < split_line:
                continue

            if conf > best_conf:
                best_conf = conf
                best_shot = shot_name
                best_box = det["box"]

        # Best shot-class detection on each court half for m2.json.
        best_by_side = {
            "far": {"shot": None, "conf": 0.0, "box": None},
            "near": {"shot": None, "conf": 0.0, "box": None},
        }
        for det in all_detections:
            shot_name = normalize_shot_name(det["class_name"])
            if shot_name not in HIT_SHOT_CLASSES:
                continue
            x1, y1, x2, y2 = det["box"]
            side = "far" if ((y1 + y2) / 2) < split_line else "near"
            if det["conf"] > best_by_side[side]["conf"]:
                best_by_side[side] = {
                    "shot": shot_name,
                    "conf": float(det["conf"]),
                    "box": det["box"],
                }

        is_hitting_motion = False

        if best_shot is not None:
            required_movement = CLASS_MOVEMENT_THRESHOLD.get(best_shot, 14)

            if movement >= required_movement:
                is_hitting_motion = True

        if shot_logic_cooldown == 0 and best_shot is not None:
            shot_buffer.append(best_shot)
        else:
            shot_buffer.append(None)

        valid_shots = [s for s in shot_buffer if s is not None]

        if len(valid_shots) >= 1 and best_box is not None:
            final_shot = Counter(valid_shots).most_common(1)[0][0]
            current_shot = final_shot
            current_conf = best_conf
            player_box = best_box
            shot_logic_cooldown = COOLDOWN_FRAMES

        if shot_logic_cooldown > 0:
            shot_logic_cooldown -= 1

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

        # Confirm left/right out events over multiple consecutive frames.
        court_status = get_left_right_court_status(sx, sy, court_points)
        is_inside = court_status["inside_court"]
        out_direction = court_status["out_direction"]
        left_bound_x = court_status["left_bound_x"]
        right_bound_x = court_status["right_bound_x"]
        confirmed_out_this_frame = False

        if is_inside is False and out_direction in ("left_out", "right_out"):
            if outside_direction_candidate == out_direction:
                outside_frame_count += 1
            else:
                outside_direction_candidate = out_direction
                outside_frame_count = 1

            if outside_frame_count >= OUT_CONFIRM_FRAMES and frame_no - last_out_event_frame > out_event_cooldown_frames:
                confirmed_out_this_frame = True
                last_out_event_frame = frame_no
                event = {
                    "event_id": len(confirmed_out_events) + 1,
                    "frame": frame_no,
                    "time_sec": round(timestamp, 3),
                    "shuttle_x": sx,
                    "shuttle_y": sy,
                    "out_direction": out_direction,
                    "left_bound_x": round(left_bound_x, 2) if left_bound_x is not None else None,
                    "right_bound_x": round(right_bound_x, 2) if right_bound_x is not None else None,
                    "rally_id": None,
                    "shot_number": None,
                    "shot_type": None,
                }
                if shot_records:
                    latest_shot = shot_records[-1]
                    latest_shot["inside_court"] = False
                    latest_shot["out_direction"] = out_direction
                    latest_shot["out_frame"] = frame_no
                    latest_shot["out_time_sec"] = round(timestamp, 3)
                    event["rally_id"] = latest_shot.get("rally_id")
                    event["shot_number"] = latest_shot.get("shot_number")
                    event["shot_type"] = latest_shot.get("shot_type")
                confirmed_out_events.append(event)
        else:
            outside_frame_count = 0
            outside_direction_candidate = None
        # Track shuttle position history for analyzing motion
        shuttle_history.append({"frame": frame_no, "x": sx, "y": sy, "visibility": sv})

        # Determine which side of the net the shuttle is on
        shuttle_side = get_shuttle_side(sx, sy, court_points)
        # Get Y position of the net line
        net_y = get_net_y(court_points)
        # Detect if shuttle is falling/dropping
        drop_event = is_shuttle_drop_event(shuttle_history)

        # ====================================================
        # M2 BOTH-PLAYER SHOT EVENTS
        # User-selected side is labelled "player"; the opposite side is
        # labelled "opponent". These events do not change shot_records.
        # ====================================================
        selected_m2_side = "near" if PLAYER_SIDE == "near" else "far"
        side_to_shuttle_side = {"far": "top", "near": "bottom"}

        for side in ("far", "near"):
            state = m2_side_state[side]
            candidate = best_by_side[side]
            candidate_shot = candidate["shot"]
            candidate_conf = candidate["conf"]
            candidate_box = candidate["box"]

            if state["cooldown"] > 0:
                state["cooldown"] -= 1

            movement_threshold = CLASS_MOVEMENT_THRESHOLD.get(
                candidate_shot, 14
            ) if candidate_shot else 999
            has_hitting_motion = (
                candidate_shot is not None
                and movements_by_side[side] >= movement_threshold
            )

            if state["cooldown"] == 0 and has_hitting_motion:
                state["buffer"].append(candidate_shot)
            else:
                state["buffer"].append(None)

            stable_candidates = [
                s for s in state["buffer"] if s is not None
            ]
            if not stable_candidates:
                continue

            side_shot = Counter(stable_candidates).most_common(1)[0][0]
            expected_shuttle_side = side_to_shuttle_side[side]
            shuttle_is_at_hitter = (
                shuttle_side == expected_shuttle_side
                or (
                    shuttle_side == "net_area"
                    and shuttle_near_net(sy, court_points, margin=70)
                )
            )

            # Follow the same practical acceptance rules as the selected-player
            # detector, but maintain an independent cooldown for each player.
            trajectory_ok = shuttle_direction_or_speed_changed(
                shuttle_history
            )
            early_contact_class = (
                side_shot in ("Service", "NetShot", "BackHand")
                and candidate_conf >= 0.50
            )
            valid_m2_event = (
                candidate_conf >= SHOT_CONF
                and shuttle_is_at_hitter
                and (trajectory_ok or early_contact_class)
            )

            enough_gap = (
                frame_no - state["last_frame"] >= COOLDOWN_FRAMES
            )
            duplicate_too_soon = (
                side_shot == state["last_shot"]
                and frame_no - state["last_frame"] < COOLDOWN_FRAMES * 2
            )

            if valid_m2_event and enough_gap and not duplicate_too_soon:
                hitter = "player" if side == selected_m2_side else "opponent"
                m2_shot_records.append({
                    "match_id": match_id,
                    "rally_id": current_rally_id,
                    "frame": frame_no,
                    "time_sec": round(timestamp, 4),
                    "hitter": hitter,
                    "hitter_physical_side": side,
                    "shot_type": side_shot,
                    "confidence": round(candidate_conf, 3),
                    "player_box": candidate_box,
                    "shuttle_x": sx,
                    "shuttle_y": sy,
                    "inside_court": is_inside,
                    "out_direction": out_direction,
                })
                state["last_frame"] = frame_no
                state["last_shot"] = side_shot
                state["cooldown"] = COOLDOWN_FRAMES
                state["buffer"].clear()

        # Focused player side based on user selection. Far player is top side; near player is bottom side.
        focused_player_side = "bottom" if PLAYER_SIDE == "near" else "top"
      
        RALLY_GAP_SECONDS = 4.0
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
                is_inside is False
                and out_direction in ("left_out", "right_out")
            )

            # Change color if shuttle passed net to opponent's side
            if shuttle_side == opposite_court_side(focused_player_side):
                shuttle_color = BLUE
                shuttle_status = "PASSED NET"

            # Out-of-bounds has highest priority in coloring
            if outside_left_right:
                shuttle_color = RED
                shuttle_status = "OUT - LEFT" if out_direction == "left_out" else "OUT - RIGHT"

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
        cv2.putText(frame, f"Tracked Player: {PLAYER_SIDE.upper()} PLAYER | Shot: {current_shot} | Move: {movement:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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
            shuttle_side=shuttle_side,
            focused_player_side=focused_player_side,
            court_points=court_points,
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
                
                # Service means a new rally starts
                if shot_records:
                    current_rally_id += 1
                    shot_sequence = []
                    shot_class_history.clear()
                    last_shot = None
                    last_shot_frame = -999
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
                "out_direction": out_direction,
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
                "out_direction": out_direction,
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
                    "out_direction": out_direction,
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
                    "out_direction": out_direction,
                    "net_y": round(net_y, 2) if net_y is not None else None,
                    "shuttle_side": shuttle_side,
                    "visibility": sv,
                    "left_bound_x": round(left_bound_x, 2) if left_bound_x is not None else None,
                    "right_bound_x": round(right_bound_x, 2) if right_bound_x is not None else None,
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
        "combine_json": OUTPUTS_DIR / f"{job_id}_combine.json",
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
        confirmed_out_events=confirmed_out_events,
    )

    # M2 uses its original both-player video detector only.
    # Do not force every selected-player event from shot_level_output.json
    # into M2; keep the earlier M2 shot filtering while retaining both players.

    # M2 rally IDs are assigned inside the M2-only cleanup function.
    # The main pipeline rally logic and all other outputs remain unchanged.

    m2_output = build_separate_m2_output(
        match_id=match_id,
        player_id=player_name,
        shot_records=m2_shot_records,
        shuttle_records=shuttle_records,
        court_points=last_court_points,
        output_path=output_paths["m2_json"],
        fps=fps,
        selected_player_side=selected_m2_side,
    )

    return JSONResponse(
        {
            "job_id": job_id,
            "message": "Analysis completed.",
            "processed_frames": frame_no,
            "player_side": PLAYER_SIDE,
            "video": f"/outputs/{browser_video.name}",

            # JSON outputs 
            "m2_json": f"/outputs/{job_id}_m2.json",
            "combine_json": f"/outputs/{job_id}_combine.json",
            "main_json": f"/outputs/{job_id}_main_tactical_output.json",
            "frame_json": f"/outputs/{job_id}_frame_level_output.json",
            "shot_json": f"/outputs/{job_id}_shot_level_output.json",
            "weak_shot_json": f"/outputs/{job_id}_weak_shots.json",
            "shuttle_json": f"/outputs/{job_id}_shuttle_trajectory_final.json",
            "transition_json": f"/outputs/{job_id}_tactical_transition_matrix.json",

            "transition_matrix": {k: dict(v) for k, v in transition_matrix.items()},
            "rally_patterns": tactical_summary["main_output_preview"].get("rally_patterns", []),
            "pattern_scores": tactical_summary["main_output_preview"].get("pattern_scores", []),
            "out_of_court_events": tactical_summary["main_output_preview"].get("out_of_court_events", []),
            "normal_total_shots": tactical_summary["normal_total_shots"],
            "weak_total_shots": tactical_summary["weak_total_shots"],
            "total_including_weak": tactical_summary["total_including_weak"],
            "shot_distribution": tactical_summary["shot_distribution"],
            "shots_in_court": tactical_summary["shots_in_court"],
            "shots_out": tactical_summary["shots_out"],
            "left_out_count": tactical_summary["left_out_count"],
            "right_out_count": tactical_summary["right_out_count"],
            "total_rallies": m2_output["total_rallies"],
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
    combined_json_path = OUTPUTS_DIR / f"{job_id}_combine.json"

    if not combined_json_path.exists():
        return JSONResponse(
            {"error": f"No previous analysis found for job_id '{job_id}'. Run /analyze first."},
            status_code=404,
        )

    with open(combined_json_path, "r", encoding="utf-8") as f:
        combined = json.load(f)

    shot_records = combined.get("shot_level_output", [])
    weak_shot_records = combined.get("weak_shots", [])
    frame_records = combined.get("frame_level_output", [])

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
        "combine_json":   OUTPUTS_DIR / f"{job_id}_combine.json",
        "main_json":      OUTPUTS_DIR / f"{job_id}_main_tactical_output.json",
        "frame_json":     OUTPUTS_DIR / f"{job_id}_frame_level_output.json",
        "shot_json":      OUTPUTS_DIR / f"{job_id}_shot_level_output.json",
        "weak_shot_json": OUTPUTS_DIR / f"{job_id}_weak_shots.json",
        "shuttle_json":   OUTPUTS_DIR / f"{job_id}_shuttle_trajectory_final.json",
        "transition_json":OUTPUTS_DIR / f"{job_id}_tactical_transition_matrix.json",
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
        match_id=combined.get("match_id", "match_001"),
        player_id=combined.get("player_id", "player_01"),
        confirmed_out_events=combined.get("out_of_court_events", []),
    )

    return JSONResponse({
        "job_id": job_id,
        "message": "Re-analysis completed using saved data.",
        "normal_total_shots":   tactical_summary["normal_total_shots"],
        "weak_total_shots":     tactical_summary["weak_total_shots"],
        "total_including_weak": tactical_summary["total_including_weak"],
        "shot_distribution":    tactical_summary["shot_distribution"],
        "shots_in_court":       tactical_summary["shots_in_court"],
        "shots_out":            tactical_summary["shots_out"],
        "left_out_count":       tactical_summary["left_out_count"],
        "right_out_count":      tactical_summary["right_out_count"],
        "main_json":      f"/outputs/{job_id}_main_tactical_output.json",
        "m2_json":        f"/outputs/{job_id}_m2.json",
        "combine_json":   f"/outputs/{job_id}_combine.json",
        "frame_json":     f"/outputs/{job_id}_frame_level_output.json",
        "shot_json":      f"/outputs/{job_id}_shot_level_output.json",
        "weak_shot_json": f"/outputs/{job_id}_weak_shots.json",
        "shuttle_json":   f"/outputs/{job_id}_shuttle_trajectory_final.json",
        "transition_json":f"/outputs/{job_id}_tactical_transition_matrix.json",
        "rally_patterns": tactical_summary["main_output_preview"].get("rally_patterns", []),
        "pattern_scores": tactical_summary["main_output_preview"].get("pattern_scores", []),
        "out_of_court_events": tactical_summary["main_output_preview"].get("out_of_court_events", []),
    })

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
