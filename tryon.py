import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"


def load_assets():
    assets = {}
    for name in ("glasses", "hat", "mustache"):
        img = cv2.imread(f"assets/{name}.png", cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            raise FileNotFoundError(f"assets/{name}.png must exist and be a transparent PNG")
        assets[name] = img
    return assets


def create_landmarker():
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1))


def add_headroom(frame, fraction=0.35):
    pad = int(frame.shape[0] * fraction)
    return cv2.copyMakeBorder(frame, pad, 0, 0, 0, cv2.BORDER_REPLICATE)


def overlay(frame, img, center, width, angle_deg):
    h0, w0 = img.shape[:2]
    s = width / w0
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    x1, y1 = int(center[0] - nw / 2), int(center[1] - nh / 2)
    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x1 + nw, frame.shape[1]), min(y1 + nh, frame.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return frame
    part = img[fy1 - y1:fy2 - y1, fx1 - x1:fx2 - x1]
    alpha = part[:, :, 3:4] / 255.0
    roi = frame[fy1:fy2, fx1:fx2]
    frame[fy1:fy2, fx1:fx2] = (alpha * part[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    return frame


def apply_tryon(frame, landmarker, assets, items,
                glasses_scale=2.2, hat_scale=1.5, mustache_scale=1.4):
    """frame: BGR image. Returns (result_bgr, face_found)."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return frame, False
    lm = result.face_landmarks[0]
    pt = lambda i: np.array([lm[i].x * w, lm[i].y * h])

    p1, p2 = pt(33), pt(263)                 # outer eye corners
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    dx, dy = p2 - p1
    phi = math.atan2(dy, dx)
    angle = -math.degrees(phi)
    up = np.array([math.sin(phi), -math.cos(phi)])
    eye_dist = math.hypot(dx, dy)
    face_w = np.linalg.norm(pt(454) - pt(234))

    if "hat" in items:
        hat = assets["hat"]
        hat_w = face_w * hat_scale
        hat_h = hat_w * hat.shape[0] / hat.shape[1]
        center = pt(10) + up * (hat_h * 0.5 - 0.12 * face_w)
        frame = overlay(frame, hat, center, hat_w, angle)
    if "glasses" in items:
        center = (p1 + p2) / 2
        frame = overlay(frame, assets["glasses"], center, eye_dist * glasses_scale, angle)
    if "mustache" in items:
        mouth_w = np.linalg.norm(pt(291) - pt(61))
        center = (pt(164) + pt(0)) / 2
        frame = overlay(frame, assets["mustache"], center, mouth_w * mustache_scale, angle)
    return frame, True