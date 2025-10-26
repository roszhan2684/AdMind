# utils/feature_extraction.py
import sys
import cv2
import numpy as np

# ---- TF/Keras alias shim for Apple Silicon (must be before importing `fer`) ----
import tensorflow as tf
import keras  # keras==2.15.0 in your venv

# Make libs that import `tensorflow.keras` work with standalone Keras
sys.modules["tensorflow.keras"] = keras
sys.modules["tensorflow.python.keras"] = keras
setattr(tf, "keras", keras)
# ------------------------------------------------------------------------------

from collections import Counter
from fer import FER

# Optional but improves accuracy as a fallback
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except Exception:
    HAS_DEEPFACE = False

# ---- Singletons (load once) ---------------------------------------------------
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_FER = FER(mtcnn=False)  # we'll feed cropped faces so mtcnn isn't needed


def _crop_faces(img_rgb: np.ndarray):
    """
    Detect faces with OpenCV and return a list of padded, upscaled face crops (RGB uint8).
    Falls back to returning the full image if no face is found.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    crops = []
    h, w = gray.shape
    for (x, y, fw, fh) in faces:
        pad = int(0.15 * max(fw, fh))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + fw + pad), min(h, y + fh + pad)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size:
            # Upscale small faces to help classifiers
            if min(crop.shape[:2]) < 96:
                new_w = max(128, crop.shape[1])
                new_h = max(128, crop.shape[0])
                crop = cv2.resize(crop, (new_w, new_h))
            crops.append(crop)

    return crops if crops else [img_rgb]


def detect_dominant_emotion(image_array: np.ndarray) -> str:
    """
    Robust emotion detection:
      1) Detect/crop face(s)
      2) Classify with FER
      3) If FER is neutral/weak, optionally fallback to DeepFace
    Returns a label string (e.g., 'sad'); if confidence is very low, returns 'sad (uncertain)'.
    """
    # Ensure uint8 RGB
    img_rgb = (image_array * 255).astype(np.uint8)
    crops = _crop_faces(img_rgb)

    labels, confidences = [], []

    for crop in crops:
        # ---- Step 1: FER (fast)
        fer_result = _FER.detect_emotions(crop)
        if fer_result:
            emodict = fer_result[0]["emotions"]
            fer_label = max(emodict, key=emodict.get)
            fer_conf = float(emodict[fer_label])
        else:
            fer_label, fer_conf = None, 0.0

        best_label, best_conf = fer_label, fer_conf

        # ---- Step 2: Fallback to DeepFace if neutral/weak
        if HAS_DEEPFACE and (best_label in (None, "neutral") or best_conf < 0.45):
            try:
                df = DeepFace.analyze(
                    crop, actions=["emotion"], enforce_detection=False
                )
                df0 = df[0] if isinstance(df, list) else df
                emo = df0.get("emotion") or df0.get("emotions") or {}
                if emo:
                    # DeepFace can return percents (0-100); normalize if needed
                    df_label = max(emo, key=emo.get)
                    raw = emo[df_label]
                    df_conf = float(raw) / 100.0 if raw > 1 else float(raw)
                    if df_conf > best_conf:
                        best_label, best_conf = df_label.lower(), df_conf
            except Exception:
                # If DeepFace fails, just keep FER result
                pass

        if best_label:
            labels.append(best_label)
            confidences.append(best_conf)

    if not labels:
        return "No face detected"

    # Majority vote; tie-break with highest confidence
    counts = Counter(labels)
    top_label, _ = counts.most_common(1)[0]
    top_conf = max(
        [c for l, c in zip(labels, confidences) if l == top_label], default=0.0
    )

    if top_conf < 0.35:
        return f"{top_label} (uncertain)"

    return top_label


def extract_color_palette(image_array: np.ndarray, num_colors: int = 5):
    """
    Extract dominant colors via k-means clustering on uint8 RGB pixels.
    Returns a list of hex color strings.
    """
    img_uint8 = (image_array * 255).astype(np.uint8)
    pixels = img_uint8.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)

    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]
    return hex_colors


def compute_layout_balance(image_array: np.ndarray) -> float:
    """
    Computes a simple left/right brightness balance metric (0-1).
    """
    gray = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    left_mean = np.mean(gray[:, : w // 2])
    right_mean = np.mean(gray[:, w // 2 :])
    balance = 1 - abs(left_mean - right_mean) / 255
    return round(float(balance), 2)
