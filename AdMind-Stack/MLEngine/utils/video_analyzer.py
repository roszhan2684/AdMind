# utils/video_analyzer.py
import os
import cv2
import numpy as np
from collections import Counter, defaultdict
from PIL import Image

from utils.image_preprocessing import preprocess_image
from utils.feature_extraction import (
    detect_dominant_emotion,
    extract_color_palette,
    compute_layout_balance,
)
from utils.heatmap import generate_saliency_heatmap

def _is_video(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def analyze_video(
    video_path: str,
    *,
    fps_sample: float = 1.0,
    max_frames: int = 120,
    ocr_reader=None,
    object_model=None,
    clip_model=None,
) -> dict:
    """
    Sample frames from a video and run the existing image pipeline on them.
    Aggregates signals over time and returns a compact summary.

    Returns:
      {
        "media_type": "video",
        "duration_sec": float,
        "frames_analyzed": int,
        "fps_used": float,
        "top_emotions": [{"label": str, "count": int}],
        "objects_top": [{"label": str, "count": int}],
        "ocr_excerpt": str,
        "color_palette_global": [#hex,...],
        "layout_balance_avg": float,
        "keyframe_heatmaps": [filename, ...],
        "clip_embedding": [float,...]  # truncated (first 32)
      }
    """
    assert _is_video(video_path), f"Not a supported video file: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / max(1e-6, native_fps)

    # frame step (every N frames) to approximate `fps_sample`
    step = int(max(1, round(native_fps / max(0.1, fps_sample))))

    emotion_counter = Counter()
    object_counter = Counter()
    all_colors = []
    layout_vals = []
    all_ocr = []
    all_clip_vecs = []

    picked_heatmap_frames = []  # store (index, path)
    frames_analyzed = 0

    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            idx += 1
            continue

        # Convert BGR -> RGB and write temp image for reusing your image code
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tmp_img_path = f"__frame_{idx}.jpg"
        cv2.imwrite(tmp_img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        try:
            # (1) preprocess (float32 RGB 0..1) + PIL
            image_array = preprocess_image(tmp_img_path)
            pil_img = Image.fromarray(rgb)

            # (2) emotion
            emo = detect_dominant_emotion(image_array)
            if isinstance(emo, str):
                emotion_counter[emo] += 1

            # (3) color palette (take top 3 to keep global palette tight)
            colors = extract_color_palette(image_array, num_colors=5)
            all_colors.extend(colors[:3])

            # (4) layout balance
            layout_vals.append(compute_layout_balance(image_array))

            # (5) OCR
            if ocr_reader is not None:
                text = " ".join(ocr_reader.readtext(tmp_img_path, detail=0))
                if text:
                    all_ocr.append(text)

            # (6) Objects
            if object_model is not None:
                dets = object_model(tmp_img_path)
                labels = []
                for r in dets:
                    cls_ids = getattr(r.boxes, "cls", None)
                    if cls_ids is not None:
                        for i in cls_ids.cpu().numpy().tolist():
                            labels.append(object_model.names[int(i)])
                for l in set(labels):
                    object_counter[l] += 1

            # (7) CLIP embedding
            if clip_model is not None:
                vec = clip_model.encode(pil_img, convert_to_numpy=True)
                all_clip_vecs.append(vec)

            # (8) heatmap for a few keyframes (e.g., every ~10th sampled frame)
            if (frames_analyzed % 10) == 0:
                heatmap = generate_saliency_heatmap(tmp_img_path, alpha=0.45)
                picked_heatmap_frames.append((idx, heatmap))

            frames_analyzed += 1
            if frames_analyzed >= max_frames:
                break
        finally:
            try:
                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)
            except Exception:
                pass

        idx += 1

    cap.release()

    # Aggregate ----------------------------------------------------------------
    # Global color palette: cluster the accumulated hex colors by frequency
    palette_counter = Counter(all_colors)
    color_palette_global = [c for c, _ in palette_counter.most_common(5)]

    layout_balance_avg = round(float(np.mean(layout_vals))) if layout_vals else None

    # Average CLIP vector
    clip_embedding = None
    if all_clip_vecs:
        mat = np.stack(all_clip_vecs, axis=0)
        mean_vec = mat.mean(axis=0)
        clip_embedding = mean_vec.tolist()[:32]  # truncate

    # OCR sample (trim)
    ocr_excerpt = ""
    if all_ocr:
        merged = " ".join(all_ocr)
        ocr_excerpt = merged[:600]

    keyframe_heatmaps = [p for _, p in picked_heatmap_frames[:6]]

    result = {
        "media_type": "video",
        "duration_sec": round(float(duration_sec), 2),
        "frames_analyzed": int(frames_analyzed),
        "fps_used": float(fps_sample),
        "top_emotions": [{"label": k, "count": v} for k, v in emotion_counter.most_common(5)],
        "objects_top": [{"label": k, "count": v} for k, v in object_counter.most_common(8)],
        "ocr_excerpt": ocr_excerpt,
        "color_palette_global": color_palette_global,
        "layout_balance_avg": layout_balance_avg,
        "keyframe_heatmaps": keyframe_heatmaps,
        "clip_embedding": clip_embedding,
    }
    return result
