# # main.py — AdMind ML Engine (Milestone 3 + Robust Video Emotions + Gemini Insight)
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import re
# import unicodedata
# import numpy as np
# import cv2
# from collections import Counter, defaultdict
# import json
# import requests

# # Try to load .env (optional)
# try:
#     from dotenv import load_toml as _load_toml  # moviepy bundles python-dotenv sometimes
# except Exception:
#     def _load_toml(*args, **kwargs): ...
# try:
#     from dotenv import load_dotenv as _load_env
#     _load_env()
# except Exception:
#     pass

# # ---- Config ----
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # set: export GEMINI_API_KEY=your_key
# GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

# # ---- Utils (your modules) ----
# from utils.heatmap import generate_saliency_heatmap
# from utils.image_preprocessing import preprocess_image
# from utils.feature_extraction import (
#     detect_dominant_emotion,
#     extract_color_palette,
#     compute_layout_balance,
# )
# from utils.zero_shot import clip_zero_shot_labels

# # ---- 3rd party ----
# import easyocr
# from ultralytics import YOLO
# from sentence_transformers import SentenceTransformer
# from PIL import Image
# from fer import FER

# # Optional NSFW classifier
# try:
#     from transformers import pipeline
# except Exception:
#     pipeline = None  # guarded below

# # ==========================================================
# # Flask
# # ==========================================================
# app = Flask(__name__)
# CORS(app)
# # app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # optional hard cap

# # ==========================================================
# # Model singletons (load once)
# # ==========================================================
# ocr_reader = easyocr.Reader(["en"])
# object_model = YOLO("yolov8n.pt")
# clip_model = SentenceTransformer("clip-ViT-B-32")
# fer_detector = FER(mtcnn=True)

# # Face detector for video (better crops -> better emotions)
# try:
#     face_model = YOLO("yolov8n-face.pt")  # auto-downloads if missing
# except Exception:
#     face_model = None

# # NSFW pipeline (optional)
# nsfw_pipe = None
# if pipeline is not None:
#     try:
#         nsfw_pipe = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device_map="auto")
#     except Exception:
#         nsfw_pipe = None

# # ==========================================================
# # Health
# # ==========================================================
# @app.route("/health", methods=["GET"])
# def health_check():
#     return jsonify({
#         "status": "ok",
#         "message": "ML Engine running",
#         "nsfw_ready": bool(nsfw_pipe),
#         "gemini_ready": bool(GEI()),
#     })

# # ==========================================================
# # Helpers
# # ==========================================================
# def GEI():
#     """Return API key if configured, else None."""
#     return GEMINI_API_KEY or None

# def gemini_generate_insight(feature_dict: dict, media_type: str) -> tuple[str | None, str | None]:
#     """
#     Calls Google Generative Language (Gemini) to produce a concise JSON insight.
#     Returns (insight_text_json, error_message_or_None).
#     """
#     api_key = GEI()
#     if not api_key:
#         return None, "GEMINI_API_KEY not configured"

#     url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

#     # Build a compact summary of the extracted features so we don’t blow token limits
#     def shorten(s, n=600):
#         s = (s or "").strip()
#         return s if len(s) <= n else s[: n - 3] + "..."

#     summary = {}
#     if media_type == "image":
#         summary = {
#             "media_type": "image",
#             "dominant_emotion": feature_dict.get("dominant_emotion"),
#             "emotion_confidence": feature_dict.get("emotion_confidence"),
#             "face_count": feature_dict.get("face_count"),
#             "layout_balance": feature_dict.get("layout_balance"),
#             "top_categories": feature_dict.get("top_categories"),
#             "detected_objects": feature_dict.get("detected_objects"),
#             "brands": [b.get("brand") for b in (feature_dict.get("brands") or [])],
#             "ocr_text_excerpt": shorten(feature_dict.get("text_content"), 800),
#             "creative_score": feature_dict.get("creative_score"),
#             "nsfw_safe": feature_dict.get("nsfw", {}).get("is_safe"),
#         }
#     else:
#         ve = feature_dict.get("video_emotions", {}) or {}
#         summary = {
#             "media_type": "video",
#             "final_emotion": (ve.get("summary") or {}).get("final_top"),
#             "avg_faces_per_sec": (ve.get("summary") or {}).get("avg_faces_per_sec"),
#             "top_emotion_counts": (ve.get("summary") or {}).get("counts"),
#             "layout_balance_avg": feature_dict.get("layout_balance_avg"),
#             "objects_top": feature_dict.get("objects_top"),
#             "brands": [b.get("brand") for b in (feature_dict.get("brands") or [])],
#             "ocr_excerpt": shorten(feature_dict.get("ocr_excerpt"), 800),
#             "creative_score": feature_dict.get("creative_score"),
#             "nsfw_safe": (feature_dict.get("nsfw") or {}).get("is_safe"),
#         }

#     prompt = f"""
# You are an ad-creatives intelligence analyst. Given the extracted features of an {media_type} ad below,
# return a compact JSON object with the following keys **only**:

# - "emotion": a single primary emotion word (e.g., "happy", "excited", "calm", "funny", "luxurious").
# - "insight_summary": 3–5 short sentences (one paragraph) explaining what this creative is conveying,
#   which elements drive attention (text/objects/colors/motion), and why it might perform well or poorly.
# - "suggestions": an array of 3–6 concrete suggestions to potentially improve performance (CTA, pacing, framing, copy tweaks, safety/brand issues, etc).

# Input features (summarized):
# {json.dumps(summary, ensure_ascii=False, indent=2)}

# Output strictly as JSON, no extra commentary. Example:
# {{
#   "emotion": "playful",
#   "insight_summary": "…",
#   "suggestions": ["…","…","…"]
# }}
# """.strip()

#     payload = {
#         "contents": [{"role": "user", "parts": [{"text": prompt}]}],
#         "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400},
#     }

#     try:
#         r = requests.post(url, json=payload, timeout=45)
#         if r.status_code != 200:
#             return None, f"Gemini API HTTP {r.status_code}: {r.text[:300]}"

#         data = r.json()
#         # Extract text
#         text = ""
#         try:
#             text = data["candidates"][0]["content"]["parts"][0]["text"]
#         except Exception:
#             return None, f"Gemini response parsing error: {json.dumps(data)[:300]}"

#         # Strip ```json fences if present and parse
#         cleaned = text.strip()
#         if cleaned.startswith("```"):
#             cleaned = cleaned.strip("`")
#             # remove optional leading 'json'
#             if cleaned.lower().startswith("json"):
#                 cleaned = cleaned[4:].lstrip()
#         # Ensure JSON
#         try:
#             _ = json.loads(cleaned)
#             return cleaned, None
#         except Exception:
#             # Fall back to plain text in "insight_summary" field if not valid JSON
#             return json.dumps({"insight_summary": cleaned}), None

#     except Exception as e:
#         return None, f"Gemini request error: {e}"

# def _emotion_with_conf_and_faces(img_rgb_float01):
#     try:
#         img_uint8 = (img_rgb_float01 * 255).astype(np.uint8)
#         results = fer_detector.detect_emotions(img_uint8) or []
#         face_count = len(results)
#         if face_count == 0:
#             return "No face detected", 0.0, 0
#         per_face = []
#         for r in results:
#             emo = r.get("emotions", {}) or {}
#             if not emo:
#                 continue
#             best_label = max(emo, key=emo.get)
#             per_face.append((best_label, float(emo[best_label])))
#         if not per_face:
#             return "No face detected", 0.0, face_count
#         labels = [l for (l, _s) in per_face]
#         dominant = max(set(labels), key=labels.count)
#         confs = [s for (l, s) in per_face if l == dominant]
#         confidence = float(np.mean(confs)) if confs else 0.0
#         return dominant, confidence, face_count
#     except Exception:
#         return detect_dominant_emotion(img_rgb_float01), 0.0, 0

# def _normalize_cosine_to_01(cos_sim):
#     lo, hi = -0.2, 0.6
#     x = (float(cos_sim) - lo) / (hi - lo)
#     return max(0.0, min(1.0, x))

# def _caption_candidates(ocr_text, detected_labels):
#     cands = []
#     txt = (ocr_text or "").strip()
#     if txt:
#         cands.extend([txt, txt.capitalize(), txt.split("\n")[0][:120]])
#     if detected_labels:
#         objlist = ", ".join(detected_labels[:3])
#         cands.extend([
#             f"A {objlist} focused ad creative",
#             f"An ad featuring {objlist}",
#         ])
#     cands.extend([
#         "A person reacting emotionally in an advertisement",
#         "A product-focused ad creative with bold headline text",
#         "An app install ad with a strong call to action",
#     ])
#     uniq, seen = [], set()
#     for s in cands:
#         s = (s or "").strip()
#         if s and s.lower() not in seen:
#             uniq.append(s); seen.add(s.lower())
#     return uniq[:12]

# def _clip_alignment(image_pil, ocr_text, detected_labels):
#     img_emb = clip_model.encode(image_pil, convert_to_numpy=True, normalize_embeddings=True)
#     ocr_score_raw, ocr_score01 = None, None
#     if (ocr_text or "").strip():
#         txt_emb = clip_model.encode([ocr_text], convert_to_numpy=True, normalize_embeddings=True)[0]
#         ocr_score_raw = float(np.dot(img_emb, txt_emb))
#         ocr_score01 = _normalize_cosine_to_01(ocr_score_raw)

#     cands = _caption_candidates(ocr_text, detected_labels)
#     if cands:
#         txt_embs = clip_sim = clip_model.encode(cands, convert_to_numpy=True, normalize_embeddings=True)
#         sims = (img_emb @ clip_sim.T).tolist()
#         best_idx = int(np.argmax(sims))
#         best_raw = float(sims[best_idx])
#         best01 = _normalize_cosine_to_01(best_raw)
#         candidates = [
#             {"text": c, "score_raw": float(s), "score": _normalize_cosine_to_01(s)}
#             for c, s in zip(cands, sims)
#         ]
#         best = {"text": cands[best_idx], "score_raw": best_raw, "score": best01}
#     else:
#         best, candidates = {"text": "", "score_raw": 0.0, "score": 0.0}, []

#     return {
#         "image_embedding_dim": len(img_emb),
#         "ocr_alignment": None if ocr_score01 is None else {
#             "text": ocr_text, "score_raw": ocr_score_raw, "score": ocr_score01
#         },
#         "best_caption": best,
#         "candidates": sorted(candidates, key=lambda x: x["score_raw"], reverse=True)[:5],
#     }

# def _heuristic_creative_score(layout_balance, face_count, has_text, obj_count, align_best01, brand_hits, nsfw_safe):
#     bal = float(layout_balance or 0.0)
#     face = min(1.0, float(face_count or 0.0))
#     text = 1.0 if has_text else 0.0
#     objs = 1.0 if (obj_count or 0) > 0 else 0.0
#     align = float(max(0.0, min(1.0, align_best01)))
#     brands = 1.0 if (brand_hits or 0) > 0 else 0.0
#     safe = 1.0 if nsfw_safe else 0.0
#     score01 = (0.25 * bal + 0.15 * face + 0.10 * text + 0.10 * objs +
#                0.20 * align + 0.10 * brands + 0.10 * safe)
#     return int(round(score01 * 100))

# # --- Brand detection (heuristic from OCR) ---
# _BRANDS = {
#     "apple": ["iphone", "ipad", "macbook", "ios"],
#     "google": ["android", "pixel", "gmail", "youtube"],
#     "youtube": ["subscribe", "yt"],
#     "amazon": ["prime", "alexa"],
#     "meta": ["facebook", "instagram", "whatsapp"],
#     "facebook": ["fb"],
#     "instagram": ["ig", "insta"],
#     "whatsapp": ["wa"],
#     "tiktok": ["tik tok"],
#     "netflix": ["nflx"],
#     "spotify": ["playlist"],
#     "adidas": [],
#     "nike": ["just do it", "air max"],
#     "starbucks": ["coffee"],
#     "mcdonalds": ["mcdonald's", "mcd", "mc donalds"],
#     "uber": [],
#     "lyft": [],
#     "snapchat": ["snap"],
#     "x": ["twitter"],
# }
# _WORD = re.compile(r"[A-Za-z0-9\-\']{2,}")

# def _norm(s: str) -> str:
#     s = unicodedata.normalize("NFKC", s or "")
#     return s.casefold()

# def _detect_brands_from_text(ocr_text: str):
#     text = _norm(ocr_text or "")
#     tokens = set(m.group(0).casefold() for m in _WORD.finditer(text))
#     hits = []
#     for brand, synonyms in _BRANDS.items():
#         brand_n = _norm(brand)
#         found = False
#         conf = 0.0
#         evidence = None
#         if brand_n in tokens:
#             found, conf, evidence = True, 0.95, brand
#         if not found and brand_n in text:
#             found, conf, evidence = True, 0.8, brand
#         if not found:
#             for syn in synonyms:
#                 syn_n = _norm(syn)
#                 if syn_n in tokens:
#                     found, conf, evidence = True, 0.85, syn
#                     break
#                 if syn_n in text:
#                     found, conf, evidence = True, 0.7, syn
#                     break
#         if found:
#             hits.append({"brand": brand, "source": "ocr_text", "confidence": round(conf, 2), "evidence": evidence})
#     return sorted(hits, key=lambda x: x["confidence"], reverse=True)[:8]

# def _nsfw_scores(image_pil):
#     if nsfw_pipe is None:
#         return {"available": False, "scores": {}, "top_label": "unknown", "top_score": 0.0, "is_safe": True}
#     try:
#         preds = nsfw_pipe(image_pil)
#         scores = {p["label"].lower(): float(p["score"]) for p in preds}
#         top = max(scores.items(), key=lambda kv: kv[1]) if scores else ("unknown", 0.0)
#         nsfw_prob = scores.get("nsfw", 0.0)
#         sfw_prob = scores.get("sfw", 0.0)
#         is_safe = nsfw_prob < 0.5 or (sfw_prob >= nsfw_prob)
#         return {
#             "available": True,
#             "scores": {k: round(v, 4) for k, v in scores.items()},
#             "top_label": top[0],
#             "top_score": round(float(top[1]), 4),
#             "is_safe": bool(is_safe),
#         }
#     except Exception:
#         return {"available": False, "scores": {}, "top_label": "unknown", "top_score": 0.0, "is_safe": True}

# # ---------- Video emotion helpers ----------
# def _clahe_rgb(img_bgr):
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l2 = clahe.apply(l)
#     lab2 = cv2.merge([l2, a, b])
#     return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# def _gamma(img_bgr, gamma=1.2):
#     inv = 1.0 / max(gamma, 1e-6)
#     table = ((np.arange(256) / 255.0) ** inv * 255.0).astype(np.uint8)
#     return cv2.LUT(img_bgr, table)

# def _prep_face_crop(frame_bgr):
#     img = _clahe_rgb(frame_bgr)
#     img = _gamma(img, gamma=1.2)
#     return img

# def _detect_faces_bboxes(frame_bgr, conf_thresh=0.25):
#     if face_model is None:
#         return []
#     try:
#         r = face_model.predict(source=frame_bgr, verbose=False)[0]
#         boxes = []
#         for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
#             if float(c) >= conf_thresh:
#                 x1, y1, x2, y2 = map(int, b.tolist())
#                 boxes.append([max(0, x1), max(0, y1), max(0, x2), max(0, y2)])
#         return boxes
#     except Exception:
#         return []

# def _fer_on_crop(bgr_crop):
#     rgb = cv2.cvtColor(_prep_face_crop(bgr_crop), cv2.COLOR_BGR2RGB)
#     res = fer_detector.detect_emotions(rgb) or []
#     if not res:
#         return None, 0.0
#     emo = res[0].get("emotions", {})
#     if not emo:
#         return None, 0.0
#     lab = max(emo, key=emo.get)
#     return lab, float(emo[lab])

# def _ema(prev, cur, alpha=0.4):
#     return alpha * cur + (1.0 - alpha) * prev

# def analyze_video_emotions_every_sec(video_path, fps_used=1, max_secs=300):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError("OpenCV could not open video")

#     fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     total_frames = int(cap.get_crcPropFrameCount() if hasattr(cap, "get_crcPropFrameCount") else cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     duration = total_frames / max(fps_src, 1e-6)

#     step = max(int(round(fps_src / max(fps_used, 1))), 1)
#     labels = ["happy", "angry", "sad", "fear", "surprise", "disgust", "neutral"]
#     ema_scores = {l: 0.0 for l in labels}

#     per_second = []
#     global_counts = Counter()
#     global_scores = defaultdict(float)
#     total_faces_seen = 0

#     frame_idx = 0
#     analyzed = 0

#     while True:
#         ret = cap.grab()
#         if not ret:
#             break

#         if frame_idx % step == 0:
#             ok, frame = cap.retrieve()
#             if not ok or frame is None:
#                 break

#             t_sec = frame_idx / max(fps_src, 1e-6)
#             if t_sec > max_secs:
#                 break

#             bboxes = _detect_faces_bboxes(frame) or []
#             per_face = []
#             if bboxes:
#                 for (x1, y1, x2, y2) in bboxes[:4]:
#                     crop = frame[y1:y2, x1:x2].copy()
#                     if crop.size > 0:
#                         lab, sc = _fer_on_crop(crop)
#                         if lab:
#                             per_face.append((lab, sc))
#             else:
#                 lab, sc = _fer_on_crop(frame)
#                 if lab:
#                     per_face.append((lab, sc))

#             face_count_this_sec = len(per_face)
#             total_faces_seen += face_count_this_sec

#             sec_scores = defaultdict(float)
#             for lab, sc in per_face:
#                 sec_scores[lab] += sc
#             if sec_scores:
#                 m = max(sec_scores.values())
#                 if m > 0:
#                     for k in sec_scores:
#                         sec_scores[k] /= m

#             if sec_scores:
#                 for l in labels:
#                     ema_scores[l] = _ema(ema_scores[l], sec_scores.get(l, 0.0), alpha=0.4)

#             top_label = max(ema_scores, key=ema_scores.get) if ema_scores else "neutral"

#             per_second.append({
#                 "t": round(t_sec, 2),
#                 "top": top_label,
#                 "face_count": face_count_this_sec,
#                 "scores": {l: round(ema_scores[l], 3) for l in labels if ema_scores[l] > 0},
#             })

#             global_counts[top_label] += 1
#             for l in labels:
#                 global_scores[l] += ema_scores[l]

#             analyzed += 1

#         frame_idx += 1

#     cap.release()

#     avg_scores = {l: round(global_scores[l] / max(analyzed, 1), 3) for l in labels}
#     final_top = max(avg_scores, key=avg_scores.get) if avg_scores else "neutral"
#     avg_faces_per_sec = round(float(total_faces_seen) / max(analyzed, 1), 2)

#     return {
#         "duration": round(duration, 2),
#         "frames_analyzed": analyzed,
#         "fps_used": fps_used,
#         "per_second": per_second,
#         "summary": {
#             "final_top": final_top,
#             "avg_scores": avg_scores,
#             "counts": dict(global_counts),
#             "avg_faces_per_sec": avg_faces_per_sec,
#         },
#     }

# # ==========================================================
# # Image analyze
# # ==========================================================
# @app.route("/analyze", methods=["POST"])
# def analyze_ad():
#     file = request.files.get("file")
#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400

#     tmp_name = "input.jpg"
#     file.save(tmp_name)

#     try:
#         image_array = preprocess_image(tmp_name)
#         image_pil = Image.open(tmp_name).convert("RGB")

#         emotion_label, emotion_conf, face_count = _emotion_with_conf_and_faces(image_array)
#         if emotion_label == "No face detected":
#             emotion_label = detect_dominant_emotion(image_array)

#         colors = extract_color_palette(image_array)
#         layout_balance = compute_layout_balance(image_array)
#         heatmap_path = generate_saliency_heatmap(tmp_name, alpha=0.45)

#         ocr_text = " ".join(ocr_reader.readtext(tmp_name, detail=0)).strip()

#         detections = object_model(tmp_name)
#         detected_labels = []
#         for r in detections:
#             cls_ids = getattr(r.boxes, "cls", None)
#             if cls_ids is not None:
#                 for i in cls_ids.cpu().numpy().tolist():
#                     detected_labels.append(object_model.names[int(i)])
#         detected_labels = sorted(list(set(detected_labels)))[:10]

#         img_vec = clip_model.encode(image_pil, convert_to_numpy=True)
#         clip_embedding = img_vec.tolist()[:32]

#         top_categories = clip_zero_shot_labels(clip_model, clip_model, image_pil, top_k=3)

#         alignment = _clip_alignment(image_pil, ocr_text, detected_labels)
#         align_best01 = float(alignment.get("best_caption", {}).get("score", 0.0))

#         brand_hits = _detect_brands_from_text(ocr_text)
#         nsfw = _nsfw_scores(image_pil)

#         creative_score = _heuristic_creative_score(
#             layout_balance=layout_balance,
#             face_count=face_count,
#             has_text=bool(ocr_text),
#             obj_count=len(detected_labels),
#             align_best01=align_best01,
#             brand_hits=len(brand_hits),
#             nsfw_safe=nsfw.get("is_safe", True),
#         )

#         # Gemini insight (JSON text)
#         insight_text, insight_err = gemini_generate_insight({
#             "media_type": "image",
#             "dominant_emotion": emotion_label,
#             "emotion_confidence": round(float(emotion_conf), 3),
#             "face_count": int(face_count),
#             "color_palette": colors,
#             "layout_balance": layout_balance,
#             "text_content": ocr_text,
#             "detected_objects": detected_labels,
#             "top_categories": top_categories,
#             "alignment": alignment,
#             "brands": brand_hits,
#             "nsfw": nsfw,
#             "creative_score": creative_score,
#         }, media_type="image")

#         result = {
#             "media_type": "image",
#             "dominant_emotion": emotion_label,
#             "emotion_confidence": round(float(emotion_conf), 3),
#             "face_count": int(face_count),
#             "color_palette": colors,
#             "layout_balance": layout_balance,
#             "heatmap_url": heatmap_path,
#             "text_content": ocr_text,
#             "detected_objects": detected_labels,
#             "clip_embedding": clip_embedding,
#             "top_categories": top_categories,
#             "alignment": alignment,
#             "brands": brand_hits,
#             "nsfw": nsfw,
#             "creative_score": creative_score,
#         }
#         if insight_text:
#             result["insight"] = json.loads(insight_text)
#         if insight_err:
#             result["insight_error"] = insight_err

#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     finally:
#         try:
#             if os.path.exists(tmp_name):
#                 os.remove(tmp_name)
#         except Exception:
#             pass

# # ==========================================================
# # Video analyze (per-second face-crop FER + aggregates) + Gemini
# # ==========================================================
# @app.route("/analyze_video", methods=["POST"])
# def analyze_ad_video():
#     file = request.files.get("file")
#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400

#     file.seek(0, os.SEEK_END)
#     size_mb = file.tell() / (1024 * 1024)
#     file.seek(0)
#     if size_mb > 500:
#         return jsonify({"error": "Video too large (>500MB). Try a shorter clip."}), 400

#     ext = os.path.splitext(file.filename or "")[-1] or ".mp4"
#     tmp_name = f"input_video{ext}"
#     file.save(tmp_name)

#     try:
#         # Emotions per second
#         video_emotions = analyze_video_emotions_every_sec(tmp_name, fps_used=1, max_secs=300)
#         duration_sec = video_emotions["duration"]
#         frames_analyzed = video_emotions["frames_analyzed"]
#         # Additional aggregates at ~1fps
#         cap = cv2.VideoCapture(tmp_name)
#         if not cap.isOpened():
#             raise RuntimeError("Could not open video")

#         native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#         duration_sec_exact = frame_count / max(1e-6, native_fps)

#         fps_sample = 1.0
#         step = int(max(1, round(native_fps / max(0.1, fps_sample))))

#         color_bag = []
#         layout_vals = []
#         all_ocr = []
#         object_counter = Counter()
#         clip_vecs = []

#         nsfw_votes_safe = 0
#         nsfw_checked = 0
#         keyframe_heatmaps = []

#         idx = 0
#         analyzed_aux = 0
#         max_frames = 120

#         while True:
#             ret = cap.grab()
#             if not ret:
#                 break
#             if idx % step != 0:
#                 idx += 1
#                 continue

#             ok, frame = cap.retrieve()
#             if not ok or frame is None:
#                 idx += 1
#                 continue

#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             tmp_img_path = f"__frame_{idx}.jpg"
#             cv2.imwrite(tmp_img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

#             try:
#                 image_array = preprocess_image(tmp_img_path)
#                 pil_img = Image.fromarray(rgb)

#                 color_bag.extend(extract_color_palette(image_array, num_colors=5)[:3])
#                 layout_vals.append(compute_layout_balance(image_array))

#                 ocr_text = " ".join(ocr_reader.readtext(tmp_img_path, detail=0)).strip()
#                 if ocr_text:
#                     all_ocr.append(ocr_text)

#                 dets = object_model(tmp_img_path)
#                 labels = []
#                 for r in dets:
#                     cls_ids = getattr(r.boxes, "cls", None)
#                     if cls_ids is not None:
#                         for i in cls_ids.cpu().numpy().tolist():
#                             labels.append(object_model.names[int(i)])
#                 for l in set(labels):
#                     object_counter[l] += 1

#                 vec = clip_model.encode(pil_img, convert_to_numpy=True)
#                 clip_vecs.append(vec)

#                 if (analyzed_aux % 10) == 0:
#                     hm = generate_saliency_heatmap(tmp_img_path, alpha=0.45)
#                     keyframe_heatmaps.append(hm)

#                 if (analyzed_aux % 5) == 0:
#                     ns = _nsfw_scores(pil_img)
#                     if ns.get("is_safe", True):
#                         nsfw_votes_safe += 1
#                     nsfw_checked += 1

#                 analyzed_aux += 1
#                 if analyzed_aux >= max_frames:
#                     pass
#             finally:
#                 try:
#                     if os.path.exists(tmp_img_path):
#                         os.remove(tmp_img_path)
#                 except Exception:
#                     pass

#             if analyzed_aux >= max_frames:
#                 break
#             idx += 1

#         cap.release()

#         duration_sec_out = round(float(duration_sec_exact), 2)
#         palette_counter = Counter(color_bag)
#         color_palette_global = [c for c, _ in palette_counter.most_common(5)]
#         layout_balance_avg = round(float(np.mean(layout_vals)), 2) if layout_vals else None

#         ocr_excerpt = ""
#         if all_ocr:
#             merged = " ".join(all_ocr)
#             ocr_excerpt = merged[:600]

#         objects_top = [{"label": k, "count": int(v)} for k, v in object_counter.most_common(8)]

#         clip_embedding = None
#         if clip_vecs:
#             mat = np.stack(clip_vecs, axis=0)
#             clip_embedding = mat.mean(axis=0).tolist()[:32]

#         combined_brands = Counter()
#         for txt in all_ocr:
#             for hit in _detect_brands_from_text(txt):
#                 combined_brands[hit["brand"]] += 1
#         brands = [{"brand": b, "source": "ocr_text", "confidence": 0.8, "evidence": b}
#                   for b, _cnt in combined_brands.most_common(8)]

#         nsfw_safe = True
#         if nsfw_checked > 0:
#             nsfw_safe = (nsfw_votes_safe / nsfw_checked) >= 0.5
#         nsfw_summary = {
#             "available": nsfw_pipe is not None,
#             "frames_checked": int(nsfw_checked),
#             "safe_votes": int(nsfw_votes_safe),
#             "is_safe": bool(nsfw_safe),
#         }

#         # Map emotion summary to root keys for UI compatibility
#         final_top = (video_emotions.get("summary") or {}).get("final_top")
#         avg_faces_per_sec = (video_emotions.get("summary") or {}).get("avg_faces_per_sec")
#         top_counts = (video_emotions.get("summary") or {}).get("counts") or {}
#         top_emotions = [{"label": k, "count": int(v)} for k, v in sorted(top_counts.items(), key=lambda kv: kv[1], reverse=True)]

#         # Gemini insight
#         insight_text, insight_err = gemini_generate_insight({
#             "media_type": "video",
#             **{
#                 "video_emotions": video_emotions,
#                 "layout_balance_avg": layout_balance_avg,
#                 "objects_top": objects_top,
#                 "ocr_excerpt": ocr_excerpt,
#                 "brands": brands,
#                 "nsfw": nsfw_summary,
#                 "creative_score": None,  # will fill after computing
#             },
#         }, media_type="video")

#         creative_score = _heuristic_creative_score(
#             layout_balance=layout_balance_avg or 0.0,
#             face_count=avg_faces_per_sec,
#             has_text=bool(ocr_excerpt),
#             obj_count=sum(v["count"] for v in objects_top),
#             align_best01=0.0,
#             brand_hits=len(brands),
#             nsfw_safe=nsfw_safe,
#         )

#         # now update insight dict with final score
#         if insight_text:
#             try:
#                 tmp = json.loads(insight_text)
#                 # inject score if not present
#                 tmp.setdefault("creative_score", creative_score)
#                 insight_text = json.dumps(tmp)
#             except Exception:
#                 pass

#         result = {
#             "media_type": "video",
#             "duration_sec": duration_sec_out,
#             "frames_analyzed": int(frames_analyzed),
#             "fps_used": float(video_emotions["fps_used"]),
#             "video_emotions": video_emotions,  # full timeline + summary
#             # root-level convenience fields (so UI can keep using them)
#             "dominant_emotion": final_top,
#             "avg_faces_per_frame": avg_faces_per_sec,  # backward-compat label
#             "top_emotions": top_emotions,
#             # other aggregates:
#             "objects_top": objects_top,
#             "ocr_excerpt": ocr_excerpt,
#             "color_palette_global": color_palette_global,
#             "layout_balance_avg": layout_balance_avg,
#             "keyframe_heatmaps": keyframe_heatmaps[:6],
#             "clip_embedding": clip_embedding,
#             "brands": brands,
#             "nsfw": nsfw_summary,
#             "alignment": {
#                 "frame_index": None,
#                 "best_caption": {"text": "A person reacting emotionally in an advertisement", "score": 0.0},
#                 "alignment_detail": {},
#             },
#             "creative_score": creative_score,
#         }
#         if insight_text:
#             result["insight"] = json.loads(insight_text)
#         if insight_err:
#             result["insight_error"] = insight_err

#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         try:
#             if os.path.exists(tmp_name):
#                 os.remove(tmp_name)
#         except Exception:
#             pass

# # ==========================================================
# # Run
# # ==========================================================
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)
# main.py — AdMind ML Engine (robust video emotions + Gemini insight)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import unicodedata
import numpy as np
import cv2
from collections import Counter, defaultdict
import json
import requests

# Optional: imageio fallback for video decoding
import imageio.v3 as iio

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- Config ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # set: export GEMINI_API_KEY=your_key
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

# ---- Utils (your modules) ----
from utils.heatmap import generate_saliency_heatmap
from utils.image_preprocessing import preprocess_image
from utils.feature_extraction import (
    detect_dominant_emotion,
    extract_color_palette,
    compute_layout_balance,
)
from utils.zero_shot import clip_zero_shot_labels

# ---- 3rd party ----
import easyocr
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from PIL import Image
from fer import FER

# Optional NSFW classifier
try:
    from transformers import pipeline
except Exception:
    pipeline = None  # guarded below

# ==========================================================
# Flask
# ==========================================================
app = Flask(__name__)
CORS(app)

# ==========================================================
# Model singletons (load once)
# ==========================================================
ocr_reader = easyocr.Reader(["en"])
object_model = YOLO("yolov8n.pt")
clip_model = SentenceTransformer("clip-ViT-B-32")
fer_detector = FER(mtcnn=True)

# Face detector for video (better crops -> better emotions)
try:
    face_model = YOLO("yolov8n-face.pt")  # auto-downloads if missing
except Exception:
    face_model = None

# NSFW pipeline (optional)
nsfw_pipe = None
if pipeline is not None:
    try:
        nsfw_pipe = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device_map="auto")
    except Exception:
        nsfw_pipe = None

# ==========================================================
# Health
# ==========================================================
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "ML Engine running",
        "nsfw_ready": bool(nsfw_pipe),
        "gemini_ready": bool(GEMINI_API_KEY),
    })

# ==========================================================
# Helpers
# ==========================================================
def gemini_generate_insight(feature_dict: dict, media_type: str):
    """
    Calls Google Generative Language (Gemini) to produce a concise JSON insight.
    Returns (insight_text_json, error_message_or_None).
    """
    api_key = GEMINI_API_KEY
    if not api_key:
        return None, "GEMINI_API_KEY not configured"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    def shorten(s, n=600):
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 3] + "..."

    if media_type == "image":
        summary = {
            "media_type": "image",
            "dominant_emotion": feature_dict.get("dominant_emotion"),
            "emotion_confidence": feature_dict.get("emotion_confidence"),
            "face_count": feature_dict.get("face_count"),
            "layout_balance": feature_dict.get("layout_balance"),
            "top_categories": feature_dict.get("top_categories"),
            "detected_objects": feature_dict.get("detected_objects"),
            "brands": [b.get("brand") for b in (feature_dict.get("brands") or [])],
            "ocr_text_excerpt": shorten(feature_dict.get("text_content"), 800),
            "creative_score": feature_dict.get("creative_score"),
            "nsfw_safe": feature_dict.get("nsfw", {}).get("is_safe"),
        }
    else:
        ve = feature_dict.get("video_emotions", {}) or {}
        summary = {
            "media_type": "video",
            "final_emotion": (ve.get("summary") or {}).get("final_top"),
            "avg_faces_per_sec": (ve.get("summary") or {}).get("avg_faces_per_sec"),
            "top_emotion_counts": (ve.get("summary") or {}).get("counts"),
            "layout_balance_avg": feature_dict.get("layout_balance_avg"),
            "objects_top": feature_dict.get("objects_top"),
            "brands": [b.get("brand") for b in (feature_dict.get("brands") or [])],
            "ocr_excerpt": shorten(feature_dict.get("ocr_excerpt"), 800),
            "creative_score": feature_dict.get("creative_score"),
            "nsfw_safe": (feature_dict.get("nsfw") or {}).get("is_safe"),
        }

    prompt = f"""
You are an ad-creatives intelligence analyst. Given the extracted features of an {media_type} ad below,
return a compact JSON object with the following keys **only**:

- "emotion": a single primary emotion word (e.g., "happy", "excited", "calm", "funny", "luxurious").
- "insight_summary": 3–5 short sentences (one paragraph) explaining what this creative is conveying,
  which elements drive attention (text/objects/colors/motion), and why it might perform well or poorly.
- "suggestions": an array of 3–6 concrete suggestions to potentially improve performance (CTA, pacing, framing, copy tweaks, safety/brand issues, etc).

Input features (summarized):
{json.dumps(summary, ensure_ascii=False, indent=2)}

Output strictly as JSON, no extra commentary. Example:
{{
  "emotion": "playful",
  "insight_summary": "…",
  "suggestions": ["…","…","…"]
}}
""".strip()

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400},
    }

    try:
        r = requests.post(url, json=payload, timeout=45)
        if r.status_code != 200:
            return None, f"Gemini API HTTP {r.status_code}: {r.text[:300]}"

        data = r.json()
        # Extract text
        text = ""
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return None, f"Gemini response parsing error: {json.dumps(data)[:300]}"

        # Strip ```json fences if present and parse
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        try:
            _ = json.loads(cleaned)
            return cleaned, None
        except Exception:
            return json.dumps({"insight_summary": cleaned}), None

    except Exception as e:
        return None, f"Gemini request error: {e}"

def _emotion_with_conf_and_faces(img_rgb_float01):
    try:
        img_uint8 = (img_rgb_float01 * 255).astype(np.uint8)
        results = fer_detector.detect_emotions(img_uint8) or []
        face_count = len(results)
        if face_count == 0:
            return "No face detected", 0.0, 0
        per_face = []
        for r in results:
            emo = r.get("emotions", {}) or {}
            if not emo:
                continue
            best_label = max(emo, key=emo.get)
            per_face.append((best_label, float(emo[best_label])))
        if not per_face:
            return "No face detected", 0.0, face_count
        labels = [l for (l, _s) in per_face]
        dominant = max(set(labels), key=labels.count)
        confs = [s for (l, s) in per_face if l == dominant]
        confidence = float(np.mean(confs)) if confs else 0.0
        return dominant, confidence, face_count
    except Exception:
        return detect_dominant_emotion(img_rgb_float01), 0.0, 0

def _normalize_cosine_to_01(cos_sim):
    lo, hi = -0.2, 0.6
    x = (float(cos_sim) - lo) / (hi - lo)
    return max(0.0, min(1.0, x))

def _caption_candidates(ocr_text, detected_labels):
    cands = []
    txt = (ocr_text or "").strip()
    if txt:
        cands.extend([txt, txt.capitalize(), txt.split("\n")[0][:120]])
    if detected_labels:
        objlist = ", ".join(detected_labels[:3])
        cands.extend([
            f"A {objlist} focused ad creative",
            f"An ad featuring {objlist}",
        ])
    cands.extend([
        "A person reacting emotionally in an advertisement",
        "A product-focused ad creative with bold headline text",
        "An app install ad with a strong call to action",
    ])
    uniq, seen = [], set()
    for s in cands:
        s = (s or "").strip()
        if s and s.lower() not in seen:
            uniq.append(s); seen.add(s.lower())
    return uniq[:12]

def _clip_alignment(image_pil, ocr_text, detected_labels):
    img_emb = clip_model.encode(image_pil, convert_to_numpy=True, normalize_embeddings=True)
    ocr_score_raw, ocr_score01 = None, None
    if (ocr_text or "").strip():
        txt_emb = clip_model.encode([ocr_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        ocr_score_raw = float(np.dot(img_emb, txt_emb))
        ocr_score01 = _normalize_cosine_to_01(ocr_score_raw)

    cands = _caption_candidates(ocr_text, detected_labels)
    if cands:
        txt_embs = clip_model.encode(cands, convert_to_numpy=True, normalize_embeddings=True)
        sims = (img_emb @ txt_embs.T).tolist()
        best_idx = int(np.argmax(sims))
        best_raw = float(sims[best_idx])
        best01 = _normalize_cosine_to_01(best_raw)
        candidates = [
            {"text": c, "score_raw": float(s), "score": _normalize_cosine_to_01(s)}
            for c, s in zip(cands, sims)
        ]
        best = {"text": cands[best_idx], "score_raw": best_raw, "score": best01}
    else:
        best, candidates = {"text": "", "score_raw": 0.0, "score": 0.0}, []

    return {
        "image_embedding_dim": len(img_emb),
        "ocr_alignment": None if ocr_score01 is None else {
            "text": ocr_text, "score_raw": ocr_score_raw, "score": ocr_score01
        },
        "best_caption": best,
        "candidates": sorted(candidates, key=lambda x: x["score_raw"], reverse=True)[:5],
    }

def _heuristic_creative_score(layout_balance, face_count, has_text, obj_count, align_best01, brand_hits, nsfw_safe):
    bal = float(layout_balance or 0.0)
    face = min(1.0, float(face_count or 0.0))
    text = 1.0 if has_text else 0.0
    objs = 1.0 if (obj_count or 0) > 0 else 0.0
    align = float(max(0.0, min(1.0, align_best01)))
    brands = 1.0 if (brand_hits or 0) > 0 else 0.0
    safe = 1.0 if nsfw_safe else 0.0
    score01 = (0.25 * bal + 0.15 * face + 0.10 * text + 0.10 * objs +
               0.20 * align + 0.10 * brands + 0.10 * safe)
    return int(round(score01 * 100))

# --- Brand detection (heuristic from OCR) ---
_BRANDS = {
    "apple": ["iphone", "ipad", "macbook", "ios"],
    "google": ["android", "pixel", "gmail", "youtube"],
    "youtube": ["subscribe", "yt"],
    "amazon": ["prime", "alexa"],
    "meta": ["facebook", "instagram", "whatsapp"],
    "facebook": ["fb"],
    "instagram": ["ig", "insta"],
    "whatsapp": ["wa"],
    "tiktok": ["tik tok"],
    "netflix": ["nflx"],
    "spotify": ["playlist"],
    "adidas": [],
    "nike": ["just do it", "air max"],
    "starbucks": ["coffee"],
    "mcdonalds": ["mcdonald's", "mcd", "mc donalds"],
    "uber": [],
    "lyft": [],
    "snapchat": ["snap"],
    "x": ["twitter"],
}
_WORD = re.compile(r"[A-Za-z0-9\-\']{2,}")

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return s.casefold()

def _detect_brands_from_text(ocr_text: str):
    text = _norm(ocr_text or "")
    tokens = set(m.group(0).casefold() for m in _WORD.finditer(text))
    hits = []
    for brand, synonyms in _BRANDS.items():
        brand_n = _norm(brand)
        found = False
        conf = 0.0
        evidence = None
        if brand_n in tokens:
            found, conf, evidence = True, 0.95, brand
        if not found and brand_n in text:
            found, conf, evidence = True, 0.8, brand
        if not found:
            for syn in synonyms:
                syn_n = _norm(syn)
                if syn_n in tokens:
                    found, conf, evidence = True, 0.85, syn
                    break
                if syn_n in text:
                    found, conf, evidence = True, 0.7, syn
                    break
        if found:
            hits.append({"brand": brand, "source": "ocr_text", "confidence": round(conf, 2), "evidence": evidence})
    return sorted(hits, key=lambda x: x["confidence"], reverse=True)[:8]

def _nsfw_scores(image_pil):
    if nsfw_pipe is None:
        return {"available": False, "scores": {}, "top_label": "unknown", "top_score": 0.0, "is_safe": True}
    try:
        preds = nsfw_pipe(image_pil)
        scores = {p["label"].lower(): float(p["score"]) for p in preds}
        top = max(scores.items(), key=lambda kv: kv[1]) if scores else ("unknown", 0.0)
        nsfw_prob = scores.get("nsfw", 0.0)
        sfw_prob = scores.get("sfw", 0.0)
        is_safe = nsfw_prob < 0.5 or (sfw_prob >= nsfw_prob)
        return {
            "available": True,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "top_label": top[0],
            "top_score": round(float(top[1]), 4),
            "is_safe": bool(is_safe),
        }
    except Exception:
        return {"available": False, "scores": {}, "top_label": "unknown", "top_score": 0.0, "is_safe": True}

# ---------- Video helpers ----------
def _clahe_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def _gamma(img_bgr, gamma=1.2):
    inv = 1.0 / max(gamma, 1e-6)
    table = ((np.arange(256) / 255.0) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img_bgr, table)

def _prep_face_crop(frame_bgr):
    img = _clahe_rgb(frame_bgr)
    img = _gamma(img, gamma=1.2)
    return img

def _detect_faces_bboxes(frame_bgr, conf_thresh=0.25):
    if face_model is None:
        return []
    try:
        r = face_model.predict(source=frame_bgr, verbose=False)[0]
        boxes = []
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if float(c) >= conf_thresh:
                x1, y1, x2, y2 = map(int, b.tolist())
                boxes.append([max(0, x1), max(0, y1), max(0, x2), max(0, y2)])
        return boxes
    except Exception:
        return []

def _fer_on_crop(bgr_crop):
    rgb = cv2.cvtColor(_prep_face_crop(bgr_crop), cv2.COLOR_BGR2RGB)
    res = fer_detector.detect_emotions(rgb) or []
    if not res:
        return None, 0.0
    emo = res[0].get("emotions", {})
    if not emo:
        return None, 0.0
    lab = max(emo, key=emo.get)
    return lab, float(emo[lab])

def _ema(prev, cur, alpha=0.4):
    return alpha * cur + (1.0 - alpha) * prev

def _iter_video_frames_every_sec(path, fps_target=1.0, max_secs=300):
    """Yield (t_sec, frame_bgr) – OpenCV first, then imageio fallback."""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps_src / max(fps_target, 1e-6))), 1)
        frame_idx, grabbed = 0, True
        while grabbed:
            grabbed = cap.grab()
            if not grabbed:
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                t_sec = frame_idx / max(fps_src, 1e-6)
                if t_sec > max_secs:
                    break
                yield t_sec, frame  # BGR
            frame_idx += 1
        cap.release()
        return

    # Fallback with imageio
    try:
        meta = iio.immeta(path)
        fps_src = float(meta.get("fps", 30.0))
    except Exception:
        fps_src = 30.0
    step_secs = 1.0 / max(fps_target, 1e-6)
    t_sec = 0.0
    for frame_rgb in iio.imiter(path):
        if t_sec > max_secs:
            break
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        yield t_sec, frame_bgr
        t_sec += step_secs

def analyze_video_emotions_every_sec(video_path, fps_used=1, max_secs=300):
    labels = ["happy", "angry", "sad", "fear", "surprise", "disgust", "neutral"]
    ema_scores = {l: 0.0 for l in labels}
    per_second, global_counts, global_scores = [], Counter(), defaultdict(float)
    total_faces_seen = 0
    analyzed = 0
    duration = 0.0

    for t_sec, frame in _iter_video_frames_every_sec(video_path, fps_target=fps_used, max_secs=max_secs):
        duration = max(duration, t_sec)

        bboxes = _detect_faces_bboxes(frame) or []
        per_face = []
        if bboxes:
            for (x1, y1, x2, y2) in bboxes[:4]:
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    lab, sc = _fer_on_crop(crop)
                    if lab:
                        per_face.append((lab, sc))
        else:
            lab, sc = _fer_on_crop(frame)
            if lab:
                per_face.append((lab, sc))

        face_count_this_sec = len(per_face)
        total_faces_seen += face_count_this_sec

        sec_scores = defaultdict(float)
        for lab, sc in per_face:
            sec_scores[lab] += sc
        if sec_scores:
            m = max(sec_scores.values())
            if m > 0:
                for k in sec_scores:
                    sec_scores[k] /= m

        if sec_scores:
            for l in labels:
                ema_scores[l] = _ema(ema_scores[l], sec_scores.get(l, 0.0), alpha=0.4)

        top_label = max(ema_scores, key=ema_scores.get) if ema_scores else "neutral"

        per_second.append({
            "t": round(t_sec, 2),
            "top": top_label,
            "face_count": face_count_this_sec,
            "scores": {l: round(ema_scores[l], 3) for l in labels if ema_scores[l] > 0},
        })

        global_counts[top_label] += 1
        for l in labels:
            global_scores[l] += ema_scores[l]

        analyzed += 1

    avg_scores = {l: round(global_scores[l] / max(analyzed, 1), 3) for l in labels}
    final_top = max(avg_scores, key=avg_scores.get) if avg_scores else "neutral"
    avg_faces_per_sec = round(float(total_faces_seen) / max(analyzed, 1), 2)

    return {
        "duration": round(duration, 2),
        "frames_analyzed": analyzed,
        "fps_used": fps_used,
        "per_second": per_second,
        "summary": {
            "final_top": final_top,
            "avg_scores": avg_scores,
            "counts": dict(global_counts),
            "avg_faces_per_sec": avg_faces_per_sec,
        },
    }

# ==========================================================
# Image analyze
# ==========================================================
@app.route("/analyze", methods=["POST"])
def analyze_ad():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    tmp_name = "input.jpg"
    file.save(tmp_name)

    try:
        image_array = preprocess_image(tmp_name)
        image_pil = Image.open(tmp_name).convert("RGB")

        emotion_label, emotion_conf, face_count = _emotion_with_conf_and_faces(image_array)
        if emotion_label == "No face detected":
            emotion_label = detect_dominant_emotion(image_array)

        colors = extract_color_palette(image_array)
        layout_balance = compute_layout_balance(image_array)
        heatmap_path = generate_saliency_heatmap(tmp_name, alpha=0.45)

        ocr_text = " ".join(ocr_reader.readtext(tmp_name, detail=0)).strip()

        detections = object_model(tmp_name)
        detected_labels = []
        for r in detections:
            cls_ids = getattr(r.boxes, "cls", None)
            if cls_ids is not None:
                for i in cls_ids.cpu().numpy().tolist():
                    detected_labels.append(object_model.names[int(i)])
        detected_labels = sorted(list(set(detected_labels)))[:10]

        img_vec = clip_model.encode(image_pil, convert_to_numpy=True)
        clip_embedding = img_vec.tolist()[:32]

        top_categories = clip_zero_shot_labels(clip_model, clip_model, image_pil, top_k=3)

        alignment = _clip_alignment(image_pil, ocr_text, detected_labels)
        align_best01 = float(alignment.get("best_caption", {}).get("score", 0.0))

        brand_hits = _detect_brands_from_text(ocr_text)
        nsfw = _nsfw_scores(image_pil)

        creative_score = _heuristic_creative_score(
            layout_balance=layout_balance,
            face_count=face_count,
            has_text=bool(ocr_text),
            obj_count=len(detected_labels),
            align_best01=align_best01,
            brand_hits=len(brand_hits),
            nsfw_safe=nsfw.get("is_safe", True),
        )

        # Gemini insight (JSON text)
        insight_text, insight_err = gemini_generate_insight({
            "media_type": "image",
            "dominant_emotion": emotion_label,
            "emotion_confidence": round(float(emotion_conf), 3),
            "face_count": int(face_count),
            "color_palette": colors,
            "layout_balance": layout_balance,
            "text_content": ocr_text,
            "detected_objects": detected_labels,
            "top_categories": top_categories,
            "alignment": alignment,
            "brands": brand_hits,
            "nsfw": nsfw,
            "creative_score": creative_score,
        }, media_type="image")

        result = {
            "media_type": "image",
            "dominant_emotion": emotion_label,
            "emotion_confidence": round(float(emotion_conf), 3),
            "face_count": int(face_count),
            "color_palette": colors,
            "layout_balance": layout_balance,
            "heatmap_url": heatmap_path,
            "text_content": ocr_text,
            "detected_objects": detected_labels,
            "clip_embedding": clip_embedding,
            "top_categories": top_categories,
            "alignment": alignment,
            "brands": brand_hits,
            "nsfw": nsfw,
            "creative_score": creative_score,
        }
        if insight_text:
            result["insight"] = json.loads(insight_text)
        if insight_err:
            result["insight_error"] = insight_err

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass

# ==========================================================
# Video analyze (per-second face-crop FER + aggregates) + Gemini
# ==========================================================
@app.route("/analyze_video", methods=["POST"])
def analyze_ad_video():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file.seek(0, os.SEEK_END)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > 500:
        return jsonify({"error": "Video too large (>500MB). Try a shorter clip."}), 400

    ext = os.path.splitext(file.filename or "")[-1] or ".mp4"
    tmp_name = f"input_video{ext}"
    file.save(tmp_name)

    try:
        # Emotions per second
        video_emotions = analyze_video_emotions_every_sec(tmp_name, fps_used=1, max_secs=300)
        duration_sec = video_emotions["duration"]
        frames_analyzed = video_emotions["frames_analyzed"]

        # Additional aggregates ~1fps
        cap = cv2.VideoCapture(tmp_name)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec_exact = frame_count / max(1e-6, native_fps)

        fps_sample = 1.0
        step = int(max(1, round(native_fps / max(0.1, fps_sample))))

        color_bag = []
        layout_vals = []
        all_ocr = []
        object_counter = Counter()
        clip_vecs = []

        nsfw_votes_safe = 0
        nsfw_checked = 0
        keyframe_heatmaps = []

        idx = 0
        analyzed_aux = 0
        max_frames = 120

        while True:
            ret = cap.grab()
            if not ret:
                break
            if idx % step != 0:
                idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tmp_img_path = f"__frame_{idx}.jpg"
            cv2.imwrite(tmp_img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            try:
                image_array = preprocess_image(tmp_img_path)
                pil_img = Image.fromarray(rgb)

                color_bag.extend(extract_color_palette(image_array, num_colors=5)[:3])
                layout_vals.append(compute_layout_balance(image_array))

                ocr_text = " ".join(ocr_reader.readtext(tmp_img_path, detail=0)).strip()
                if ocr_text:
                    all_ocr.append(ocr_text)

                dets = object_model(tmp_img_path)
                labels = []
                for r in dets:
                    cls_ids = getattr(r.boxes, "cls", None)
                    if cls_ids is not None:
                        for i in cls_ids.cpu().numpy().tolist():
                            labels.append(object_model.names[int(i)])
                for l in set(labels):
                    object_counter[l] += 1

                vec = clip_model.encode(pil_img, convert_to_numpy=True)
                clip_vecs.append(vec)

                if (analyzed_aux % 10) == 0:
                    hm = generate_saliency_heatmap(tmp_img_path, alpha=0.45)
                    keyframe_heatmaps.append(hm)

                if (analyzed_aux % 5) == 0:
                    ns = _nsfw_scores(pil_img)
                    if ns.get("is_safe", True):
                        nsfw_votes_safe += 1
                    nsfw_checked += 1

                analyzed_aux += 1
                if analyzed_aux >= max_frames:
                    pass
            finally:
                try:
                    if os.path.exists(tmp_img_path):
                        os.remove(tmp_img_path)
                except Exception:
                    pass

            if analyzed_aux >= max_frames:
                break
            idx += 1

        cap.release()

        duration_sec_out = round(float(duration_sec_exact), 2)
        palette_counter = Counter(color_bag)
        color_palette_global = [c for c, _ in palette_counter.most_common(5)]
        layout_balance_avg = round(float(np.mean(layout_vals)), 2) if layout_vals else None

        ocr_excerpt = ""
        if all_ocr:
            merged = " ".join(all_ocr)
            ocr_excerpt = merged[:600]

        objects_top = [{"label": k, "count": int(v)} for k, v in object_counter.most_common(8)]

        clip_embedding = None
        if clip_vecs:
            mat = np.stack(clip_vecs, axis=0)
            clip_embedding = mat.mean(axis=0).tolist()[:32]

        combined_brands = Counter()
        for txt in all_ocr:
            for hit in _detect_brands_from_text(txt):
                combined_brands[hit["brand"]] += 1
        brands = [{"brand": b, "source": "ocr_text", "confidence": 0.8, "evidence": b}
                  for b, _cnt in combined_brands.most_common(8)]

        nsfw_safe = True
        if nsfw_checked > 0:
            nsfw_safe = (nsfw_votes_safe / nsfw_checked) >= 0.5
        nsfw_summary = {
            "available": nsfw_pipe is not None,
            "frames_checked": int(nsfw_checked),
            "safe_votes": int(nsfw_votes_safe),
            "is_safe": bool(nsfw_safe),
        }

        # Map emotion summary to root keys for UI compatibility
        final_top = (video_emotions.get("summary") or {}).get("final_top")
        avg_faces_per_sec = (video_emotions.get("summary") or {}).get("avg_faces_per_sec")
        top_counts = (video_emotions.get("summary") or {}).get("counts") or {}
        top_emotions = [{"label": k, "count": int(v)} for k, v in sorted(top_counts.items(), key=lambda kv: kv[1], reverse=True)]

        # Gemini insight
        insight_text, insight_err = gemini_generate_insight({
            "media_type": "video",
            **{
                "video_emotions": video_emotions,
                "layout_balance_avg": layout_balance_avg,
                "objects_top": objects_top,
                "ocr_excerpt": ocr_excerpt,
                "brands": brands,
                "nsfw": nsfw_summary,
                "creative_score": None,  # will fill after computing
            },
        }, media_type="video")

        creative_score = _heuristic_creative_score(
            layout_balance=layout_balance_avg or 0.0,
            face_count=avg_faces_per_sec,
            has_text=bool(ocr_excerpt),
            obj_count=sum(v["count"] for v in objects_top),
            align_best01=0.0,
            brand_hits=len(brands),
            nsfw_safe=nsfw_safe,
        )

        # insert score into insight if JSON
        if insight_text:
            try:
                tmp = json.loads(insight_text)
                tmp.setdefault("creative_score", creative_score)
                insight_text = json.dumps(tmp)
            except Exception:
                pass

        result = {
            "media_type": "video",
            "duration_sec": duration_sec_out,
            "frames_analyzed": int(frames_analyzed),
            "fps_used": float(video_emotions["fps_used"]),
            "video_emotions": video_emotions,
            # root-level convenience fields
            "dominant_emotion": final_top,
            "avg_faces_per_frame": avg_faces_per_sec,
            "top_emotions": top_emotions,
            # other aggregates:
            "objects_top": objects_top,
            "ocr_excerpt": ocr_excerpt,
            "color_palette_global": color_palette_global,
            "layout_balance_avg": layout_balance_avg,
            "keyframe_heatmaps": keyframe_heatmaps[:6],
            "clip_embedding": clip_embedding,
            "brands": brands,
            "nsfw": nsfw_summary,
            "alignment": {
                "frame_index": None,
                "best_caption": {"text": "A person reacting emotionally in an advertisement", "score": 0.0},
                "alignment_detail": {},
            },
            "creative_score": creative_score,
        }
        if insight_text:
            result["insight"] = json.loads(insight_text)
        if insight_err:
            result["insight_error"] = insight_err

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass

# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
