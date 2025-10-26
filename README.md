# AdMind — Creative Intelligence for Ads

Cal Hacks 12.0 Submission • AppLovin Ad Intelligence Challenge

AdMind is an end-to-end system that analyzes ad creatives (images and videos) and turns pixels, motion, and text into structured features and a concise AI insight that can feed a recommendation engine. It combines a Python ML engine for fast, parallelizable signal extraction with a Node/Express gateway and a lightweight React UI.

---

## Why AdMind

Modern advertisers need to know *why* a creative will work before spending on distribution. AdMind extracts distinct, minimally overlapping signals—for tone, composition, content, motion, and copy—then synthesizes them into an actionable summary using Gemini. These signals are designed to be joined with campaign and user context to improve a ranking model like AppLovin’s Axon.

---

## Core Signals (Images & Video)

* **Emotion / Tone**

  * Images: FER-based dominant emotion (+ confidence, face count).
  * Video: per-second emotion timeline, smoothed with EMA; final dominant emotion; average faces/second.

* **Composition**

  * Layout balance, negative space heuristics, saliency heatmaps, global color palette.

* **Objects & Semantics**

  * YOLO object detections; CLIP zero-shot categories; best caption alignment score.

* **Copy & Brand Cues**

  * OCR text excerpt; brand name heuristics from text (with confidence/evidence).

* **NSFW & Safety**

  * Optional classifier for safe/unsafe summaries (graceful degradation if unavailable).

* **Video-specific**

  * Duration, frames analyzed, fps sample, keyframe heatmaps, top objects and OCR across time.

* **Creative Score**

  * Lightweight, interpretable score combining composition, text, object presence, brand cues, and safety.

* **Gemini Insight (Strict JSON)**

  * A 1–2 sentence summary, one weakness, and 3–5 suggestions; always present via robust fallbacks.

---

## System Architecture

```
[React Frontend]  ──►  [Node/Express Gateway :5050]
                          └─ uploads/, serves /outputs, calls Gemini SDK
                                 │
                                 ▼
                      [Python Flask ML Engine :5001]
                      OCR • YOLO • CLIP • FER • NSFW • Heatmaps
                                 │
                                 ▼
                         features + insight JSON
```

* The ML engine does all heavy lifting and returns structured features.
* The Node backend enriches results with a Gemini insight (using `@google/genai`) and serves static outputs (heatmaps).
* The UI uploads a single file and renders everything, or you can batch a whole ZIP via a CLI tool.

---

## Quick Start

### 1) Clone

```bash
git clone <your-repo-url>
cd AdMindStack
```

### 2) Environment

Create `backend/.env`:

```env
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
ML_BASE=http://127.0.0.1:5001
MAX_UPLOAD_MB=200
```

### 3) Start the ML Engine (Python, port 5001)

From `AdMindStack/MLEngine/`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # see “Python requirements” below
python main.py
```

### 4) Start the Gateway (Node, port 5050)

From `AdMindStack/backend/`:

```bash
npm install
node server.js
```

### 5) Start the Frontend

If you used the provided React app:

```bash
cd AdMindStack/frontend
npm install
npm start   # or: npm run dev
```

Open the app and upload an image/video. Heatmaps are served from `/outputs`, and Gemini Insight appears under the analysis block.

---

## Batch Processing (ZIP or Folder)

From `AdMindStack/tools/`:

```bash
python3 batch_run.py \
  --use-backend \
  --backend http://127.0.0.1:5050/api \
  --zip /path/to/ads.zip \
  --out /path/to/outputs \
  --max-workers 4 \
  --timeout 600
```

Outputs:

* `features.csv` — wide table for quick inspection.
* `features.jsonl` — one JSON per creative (full raw payloads).
* `timing.json` — summary stats.

---

## API Endpoints

### POST `/api/upload`

Form-data: `file=<image|video>`

* Routes to `ML_BASE/analyze` (images) or `ML_BASE/analyze_video` (videos).
* Enriches response with `insight` (Gemini strict JSON).
* Returns JSON including: composition, objects, OCR, brands, NSFW, creative_score, heatmap paths, and `insight`.

### GET `/api/gemini/health`

Pings Gemini with a lightweight prompt and returns `{ ok, model, reply | error }`.

### ML Engine

* `GET /health` — engine status and flags.
* `POST /analyze` — image analysis.
* `POST /analyze_video` — video analysis.

---

## Frontend Notes

* The UI displays:

  * Dominant emotion (and confidence), faces, layout balance, palette.
  * OCR text, detected objects, top categories, brands, NSFW.
  * Best caption and score; creative score; heatmap preview.
  * For videos: duration, frames analyzed, fps, per-second emotions, top objects, OCR excerpt, keyframe heatmaps.
  * **Gemini Insight** for both image and video via an `InsightBlock` that parses strict JSON or fenced JSON codeblocks.

---

## File/Folder Structure (high-level)

```
AdMindStack/
├─ MLEngine/
│  ├─ main.py
│  ├─ utils/
│  │  ├─ heatmap.py
│  │  ├─ image_preprocessing.py
│  │  ├─ feature_extraction.py
│  │  └─ zero_shot.py
│  └─ requirements.txt
├─ backend/
│  ├─ server.js
│  ├─ routes/
│  │  └─ analyze.js
│  └─ .env
├─ frontend/
│  └─ src/App.js
└─ tools/
   └─ batch_run.py
```

---

## Tech Stack

**ML Engine (Python)**

* Flask, Flask-CORS
* OpenCV, NumPy, PIL/Pillow
* EasyOCR
* Ultralytics YOLO
* Sentence-Transformers (CLIP)
* FER (facial emotion)
* Transformers (optional NSFW)
* Requests, python-dotenv

**Gateway (Node)**

* Express, Multer
* Axios, Form-Data
* `@google/genai` (Gemini SDK)
* dotenv

**Frontend**

* React
* Axios
* Minimal CSS

---

## Why These Signals Help a Recommender

* **Emotion trajectory (video)**: correlates with early attention and retention; smoother or rising valence often performs better.
* **Layout balance & saliency**: predict glanceability and CTA legibility on small screens.
* **OCR and brand cues**: capture clarity of message and brand reinforcement without training a custom OCR stack.
* **Object categories & zero-shot labels**: enable creative clustering, retrieval, and negative keywords for exploration/exploitation.
* **NSFW safety**: essential gating signal for inventory suitability.
* **Creative Score**: a transparent, tunable baseline to prioritize variants when labels are unavailable.

---

## Performance & Robustness

* Designed to run under five minutes for the provided dataset via:

  * Lightweight models (YOLO-n, CLIP B/32).
  * Video sampling at ~1 fps for auxiliary analysis.
  * Thread-pool batch runner with retries.
* All calls have sensible timeouts and degrade gracefully:

  * If Gemini is unavailable, a fallback heuristic always returns a valid “insight” JSON.
  * If NSFW is unavailable, the system defaults to safe and continues.

---

## Troubleshooting

**“connect ECONNREFUSED 127.0.0.1:5001”**
The ML engine isn’t running or is on a different port. Start `python main.py` and confirm:

```bash
curl http://127.0.0.1:5001/health
```

**“Parse Error: Expected HTTP/” when calling ML**
Something else is bound to port 5001 (non-HTTP service). Find and stop it:

```bash
lsof -i :5001
kill -9 <PID>
```

Then restart the ML engine.

**Hugging Face timeouts on first run**
Model weights download on first launch. Ensure stable internet, or pre-cache models. You can also retry; the hub has built-in exponential backoff.

**OpenCV “Could not open video”**
The file may be corrupted or an unsupported codec. Re-encode with ffmpeg:

```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -movflags +faststart fixed.mp4
```

**Gemini insight missing**
Verify `GEMINI_API_KEY` and internet access:

```bash
curl http://127.0.0.1:5050/api/gemini/health
```

If the SDK complains about safety categories, the backend already uses safe enums; upgrade `@google/genai` and keep the provided `analyze.js`.

---

## Security & Privacy

* API keys are read from environment variables and never committed.
* Uploaded files are stored temporarily and deleted after processing on the backend; heatmaps are served from `/outputs` only.
* No user data beyond the upload is retained by default.

---

## What’s Next

* Train a lightweight meta-model to predict CTR/retention using these features.
* Add audio sentiment and speech-to-text alignment.
* Platform-specific recommendations (TikTok vs. Instagram vs. YouTube).
* Creative search and clustering for creative strategy and coverage analysis.
* Simple API for bulk ingestion and CI/CD integration.

---

## License

MIT. See `LICENSE`.

---

## Acknowledgements

* Open-source models: Ultralytics, Sentence-Transformers, EasyOCR, FER.
* Google Gemini for generative insights and summarization.
* Thanks to the Cal Hacks and AppLovin teams for the challenge and resources.
