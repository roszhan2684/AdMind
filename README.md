AdMind — Ad Creative Intelligence Pipeline

AdMind is an end-to-end system that analyzes ad creatives (images and videos) and extracts high-value, minimally overlapping signals for ranking and recommendation systems. It is designed for speed, robustness, and batchability, with optional LLM insights to turn raw features into concise creative guidance.

Highlights

Multimodal feature extraction for images and videos:

Dominant emotion, face stats, zero-shot object categories

OCR text and brand heuristics

Layout balance and saliency heatmaps

Color palettes

Video-specific: per-second emotion timeline, keyframes, motion-robust face crops

Embeddings: CLIP-style visual alignment and caption alignment score

NSFW screening (optional)

LLM Insight (Gemini): Structured JSON insight summarizing tone, weakness, and 3 concrete suggestions. Includes a deterministic heuristic fallback so an insight is always present.

Three components:

ML Engine (Python/Flask) — model inference for images and video.

Backend API (Node/Express) — file upload, orchestration, Gemini enrichment.

Web UI (React) — upload, preview, rich result viewer.

Batch runner to process a ZIP or folder and export features.csv and features.jsonl.

Parallelizable and fast: designed to run under five minutes on the provided dataset with 4 workers on a modern laptop.

Repository Layout
AdMind/
├─ AdMindStack/
│  ├─ MLEngine/                # Python models + Flask server (port 5001)
│  │  ├─ main.py
│  │  ├─ utils/                # heatmap, preprocessing, feature extraction, zero-shot
│  │  ├─ venv/                 # local virtualenv (excluded from git)
│  ├─ backend/                 # Node/Express API (port 5050)
│  │  ├─ server.js
│  │  └─ routes/analyze.js
│  ├─ frontend/                # React app (port 3000)
│  │  ├─ src/App.js
│  │  └─ App.css
│  ├─ tools/                   # CLI + batch processing scripts
│  │  └─ batch_run.py
│  ├─ dataset/                 # ads.zip and sample media (excluded or small subset)
│  └─ outputs/                 # generated features (CSV/JSONL/heatmaps)
└─ README.md

Setup
0) System Requirements

macOS or Linux

Python 3.10–3.11

Node.js 18+ and npm

Optional: FFmpeg for robust video codecs

1) Python ML Engine
cd AdMind/AdMindStack/MLEngine

# Use a fresh virtualenv
python3 -m venv venv
source venv/bin/activate

# Install runtime deps
pip install --upgrade pip wheel
pip install -r requirements.txt  # if present
# Or, minimally:
pip install flask flask-cors easyocr ultralytics sentence-transformers fer pillow requests python-dotenv

# First run will auto-download YOLO weights, etc.
python main.py  # serves on http://127.0.0.1:5001


Environment variables (optional) for ML Engine:

# .env (ML Engine)
GEMINI_API_KEY=<your-key>         # optional; ML can run without it
GEMINI_MODEL=gemini-1.5-flash     # or gemini-2.5-flash if supported by your key

2) Backend API
cd AdMind/AdMindStack/backend
npm install

# .env for backend
cat > .env <<'EOF'
GEMINI_API_KEY=<your-key>         # optional; if omitted a heuristic insight is used
GEMINI_MODEL=gemini-2.5-flash     # prefer stable models; project falls back automatically
ML_BASE=http://127.0.0.1:5001
MAX_UPLOAD_MB=200
EOF

node server.js    # http://127.0.0.1:5050


Health checks:

curl http://127.0.0.1:5050/ping
curl http://127.0.0.1:5050/api/limits
curl http://127.0.0.1:5050/api/gemini/health

3) Frontend UI
cd AdMind/AdMindStack/frontend
npm install
npm start        # http://localhost:3000


The UI lets you upload an image or video and renders the full analysis with heatmaps and Gemini insight.

Batch Processing

Process an entire ZIP or folder and export CSV/JSONL:

cd AdMind/AdMindStack/MLEngine/tools
source ../venv/bin/activate

# Use Python ML directly:
python batch_run.py \
  --ml http://127.0.0.1:5001 \
  --zip /path/to/ads.zip \
  --out /path/to/outputs \
  --max-workers 4 \
  --timeout 600

# Or route through the backend (adds Gemini insight):
python batch_run.py \
  --use-backend \
  --backend http://127.0.0.1:5050/api \
  --zip /path/to/ads.zip \
  --out /path/to/outputs \
  --max-workers 4 \
  --timeout 600


Artifacts:

outputs/features.csv — flat, analysis-ready rows per creative

outputs/features.jsonl — full raw JSON per creative

outputs/timing.json — simple throughput summary

Extracted Signals

Core image features:

dominant_emotion, emotion_confidence, face_count

text_content (OCR), ocr_text_len

detected_objects (YOLO)

top_categories (zero-shot CLIP labels)

layout_balance (symmetry/space heuristic)

color_palette (dominant colors)

alignment.best_caption.score (CLIP image–text alignment)

brands (OCR heuristic)

nsfw summary

creative_score (interpretable composite score)

heatmap_url (saliency visualization)

Core video features:

video_emotions.per_second and video_emotions.summary.final_top

avg_faces_per_sec, top_emotions counts

objects_top, ocr_excerpt, color_palette_global

layout_balance_avg, keyframe_heatmaps

nsfw temporal sampling

creative_score (video version)

LLM Insight:

Always present via Gemini or heuristic fallback.

Flat JSON fields:

insight.emotion

insight.insight_summary

insight.weakness (when using backend’s current controller)

insight.suggestions (array)
