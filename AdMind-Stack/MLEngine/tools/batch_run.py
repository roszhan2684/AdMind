#!/usr/bin/env python3
# batch_run.py — run AdMind ML (via ML engine or Backend) over a ZIP/folder of ads and export CSV/JSONL
import argparse, io, json, os, sys, shutil, tempfile, time, zipfile, mimetypes, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# ------------------------------ config ------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".heic"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

def looks_like_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def looks_like_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def pick_engine_endpoint(ml_base: str, p: Path) -> str:
    return f"{ml_base}/analyze_video" if looks_like_video(p) else f"{ml_base}/analyze"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def skip_macos_junk(name: str) -> bool:
    # Skip __MACOSX/ folder and resource forks like ._image.png
    base = name.strip()
    if not base:
        return True
    parts = base.split("/")
    if any(part == "__MACOSX" for part in parts):
        return True
    leaf = parts[-1]
    if leaf.startswith("._"):  # resource fork
        return True
    if leaf == ".DS_Store":
        return True
    return False

# ------------------------- HTTP session w/ retry -------------------------
def make_session():
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = make_session()

# ---------------------------- insight parsing ----------------------------
def _parse_fenced_json(s: str):
    if not isinstance(s, str):
        return None
    txt = s.strip()
    if txt.startswith("```"):
        # remove opening ```
        txt = txt[3:].strip()
        # strip language tag if present (json or anything)
        nl = txt.find("\n")
        if nl != -1 and nl < 12:
            txt = txt[nl+1:].strip()
        # strip trailing ```
        if txt.endswith("```"):
            txt = txt[:-3].strip()
    try:
        return json.loads(txt)
    except Exception:
        return None

def _extract_insight_columns(insight_val):
    """
    Accepts either:
      - string (raw JSON or fenced JSON), or
      - dict with fields, or
      - anything else -> put into 'insight_raw'
    Returns dict of flat columns.
    """
    cols = {
        "insight_emotion": "",
        "insight_summary": "",
        "insight_weakness": "",
        "insight_suggestion_1": "",
        "insight_suggestion_2": "",
        "insight_suggestion_3": "",
        "insight_raw": "",
    }

    if insight_val is None:
        return cols

    if isinstance(insight_val, str):
        parsed = _parse_fenced_json(insight_val)
        if parsed is None:
            cols["insight_raw"] = insight_val
            return cols
        insight = parsed
    elif isinstance(insight_val, dict):
        insight = insight_val
    else:
        cols["insight_raw"] = json.dumps(insight_val, ensure_ascii=False)
        return cols

    cols["insight_emotion"] = str(insight.get("emotion", "") or "")
    cols["insight_summary"] = str(insight.get("insight_summary", "") or "")
    cols["insight_weakness"] = str(insight.get("weakness", "") or "")

    suggestions = insight.get("suggestions") or []
    if isinstance(suggestions, list):
        if len(suggestions) > 0: cols["insight_suggestion_1"] = str(suggestions[0])
        if len(suggestions) > 1: cols["insight_suggestion_2"] = str(suggestions[1])
        if len(suggestions) > 2: cols["insight_suggestion_3"] = str(suggestions[2])

    try:
        cols["insight_raw"] = json.dumps(insight, ensure_ascii=False)
    except Exception:
        cols["insight_raw"] = str(insight)
    return cols

# ---------------------------- flatteners ----------------------------
def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def flatten_image_result(js: dict) -> dict:
    brands = js.get("brands") or []
    objects = js.get("detected_objects") or []
    top_cats = js.get("top_categories") or []
    if top_cats and isinstance(top_cats[0], dict):
        top_cats = [f'{x.get("label", x.get("name",""))}:{x.get("score",0):.2f}' for x in top_cats]

    base = {
        "media_type": js.get("media_type", "image"),
        "dominant_emotion": js.get("dominant_emotion"),
        "emotion_confidence": js.get("emotion_confidence"),
        "face_count": js.get("face_count"),
        "layout_balance": js.get("layout_balance"),
        "creative_score": js.get("creative_score"),
        "nsfw_safe": _safe_get(js, "nsfw", "is_safe", default=None),
        "brands": ", ".join(sorted(set(b.get("brand","") for b in brands if b.get("brand")))) or None,
        "objects_top3": ", ".join(objects[:3]) if objects else None,
        "top_categories": ", ".join(top_cats[:3]) if top_cats else None,
        "ocr_text_len": len((js.get("text_content") or "").strip()),
        "heatmap_url": js.get("heatmap_url"),
    }
    base.update(_extract_insight_columns(js.get("insight")))
    return base

def flatten_video_result(js: dict) -> dict:
    brands = js.get("brands") or []
    objects_top = js.get("objects_top") or []
    objects_fmt = [f'{o.get("label","")}:{o.get("count",0)}' for o in objects_top][:5]

    ve = js.get("video_emotions") or {}
    final_top = _safe_get(ve, "summary", "final_top")
    avg_faces_per_sec = _safe_get(ve, "summary", "avg_faces_per_sec")

    base = {
        "media_type": js.get("media_type", "video"),
        "duration_sec": js.get("duration_sec"),
        "frames_analyzed": js.get("frames_analyzed"),
        "fps_used": js.get("fps_used"),
        "final_emotion": final_top,
        "avg_faces_per_sec": avg_faces_per_sec,
        "layout_balance": js.get("layout_balance_avg"),
        "creative_score": js.get("creative_score"),
        "nsfw_safe": _safe_get(js, "nsfw", "is_safe", default=None),
        "brands": ", ".join(sorted(set(b.get("brand","") for b in brands if b.get("brand")))) or None,
        "objects_top": ", ".join(objects_fmt) if objects_fmt else None,
        "ocr_text_len": len((js.get("ocr_excerpt") or "").strip()),
        "keyframe_heatmaps": ", ".join(js.get("keyframe_heatmaps") or []) or None,
    }
    base.update(_extract_insight_columns(js.get("insight")))
    return base

def flatten_result(js: dict) -> dict:
    media_type = js.get("media_type")
    if media_type == "video":
        return flatten_video_result(js)
    return flatten_image_result(js)

# ------------------------- callers (ML vs Backend) -------------------------
def call_ml(ml_base: str, file_path: Path, timeout: int = 600) -> dict:
    """Call Python ML engine directly."""
    url = pick_engine_endpoint(ml_base, file_path)
    mime = mimetypes.guess_type(str(file_path))[0] or ("video/mp4" if looks_like_video(file_path) else "image/jpeg")
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, mime)}
        resp = SESSION.post(url, files=files, timeout=timeout)
    try:
        js = resp.json()
    except Exception:
        js = {"raw": resp.text}
    js["_http_status"] = resp.status_code
    return js

def call_backend(backend_base: str, file_path: Path, timeout: int = 600) -> dict:
    """Call Node backend /api/upload (includes Gemini insight)."""
    url = f"{backend_base.rstrip('/')}/upload"
    mime = mimetypes.guess_type(str(file_path))[0] or ("video/mp4" if looks_like_video(file_path) else "image/jpeg")
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, mime)}
        resp = SESSION.post(url, files=files, timeout=timeout)
    try:
        js = resp.json()
    except Exception:
        js = {"raw": resp.text}
    js["_http_status"] = resp.status_code
    return js

def process_one(use_backend: bool, ml_base: str, backend_base: str, file_path: Path, timeout: int = 600) -> dict:
    t0 = time.time()
    try:
        js = call_backend(backend_base, file_path, timeout=timeout) if use_backend else call_ml(ml_base, file_path, timeout=timeout)
        latency = time.time() - t0
        flat = flatten_result(js) if isinstance(js, dict) else {}
        return {
            "id": file_path.name,
            "rel_path": str(file_path),
            "status": js.get("_http_status", None),
            "latency_s": round(latency, 3),
            **flat,
            "_raw": js,
            "error": (js.get("error") if isinstance(js, dict) else None),
        }
    except Exception as e:
        return {
            "id": file_path.name,
            "rel_path": str(file_path),
            "status": None,
            "latency_s": None,
            "error": str(e),
            "_raw": {"error": str(e)},
        }

# ------------------------------- IO --------------------------------
def iter_media_files(src_path: Path, tmp_dir: Path, only: str | None):
    """
    Yield Paths of extracted files from a ZIP or files from a directory.
    Skips macOS junk: __MACOSX/, ._resource forks, .DS_Store, and zero-byte files.
    """
    def want(p: Path) -> bool:
        if p.stat().st_size == 0:
            return False
        if looks_like_image(p) or looks_like_video(p):
            if only == "images":
                return looks_like_image(p)
            if only == "videos":
                return looks_like_video(p)
            return True
        return False

    if src_path.is_dir():
        for p in sorted(src_path.rglob("*")):
            if not p.is_file():
                continue
            if skip_macos_junk(str(p)):
                continue
            if want(p):
                yield p
        return

    # ZIP
    with zipfile.ZipFile(src_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for m in members:
            if skip_macos_junk(m.filename):
                continue
            sfx = Path(m.filename).suffix.lower()
            if sfx not in IMAGE_EXTS and sfx not in VIDEO_EXTS:
                continue
            out = tmp_dir / Path(m.filename).name
            with zf.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if out.stat().st_size == 0:
                continue
            if want(out):
                yield out

def write_jsonl(rows, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            j = r.get("_raw", {})
            j.setdefault("id", r.get("id"))
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

def write_csv(rows, out_path: Path):
    # Collect all keys that appear (for a stable, wide CSV)
    keys = set()
    for r in rows:
        keys.update(k for k in r.keys() if k != "_raw")
    # Put error up front so failures are obvious
    front = [
        "id","rel_path","status","latency_s","error",
        "media_type","dominant_emotion","final_emotion","emotion_confidence",
        "face_count","avg_faces_per_sec","creative_score","nsfw_safe","brands",
        "objects_top3","objects_top","layout_balance","duration_sec","frames_analyzed",
        "fps_used","ocr_text_len","heatmap_url","keyframe_heatmaps",
        "insight_emotion","insight_summary","insight_weakness",
        "insight_suggestion_1","insight_suggestion_2","insight_suggestion_3","insight_raw",
    ]
    ordered = front + [k for k in sorted(keys) if k not in front]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            r2 = {k: ("" if (v is None) else v) for k, v in r.items() if k != "_raw"}
            w.writerow(r2)

# ------------------------------ CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run AdMind (ML or Backend) over a ZIP/folder and export CSV/JSONL")
    ap.add_argument("--ml", default="http://127.0.0.1:5001", help="ML engine base URL")
    ap.add_argument("--backend", default="http://127.0.0.1:5050/api", help="Backend base URL (will call /upload)")
    ap.add_argument("--use-backend", action="store_true", help="Use backend (/api/upload) so CSV includes Gemini insight")
    ap.add_argument("--zip", required=True, help="Path to ads.zip OR a folder of media")
    ap.add_argument("--out", required=True, help="Output folder for CSV/JSONL")
    ap.add_argument("--only", choices=["images","videos"], help="Process only images or only videos")
    ap.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    ap.add_argument("--timeout", type=int, default=600, help="Per-file timeout (seconds)")
    args = ap.parse_args()

    use_backend = bool(args.use_backend)
    ml_base = args.ml.rstrip("/")
    backend_base = args.backend.rstrip("/")
    src = Path(args.zip).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)

    tmp_dir = Path(tempfile.mkdtemp(prefix="admind_zip_"))
    rows = []
    try:
        files = list(iter_media_files(src, tmp_dir, only=args.only))
        if not files:
            print("No media files found in the given path.", file=sys.stderr)
            return 2

        n_img = sum(1 for p in files if looks_like_image(p))
        n_vid = sum(1 for p in files if looks_like_video(p))
        print(f"Queued files: total={len(files)}  images={n_img}  videos={n_vid}  (using_backend={use_backend})")

        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futs = {
                ex.submit(process_one, use_backend, ml_base, backend_base, p, args.timeout): p
                for p in files
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing"):
                rows.append(fut.result())

        # Outputs
        csv_path = out_dir / "features.csv"
        jsonl_path = out_dir / "features.jsonl"
        timing_path = out_dir / "timing.json"

        write_csv(rows, csv_path)
        write_jsonl(rows, jsonl_path)

        ok = [r for r in rows if r.get("status") == 200]
        fail = [r for r in rows if r.get("status") != 200]
        ok_img = sum(1 for r in ok if r.get("media_type") == "image")
        ok_vid = sum(1 for r in ok if r.get("media_type") == "video")
        summary = {
            "total": len(rows),
            "ok": len(ok),
            "ok_images": ok_img,
            "ok_videos": ok_vid,
            "fail": len(fail),
            "avg_latency_s": round(sum((r.get("latency_s") or 0) for r in rows) / max(1, len(rows)), 3),
            "used_backend": use_backend,
        }
        with open(timing_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nDone.\n CSV: {csv_path}\n JSONL: {jsonl_path}\n Timing: {timing_path}")
        print(f"Summary: {summary}")
        return 0
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    raise SystemExit(main())
