#!/usr/bin/env python3
import json, csv, os, sys
from pathlib import Path

ROOT = Path("/Users/roszhanraj/AdMind/AdMindStack/outputs")
CSV_IN  = ROOT / "features.csv"
JSONL_IN = ROOT / "features.jsonl"
CSV_OUT = ROOT / "features_wide.csv"

# Order emotion columns you want to materialize for videos
EMOS = ["happy","angry","sad","fear","surprise","disgust","neutral"]

# Load base CSV rows
rows = []
with open(CSV_IN, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Load raw JSONL (id -> raw)
raw_by_id = {}
with open(JSONL_IN, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        js = json.loads(line)
        _id = js.get("id")
        if _id: raw_by_id[_id] = js

# Extend rows with video emotion averages
for r in rows:
    if r.get("media_type") != "video":  # fill blanks for images
        for e in EMOS:
            r[f"avg_{e}"] = ""
        r["video_final_emotion"] = r.get("final_emotion","")
        continue

    rid = r.get("id")
    raw = raw_by_id.get(rid, {})
    avg_scores = (((raw.get("video_emotions") or {}).get("summary") or {})
                  .get("avg_scores") or {})
    # Fill columns
    for e in EMOS:
        r[f"avg_{e}"] = avg_scores.get(e, "")
    r["video_final_emotion"] = ((raw.get("video_emotions") or {})
                                .get("summary") or {}).get("final_top","")

# Write out the widened CSV
fieldnames = list(rows[0].keys())
# ensure our new columns are near the end (but in a consistent order)
for col in ["video_final_emotion", *[f"avg_{e}" for e in EMOS]]:
    if col not in fieldnames:
        fieldnames.append(col)

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"✅ Wrote {CSV_OUT}")
