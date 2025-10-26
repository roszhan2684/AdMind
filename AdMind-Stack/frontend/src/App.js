// src/App.js
import React, { useState } from "react";
import axios from "axios";
import "./App.css";

/** ---------- helpers for Gemini fenced JSON ---------- */
function parseFencedJson(s) {
  if (typeof s !== "string") return null;
  const cleaned = s.replace(/^```json\s*/i, "").replace(/```$/i, "").trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    return null;
  }
}

function InsightBlock({ insight }) {
  // Handles: string with ```json fences, object with { insight_summary }, plain object/string
  let parsed = null;

  if (typeof insight === "string") {
    parsed = parseFencedJson(insight) ?? null;
  } else if (insight && typeof insight === "object") {
    if (typeof insight.insight_summary === "string") {
      parsed =
        parseFencedJson(insight.insight_summary) ?? {
          insight_summary: insight.insight_summary,
        };
    } else {
      parsed = insight;
    }
  }

  if (!parsed) {
    return (
      <pre>{typeof insight === "string" ? insight : JSON.stringify(insight, null, 2)}</pre>
    );
  }

  return (
    <div>
      {"emotion" in parsed && (
        <p>
          <strong>Emotion:</strong> {String(parsed.emotion)}
        </p>
      )}
      {"insight_summary" in parsed && (
        <p>
          <strong>Summary:</strong> {String(parsed.insight_summary)}
        </p>
      )}
      {"weakness" in parsed && (
        <p>
          <strong>Weakness:</strong> {String(parsed.weakness)}
        </p>
      )}
      {"suggestions" in parsed && Array.isArray(parsed.suggestions) && (
        <>
          <h4>Suggestions</h4>
          <ul>
            {parsed.suggestions.map((s, i) => (
              <li key={i}>{String(s)}</li>
            ))}
          </ul>
        </>
      )}
      <details style={{ marginTop: 8 }}>
        <summary>Raw insight</summary>
        <pre>{JSON.stringify(parsed, null, 2)}</pre>
      </details>
    </div>
  );
}

/** ------------------------------ Image result ------------------------------ */
function ImageResult({ result }) {
  return (
    <>
      {"dominant_emotion" in result && (
        <p>
          <strong>Dominant Emotion:</strong> {String(result.dominant_emotion)}
          {typeof result.emotion_confidence === "number"
            ? ` (${result.emotion_confidence})`
            : ""}
        </p>
      )}

      {"face_count" in result && (
        <p>
          <strong>Faces:</strong> {String(result.face_count)}
        </p>
      )}

      {"layout_balance" in result && (
        <p>
          <strong>Layout Balance:</strong> {String(result.layout_balance)}
        </p>
      )}

      <p>
        <strong>Color Palette:</strong>
      </p>
      <div className="colors">
        {(result.color_palette || []).map((color, i) => (
          <div key={i} className="color-box" style={{ background: color }} />
        ))}
      </div>

      {"text_content" in result && (
        <p style={{ marginTop: 12 }}>
          <strong>OCR Text:</strong>{" "}
          {String(result.text_content || "").trim() || "(none)"}
        </p>
      )}

      {"detected_objects" in result && (
        <>
          <p style={{ marginTop: 12 }}>
            <strong>Detected Objects:</strong>
          </p>
          {(result.detected_objects || []).length ? (
            <ul>
              {result.detected_objects.map((o, i) => (
                <li key={i}>{String(o)}</li>
              ))}
            </ul>
          ) : (
            <p>(none)</p>
          )}
        </>
      )}

      {"top_categories" in result && (
        <>
          <p style={{ marginTop: 12 }}>
            <strong>Top Categories:</strong>
          </p>
          {(result.top_categories || []).length ? (
            <ol>
              {result.top_categories.map((c, i) => (
                <li key={i}>
                  {typeof c === "string" ? (
                    c
                  ) : (
                    <>
                      {String(c.label ?? c.name ?? "label")}
                      {"score" in c ? ` — ${Number(c.score).toFixed(2)}` : ""}
                    </>
                  )}
                </li>
              ))}
            </ol>
          ) : (
            <p>(none)</p>
          )}
        </>
      )}

      {"brands" in result && (
        <>
          <p style={{ marginTop: 12 }}>
            <strong>Brands (OCR heuristic):</strong>
          </p>
          {(result.brands || []).length ? (
            <ul>
              {result.brands.map((b, i) => (
                <li key={i}>
                  {b.brand} — conf {b.confidence} (src: {b.source})
                </li>
              ))}
            </ul>
          ) : (
            <p>(none)</p>
          )}
        </>
      )}

      {"nsfw" in result && (
        <p style={{ marginTop: 12 }}>
          <strong>NSFW safe:</strong>{" "}
          {String(result.nsfw?.is_safe ?? true)}{" "}
          {result.nsfw?.available === false ? "(model unavailable)" : ""}
        </p>
      )}

      {"alignment" in result && result.alignment?.best_caption && (
        <p>
          <strong>Best Caption:</strong> “{result.alignment.best_caption.text}” —{" "}
          {Number(result.alignment.best_caption.score ?? 0).toFixed(2)}
        </p>
      )}

      {"creative_score" in result && (
        <p>
          <strong>Creative Score:</strong> {String(result.creative_score)} / 100
        </p>
      )}

      {result.heatmap_url && (
        <>
          <h3>Heatmap Preview</h3>
          <img
            src={`http://127.0.0.1:5050/outputs/${result.heatmap_url}`}
            alt="Heatmap"
            width="320"
          />
        </>
      )}
    </>
  );
}

/** ------------------------------ Video result ------------------------------ */
function VideoResult({ result }) {
  const ve = result.video_emotions || {};
  const summary = ve.summary || {};
  const finalEmotion = summary.final_top || result.dominant_emotion;
  const avgFaces =
    typeof summary.avg_faces_per_sec === "number"
      ? summary.avg_faces_per_sec
      : result.avg_faces_per_frame;

  const topList =
    (summary.counts &&
      Object.entries(summary.counts)
        .map(([label, count]) => ({ label, count }))
        .sort((a, b) => b.count - a.count)) ||
    result.top_emotions ||
    [];

  return (
    <>
      <p>
        <strong>Duration:</strong> {result.duration_sec}s &nbsp;•&nbsp;{" "}
        <strong>Frames analyzed:</strong> {result.frames_analyzed} &nbsp;•&nbsp;{" "}
        <strong>FPS used:</strong> {result.fps_used}
      </p>

      {finalEmotion && (
        <p>
          <strong>Dominant/Final Emotion:</strong> {String(finalEmotion)}
        </p>
      )}

      {typeof avgFaces !== "undefined" && (
        <p>
          <strong>Avg faces / sec:</strong> {String(avgFaces)}
        </p>
      )}

      {topList.length > 0 && (
        <>
          <p style={{ marginTop: 12 }}>
            <strong>Top Emotions:</strong>
          </p>
          <ul>
            {topList.map((e, i) => (
              <li key={i}>
                {e.label}: {e.count}
              </li>
            ))}
          </ul>
        </>
      )}

      {"objects_top" in result && (
        <>
          <p style={{ marginTop: 12 }}>
            <strong>Objects (top):</strong>
          </p>
          {(result.objects_top || []).length ? (
            <ul>
              {result.objects_top.map((o, i) => (
                <li key={i}>
                  {o.label}: {o.count}
                </li>
              ))}
            </ul>
          ) : (
            <p>(none)</p>
          )}
        </>
      )}

      {"ocr_excerpt" in result && (
        <p style={{ marginTop: 12 }}>
          <strong>OCR Excerpt:</strong>{" "}
          {String(result.ocr_excerpt || "").trim() || "(none)"}
        </p>
      )}

      {"color_palette_global" in result && (
        <>
          <p>
            <strong>Global Palette:</strong>
          </p>
          <div className="colors">
            {(result.color_palette_global || []).map((c, i) => (
              <div key={i} className="color-box" style={{ background: c }} />
            ))}
          </div>
        </>
      )}

      {"layout_balance_avg" in result && (
        <p>
          <strong>Avg Layout Balance:</strong> {String(result.layout_balance_avg)}
        </p>
      )}

      {"brands" in result && (
        <>
          <p style={{ marginTop: 12 }}>
            <strong>Brands (OCR heuristic):</strong>
          </p>
          {(result.brands || []).length ? (
            <ul>
              {result.brands.map((b, i) => (
                <li key={i}>
                  {b.brand} — conf {b.confidence} (src: {b.source})
                </li>
              ))}
            </ul>
          ) : (
            <p>(none)</p>
          )}
        </>
      )}

      {"nsfw" in result && (
        <p style={{ marginTop: 12 }}>
          <strong>NSFW safe:</strong>{" "}
          {String(result.nsfw?.is_safe ?? true)}{" "}
          <small>
            (checked {result.nsfw?.safe_votes ?? 0}/{result.nsfw?.frames_checked ?? 0} frames)
          </small>
        </p>
      )}

      {"creative_score" in result && (
        <p>
          <strong>Creative Score:</strong> {String(result.creative_score)} / 100
        </p>
      )}

      {(result.keyframe_heatmaps || []).length > 0 && (
        <>
          <h3>Keyframe Heatmaps</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
            {result.keyframe_heatmaps.map((h, i) => (
              <img
                key={i}
                src={`http://127.0.0.1:5050/outputs/${h}`}
                alt={`Heatmap ${i}`}
                width="200"
              />
            ))}
          </div>
        </>
      )}

      {/* Gemini Insight */}
      {"insight" in result ? (
        <>
          <h3 style={{ marginTop: 16 }}>Gemini Insight</h3>
          <InsightBlock insight={result.insight} />
        </>
      ) : (
        <p style={{ color: "#888" }}>
          {result.insight_error
            ? `Insight not available: ${String(result.insight_error)}`
            : "Insight not generated (configure GEMINI_API_KEY to enable)."}
        </p>
      )}

      {"video_emotions" in result && result.video_emotions?.per_second && (
        <details style={{ marginTop: 12 }}>
          <summary>Per-second timeline (debug)</summary>
          <pre>{JSON.stringify(result.video_emotions.per_second.slice(0, 30), null, 2)}</pre>
        </details>
      )}
    </>
  );
}

/** ------------------------------ App ------------------------------ */
function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePick = (e) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setPreviewUrl(f ? URL.createObjectURL(f) : "");
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please select an image or video first");

    setLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:5050/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        maxBodyLength: Infinity,
      });
      setResult(res.data);
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Error analyzing creative.");
    } finally {
      setLoading(false);
    }
  };

  const isVideo = file && (file.type || "").startsWith("video/");

  return (
    <div className="App">
      <h1>Roszhan's AdMind AI Analyzer</h1>

      <form onSubmit={handleUpload}>
        <input
          type="file"
          accept="image/*,video/*"
          onChange={handlePick}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Upload & Analyze"}
        </button>
      </form>

      {/* local preview */}
      {previewUrl && (
        <div style={{ marginTop: 12 }}>
          {isVideo ? (
            <video src={previewUrl} width="360" controls />
          ) : (
            <img src={previewUrl} width="360" alt="preview" />
          )}
        </div>
      )}

      {result && (
        <div className="result" style={{ marginTop: 16 }}>
          <h2>Analysis Result</h2>

          {result.media_type === "video" ? (
            <VideoResult result={result} />
          ) : (
            <ImageResult result={result} />
          )}
        </div>
      )}
    </div>
  );
}

export default App;
