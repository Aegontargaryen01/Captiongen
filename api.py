"""
api.py — Flask REST API for AI Image Caption Generator
Run:  python api.py
      (or use gunicorn for production: gunicorn api:app)

Endpoints:
  POST /caption/upload   — multipart image file upload
  POST /caption/url      — JSON body with image URL
  GET  /health           — health check
"""

from flask import Flask, request, jsonify
from PIL import Image
import io
import time

from caption import caption_from_pil, caption_from_url

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Simple health check."""
    return jsonify({"status": "ok", "model": "blip-image-captioning-base"})


@app.post("/caption/upload")
def caption_upload():
    """
    Generate a caption from an uploaded image file.

    Form fields:
      file    (required) — image file
      prompt  (optional) — conditioning text prompt

    Returns JSON:
      { "caption": "...", "elapsed_ms": 123 }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 415

    prompt = request.form.get("prompt", "").strip()

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        t0 = time.perf_counter()
        caption = caption_from_pil(image, prompt)
        elapsed = round((time.perf_counter() - t0) * 1000)

        return jsonify({"caption": caption, "elapsed_ms": elapsed})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/caption/url")
def caption_from_url_route():
    """
    Generate a caption from an image URL.

    JSON body:
      { "url": "https://...", "prompt": "optional" }

    Returns JSON:
      { "caption": "...", "elapsed_ms": 123 }
    """
    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "JSON body must contain 'url'"}), 400

    url    = data["url"].strip()
    prompt = data.get("prompt", "").strip()

    try:
        t0 = time.perf_counter()
        caption = caption_from_url(url, prompt)
        elapsed = round((time.perf_counter() - t0) * 1000)

        return jsonify({"caption": caption, "url": url, "elapsed_ms": elapsed})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large. Max 16 MB."}), 413


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
