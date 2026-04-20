# AI Image Caption Generator

A Python toolkit for generating AI-powered captions for images using
Salesforce BLIP via Hugging Face Transformers.

## Project Structure

```
image-caption-generator/
├── caption.py        # Core library — import this in your own code
├── app.py            # Gradio web UI
├── api.py            # Flask REST API
├── batch.py          # Batch processor (folder or URL list → CSV/JSON)
└── requirements.txt  # Python dependencies
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> The first run downloads the BLIP model (~1 GB) and caches it locally.

---

## Usage

### 1. Command Line (caption.py)

```bash
# Caption a local file
python caption.py --file photo.jpg

# Caption from a URL
python caption.py --url https://example.com/dog.jpg

# With a conditioning prompt
python caption.py --file photo.jpg --prompt "a photo of"
```

### 2. Web UI (app.py)

```bash
python app.py
# Opens http://localhost:7860
```

Supports both file upload and URL tabs with optional prompt conditioning.

### 3. REST API (api.py)

```bash
python api.py
# Runs on http://localhost:5000
```

**Endpoints:**

```
GET  /health              → { "status": "ok" }

POST /caption/upload      → multipart form
  file=<image>
  prompt=<optional>
  → { "caption": "...", "elapsed_ms": 210 }

POST /caption/url         → JSON body
  { "url": "https://...", "prompt": "" }
  → { "caption": "...", "elapsed_ms": 185 }
```

**Example with curl:**
```bash
# Upload file
curl -X POST http://localhost:5000/caption/upload \
  -F "file=@photo.jpg" \
  -F "prompt=a photo of"

# From URL
curl -X POST http://localhost:5000/caption/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/cat.jpg"}'
```

### 4. Batch Processing (batch.py)

```bash
# Caption all images in a folder → CSV
python batch.py --folder ./photos --output results.csv

# Caption URLs from a text file → JSON
python batch.py --urls urls.txt --output results.json

# Parallel processing (2 threads)
python batch.py --folder ./photos --output results.csv --workers 2
```

`urls.txt` format — one URL per line, `#` for comments:
```
https://example.com/img1.jpg
https://example.com/img2.png
# this line is ignored
```

---

## Changing the Model

Edit `caption.py` or set the `CAPTION_MODEL` environment variable:

| Model | Size | Quality |
|---|---|---|
| `Salesforce/blip-image-captioning-base` | ~990 MB | Good (default) |
| `Salesforce/blip-image-captioning-large` | ~1.9 GB | Better |
| `Salesforce/blip2-opt-2.7b` | ~15 GB | Excellent |

```bash
CAPTION_MODEL=Salesforce/blip-image-captioning-large python app.py
```

---

## Using as a Library

```python
from caption import caption_from_file, caption_from_url, caption_from_pil
from PIL import Image

# From file
print(caption_from_file("photo.jpg"))

# From URL
print(caption_from_url("https://example.com/image.jpg"))

# From PIL Image
img = Image.open("photo.jpg")
print(caption_from_pil(img, prompt="a photo of"))
```
