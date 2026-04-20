"""
caption.py — Core AI Image Captioning Module
Uses Salesforce BLIP model via Hugging Face Transformers.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import os

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
# Options:
#   "Salesforce/blip-image-captioning-base"   (~990 MB, fast)
#   "Salesforce/blip-image-captioning-large"  (~1.9 GB, better quality)

_processor = None
_model = None


def _load_model():
    """Lazy-load the model (only once)."""
    global _processor, _model
    if _processor is None:
        print(f"[caption] Loading model: {MODEL_NAME} ...")
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        print(f"[caption] Model loaded on {device}.")
    return _processor, _model


# ── Public API ────────────────────────────────────────────────────────────────

def caption_from_pil(image: Image.Image, prompt: str = "") -> str:
    """
    Generate a caption for a PIL Image object.

    Args:
        image:  PIL Image (RGB)
        prompt: Optional text prompt to condition the caption
                e.g. "a photo of" helps the model focus.

    Returns:
        Caption string.
    """
    processor, model = _load_model()
    image = image.convert("RGB")

    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)
    else:
        inputs = processor(image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=60)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def caption_from_file(path: str, prompt: str = "") -> str:
    """Generate a caption for a local image file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path)
    return caption_from_pil(image, prompt)


def caption_from_url(url: str, prompt: str = "", timeout: int = 10) -> str:
    """Generate a caption for an image fetched from a URL."""
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    image = Image.open(response.raw)
    return caption_from_pil(image, prompt)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Image Caption Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python caption.py --file photo.jpg
  python caption.py --url https://example.com/dog.jpg
  python caption.py --file photo.jpg --prompt "a photo of"
        """,
    )
    parser.add_argument("--file",   type=str, help="Path to a local image file")
    parser.add_argument("--url",    type=str, help="URL of an image")
    parser.add_argument("--prompt", type=str, default="", help="Optional conditioning prompt")
    args = parser.parse_args()

    if args.file:
        result = caption_from_file(args.file, args.prompt)
    elif args.url:
        result = caption_from_url(args.url, args.prompt)
    else:
        parser.error("Provide --file or --url")

    print(f"\nCaption: {result}\n")
