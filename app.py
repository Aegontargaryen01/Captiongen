"""
app.py — Gradio Web UI for AI Image Caption Generator
Run:  python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from caption import caption_from_pil, caption_from_url
from PIL import Image

# ── Caption handler ───────────────────────────────────────────────────────────

def handle_upload(image: Image.Image, prompt: str) -> str:
    if image is None:
        return "⚠️  Please upload an image."
    try:
        return caption_from_pil(image, prompt.strip())
    except Exception as e:
        return f"❌ Error: {e}"


def handle_url(url: str, prompt: str) -> tuple:
    if not url.strip():
        return None, "⚠️  Please enter an image URL."
    try:
        import requests
        from PIL import Image
        response = requests.get(url.strip(), stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        caption = caption_from_pil(image, prompt.strip())
        return image, caption
    except Exception as e:
        return None, f"❌ Error: {e}"


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="AI Image Caption Generator",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 860px; margin: auto; }
    h1 { text-align: center; font-size: 2rem; margin-bottom: 0.25rem; }
    .subtitle { text-align: center; color: #666; margin-bottom: 1.5rem; font-size: 0.95rem; }
    """,
) as demo:

    gr.Markdown("# 🖼️ AI Image Caption Generator")
    gr.HTML('<p class="subtitle">Powered by Salesforce BLIP · Hugging Face Transformers</p>')

    with gr.Tab("📁 Upload Image"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload an image")
                prompt_a  = gr.Textbox(
                    label="Conditioning prompt (optional)",
                    placeholder='e.g. "a photo of" or "this image shows"',
                )
                btn_a = gr.Button("Generate Caption", variant="primary")
            with gr.Column():
                caption_a = gr.Textbox(label="Generated Caption", lines=4, interactive=False)

        btn_a.click(fn=handle_upload, inputs=[img_input, prompt_a], outputs=caption_a)

    with gr.Tab("🔗 Image URL"):
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(label="Image URL", placeholder="https://example.com/image.jpg")
                prompt_b  = gr.Textbox(
                    label="Conditioning prompt (optional)",
                    placeholder='e.g. "a photo of"',
                )
                btn_b = gr.Button("Fetch & Caption", variant="primary")
            with gr.Column():
                img_preview = gr.Image(label="Preview", interactive=False)
                caption_b   = gr.Textbox(label="Generated Caption", lines=4, interactive=False)

        btn_b.click(fn=handle_url, inputs=[url_input, prompt_b], outputs=[img_preview, caption_b])

    gr.Examples(
        examples=[
            ["https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg", "a photo of"],
            ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png", ""],
        ],
        inputs=[url_input, prompt_b],
        label="Example URLs",
    )

    gr.Markdown(
        "---\n"
        "**Tips:** A conditioning prompt like `\"a photo of\"` can improve specificity. "
        "Change the model in `caption.py` to `blip-image-captioning-large` for higher quality."
    )

# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(share=False)
