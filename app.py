import logging

import gradio as gr
from rag import run_pipeline
from ingest import build_index
from pathlib import Path
from scorer import score_transformation, format_score_for_ui
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CSS = """
body {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 55%, #f8fafc 100%);
    min-height: 100vh;
    color: #0f172a;
}
.gradio-container {
    max-width: 1140px;
    margin: 0 auto;
    padding: 22px 20px 32px;
}
#app_header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 20px;
}
.app-label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #ff7a59;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.82rem;
    font-weight: 700;
}
.app-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0.1rem 0 0.4rem;
    line-height: 1.05;
}
.app-subtitle {
    color: #475569;
    font-size: 1rem;
    max-width: 660px;
    line-height: 1.7;
}
.app-box {
    border-radius: 24px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
    padding: 24px;
}
.app-footer {
    color: #64748b;
    font-size: 0.95rem;
    text-align: center;
    margin-top: 24px;
}
.gradio-row, .gradio-column {
    gap: 24px !important;
}
.gradio-box, .app-box {
    min-width: 0;
}
.gr-textbox, .gr-dropdown, .gr-button, .gr-markdown {
    border-radius: 16px;
}
.gr-button.primary {
    background: #ff7a59 !important;
    color: white !important;
    border: none !important;
    min-height: 54px;
    font-weight: 700;
}
.gr-button.primary:hover {
    background: #f15f3e !important;
}
#status_display textarea, #status_display .gr-textbox {
    min-height: 80px;
    background: #f8fafc;
    border-color: #e2e8f0;
}
#output_display textarea, #copy_output textarea, #exemplar_display textarea, #score_display textarea {
    border-color: #e2e8f0;
}
.gr-block .gr-markdown h1, .gr-block .gr-markdown h2 {
    color: #0f172a;
}
@media (max-width: 920px) {
    .gradio-row {
        flex-direction: column !important;
    }
    #app_header {
        flex-direction: column;
        align-items: flex-start;
    }
}
@media (max-width: 640px) {
    .gradio-container {
        padding: 16px 14px 24px;
    }
    .app-box {
        padding: 18px;
    }
    .gr-button.primary {
        min-height: 48px;
    }
    .app-title {
        font-size: 1.75rem;
    }
}
"""

# ── Build index on startup if missing ─────────────────────────────────────────
print("Checking index...")
if not Path("lancedb_store").exists():
    print("Index not found — building on startup...")
    build_index(force=True)
else:
    print("Index found — skipping rebuild.")

# ── Options ───────────────────────────────────────────────────────────────────
MODELS = [
    "claude-code",
    "gpt-4",
    "cursor",
    "gemini",
    "general",
]

DEPTHS = [
    "concise",
    "standard",
    "comprehensive",
]

# ── Core function called by Gradio ────────────────────────────────────────────
def forge(raw_prompt: str, target_model: str, depth: str):
    raw_prompt = raw_prompt.strip()
    if not raw_prompt:
        return "", "⚠️ Please paste a prompt first.", "", "", ""
    if len(raw_prompt) < 15:
        return "", "⚠️ Prompt too short — please add more detail.", "", "", ""
    if target_model not in MODELS:
        return "", "⚠️ Invalid target model selected.", "", "", ""
    if depth not in DEPTHS:
        return "", "⚠️ Invalid depth selected.", "", "", ""

    try:
        result = run_pipeline(
            raw_prompt   = raw_prompt,
            target_model = target_model,
            depth        = depth,
        )
    except Exception as e:
        logger.exception("Unhandled error in forge()")
        return "", f"❌ Error: {str(e)}", "", "", ""

    if result["error"]:
        return "", f"❌ Error: {result['error']}", "", "", ""

    # Score the transformation
    score_result = score_transformation(
        raw_prompt   = raw_prompt,
        output       = result["transformed"],
        target_model = target_model,
        task_type    = result["task_type"],
    )
    score_text = format_score_for_ui(score_result)

    # Format exemplars
    exemplar_text = ""
    if result["exemplars"]:
        for i, ex in enumerate(result["exemplars"], 1):
            exemplar_text += (
                f"Exemplar {i}: [{ex.get('target_model')} / "
                f"{ex.get('task_type')}] "
                f"from {ex.get('source_repo')}\n"
            )
    else:
        exemplar_text = "No exemplars retrieved."

    usage = result.get("usage", {})
    token_info = ""
    if usage:
        total_tokens = usage.get("total_tokens")
        cost = usage.get("cost")
        approximate = usage.get("approximate", False)
        token_info = (
            f" | Tokens: {total_tokens} "
            f"| Cost: ${cost:.6f} "
            f"({'approx' if approximate else 'exact'})"
        )

    stats = (
        f"✅ Provider: {result['provider']}  |  "
        f"Task: {result['task_type']}  |  "
        f"Exemplars: {len(result['exemplars'])}  |  "
        f"Score: {score_result['overall']}/10 — {score_result['grade']}"
        f"{token_info}"
    )

    return (
        result["transformed"],
        stats,
        exemplar_text,
        result["transformed"],
        score_text,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Prompt Forge RAG", css=CSS) as demo:

    with gr.Row(elem_id="app_header"):
        with gr.Column(scale=1, min_width=0):
            gr.Markdown(
                """
                <div class=\"app-label\">Prompt Forge RAG</div>
                <div class=\"app-title\">Clean prompts. Faster results. Better models.</div>
                <div class=\"app-subtitle\">Paste any messy prompt and get a polished, model-optimized instruction set ready for production.</div>
                """
            )
        with gr.Column(scale=0, min_width=260):
            gr.Markdown(
                """
                - ✅ Lightweight modern UI
                - ✅ Token & cost visibility
                - ✅ Mobile-friendly layout
                """
            )

    with gr.Row():

        # ── Left column — inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["app-box"]):
                gr.Markdown("### Input")
                raw_input = gr.Textbox(
                    label       = "Your raw prompt",
                    placeholder = "e.g. write me a python script that reads csv and finds duplicates...",
                    lines       = 12,
                )

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices = MODELS,
                        value   = "general",
                        label   = "Target model",
                    )
                    depth_dropdown = gr.Dropdown(
                        choices = DEPTHS,
                        value   = "standard",
                        label   = "Depth",
                    )

                forge_btn = gr.Button(
                    "⚡ Forge it",
                    variant = "primary",
                    size    = "lg",
                )

            with gr.Column(elem_classes=["app-box"]):
                gr.Markdown("### Retrieved exemplars")
                exemplar_display = gr.Textbox(
                    label       = "Exemplars used as context",
                    lines       = 5,
                    interactive = False,
                )

            with gr.Column(elem_classes=["app-box"]):
                gr.Markdown("### Quality score")
                score_display = gr.Textbox(
                    label       = "Transformation score breakdown",
                    lines       = 8,
                    interactive = False,
                    info        = "Automated quality assessment across 5 dimensions."
                )

        # ── Right column — outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["app-box"]):
                gr.Markdown("### Output")

                status_display = gr.Textbox(
                    label       = "Status",
                    lines       = 2,
                    interactive = False,
                    elem_id     = "status_display",
                )

                output_display = gr.Textbox(
                    label       = "Transformed prompt",
                    lines       = 20,
                    interactive = False,
                    elem_id     = "output_display",
                )

            with gr.Column(elem_classes=["app-box"]):
                gr.Markdown("### Copy-paste ready")
                copy_output = gr.Textbox(
                    label       = "Editable prompt",
                    lines       = 6,
                    interactive = True,
                    info        = "Make final tweaks here before copying.",
                    elem_id     = "copy_output",
                )

    # ── Examples ─────────────────────────────────────────────────────────────
    gr.Markdown("### Try these examples")
    gr.Examples(
        examples=[
            [
                "write me a python function that reads a csv and finds duplicate rows",
                "claude-code",
                "standard",
            ],
            [
                "fix the bug in my code its not working",
                "cursor",
                "concise",
            ],
            [
                "analyze this data and tell me whats interesting",
                "gemini",
                "comprehensive",
            ],
            [
                "write a blog post about ai trends",
                "gpt-4",
                "standard",
            ],
            [
                "make a system prompt for a customer support agent",
                "general",
                "comprehensive",
            ],
            [
                "i want to build a n8n workflow that reads leads from google sheets and sends personalized emails",
                "claude-code",
                "comprehensive",
            ],
        ],
        inputs=[raw_input, model_dropdown, depth_dropdown],
    )

    # ── Wire up the button ────────────────────────────────────────────────────
    forge_btn.click(
        fn      = forge,
        inputs  = [raw_input, model_dropdown, depth_dropdown],
        outputs = [
            output_display,
            status_display,
            exemplar_display,
            copy_output,
            score_display,
        ],
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    with gr.Row():
        gr.Markdown(
            '<div class="app-footer">Built with Gradio · BGE-small embeddings · LanceDB · Groq llama-3.3-70b · Cerebras fallback</div>'
        )

if __name__ == "__main__":
    demo.launch()