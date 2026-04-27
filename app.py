import logging

import gradio as gr
from rag import run_pipeline
from ingest import build_index
from pathlib import Path
from scorer import score_transformation, format_score_for_ui

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
with gr.Blocks(title="Prompt Forge RAG") as demo:

    gr.Markdown("# 🔥 Prompt Forge RAG")
    gr.Markdown(
        "Paste any messy prompt — get a structured, "
        "model-optimized, copy-ready prompt back."
    )

    with gr.Row():

        # ── Left column — inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
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

            gr.Markdown("### Retrieved exemplars")
            exemplar_display = gr.Textbox(
                label       = "Exemplars used as context",
                lines       = 5,
                interactive = False,
            )

            gr.Markdown("### Quality score")
            score_display = gr.Textbox(
                label       = "Transformation score breakdown",
                lines       = 8,
                interactive = False,
                info        = "Automated quality assessment across 5 dimensions."
            )

        # ── Right column — outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Output")

            status_display = gr.Textbox(
                label       = "Status",
                lines       = 2,
                interactive = False,
            )

            output_display = gr.Textbox(
                label       = "Transformed prompt",
                lines       = 20,
                interactive = False,
            )

            copy_output = gr.Textbox(
                label       = "Copy-paste ready",
                lines       = 4,
                interactive = True,
                info        = "Editable — make final tweaks here before copying."
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
    gr.Markdown(
        "Built with Gradio · BGE-small embeddings · "
        "LanceDB · Groq llama-3.3-70b · Cerebras fallback"
    )

if __name__ == "__main__":
    demo.launch()