import gradio as gr
from rag import run_pipeline
from ingest import build_index
from pathlib import Path

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
    if not raw_prompt.strip():
        return (
            "",
            "⚠️ Please paste a prompt first.",
            "",
            "",
        )

    result = run_pipeline(
        raw_prompt   = raw_prompt,
        target_model = target_model,
        depth        = depth,
    )

    if result["error"]:
        return (
            "",
            f"❌ Error: {result['error']}",
            "",
            "",
        )

    # Format exemplars for display
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

    stats = (
        f"✅ Provider: {result['provider']}  |  "
        f"Task detected: {result['task_type']}  |  "
        f"Exemplars used: {len(result['exemplars'])}"
    )

    return (
        result["transformed"],
        stats,
        exemplar_text,
        result["transformed"],
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Prompt Forge RAG") as demo:

    gr.Markdown("# 🔥 Prompt Forge RAG")
    gr.Markdown(
        "Paste any messy prompt — get a structured, model-optimized, copy-ready prompt back."
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
                lines       = 6,
                interactive = False,
            )

        # ── Right column — outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Output")

            status_display = gr.Textbox(
                label       = "Status",
                lines       = 1,
                interactive = False,
            )

            output_display = gr.Textbox(
                label       = "Transformed prompt",
                lines       = 18,
                interactive = False,
            )

            copy_output = gr.Textbox(
                label       = "Copy-paste ready (same content)",
                lines       = 3,
                interactive = True,
                info        = "This box is editable — make final tweaks here before copying."
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
        ],
        inputs=[raw_input, model_dropdown, depth_dropdown],
    )

    # ── Wire up the button ────────────────────────────────────────────────────
    forge_btn.click(
        fn      = forge,
        inputs  = [raw_input, model_dropdown, depth_dropdown],
        outputs = [output_display, status_display, exemplar_display, copy_output],
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(
        "Built with Gradio · BGE-small embeddings · LanceDB · Gemini 2.0 Flash + Cerebras fallback"
    )

if __name__ == "__main__":
    demo.launch()