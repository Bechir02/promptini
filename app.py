import gradio as gr
from rag import run_pipeline
from ingest import build_index
from pathlib import Path
from scorer import score_transformation, format_score_for_ui
from dotenv import load_dotenv
load_dotenv()

# ── Build index on startup if missing ─────────────────────────────────────────
print("Checking index...")
if not Path("lancedb_store").exists():
    print("Index not found — building on startup...")
    build_index(force=True)
else:
    print("Index found — skipping rebuild.")

# ── Options ───────────────────────────────────────────────────────────────────
MODELS = ["claude-code", "gpt-4", "cursor", "gemini", "general"]
DEPTHS = ["concise", "standard", "comprehensive"]

# ── Core function ─────────────────────────────────────────────────────────────
def forge(raw_prompt: str, target_model: str, depth: str):
    if not raw_prompt.strip():
        return "", "⚠ Enter a prompt to transform.", "", ""
    if len(raw_prompt.strip()) < 15:
        return "", "⚠ Prompt too short — add more detail.", "", ""
    if target_model not in MODELS:
        return "", "⚠ Invalid model selected.", "", ""
    if depth not in DEPTHS:
        return "", "⚠ Invalid depth selected.", "", ""

    result = run_pipeline(
        raw_prompt   = raw_prompt,
        target_model = target_model,
        depth        = depth,
    )

    if result["error"]:
        return "", f"✗ {result['error']}", "", ""

    score_result = score_transformation(
        raw_prompt   = raw_prompt,
        output       = result["transformed"],
        target_model = target_model,
        task_type    = result["task_type"],
    )

    status = (
        f"✓  {result['provider']}  ·  "
        f"task: {result['task_type']}  ·  "
        f"exemplars: {len(result['exemplars'])}  ·  "
        f"score: {score_result['overall']}/10 — {score_result['grade']}"
    )

    exemplar_text = ""
    if result["exemplars"]:
        for i, ex in enumerate(result["exemplars"], 1):
            exemplar_text += (
                f"{i}. [{ex.get('target_model')} / {ex.get('task_type')}] "
                f"{ex.get('source_repo')}\n"
            )
    else:
        exemplar_text = "No exemplars retrieved."

    score_text = format_score_for_ui(score_result)

    return (
        result["transformed"],
        status,
        exemplar_text,
        score_text,
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0d0d0d !important;
    color: #e8e0d0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide gradio branding */
footer { display: none !important; }
.svelte-1kcf4d2 { display: none !important; }

/* Header */
.header-block {
    border-bottom: 1px solid #222;
    padding-bottom: 24px;
    margin-bottom: 32px;
}

/* Labels */
label span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #666 !important;
}

/* Textareas and inputs */
textarea, input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    background: #111 !important;
    border: 1px solid #222 !important;
    color: #e8e0d0 !important;
    border-radius: 2px !important;
    line-height: 1.6 !important;
}
textarea:focus, input:focus {
    border-color: #c4633e !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Dropdowns */
.wrap { background: #111 !important; border: 1px solid #222 !important; border-radius: 2px !important; }
.wrap:hover { border-color: #444 !important; }
select { background: #111 !important; color: #e8e0d0 !important; }

/* Primary button */
button.primary {
    background: #c4633e !important;
    color: #0d0d0d !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 14px 32px !important;
    transition: background 0.15s ease !important;
}
button.primary:hover {
    background: #d97842 !important;
}

/* Secondary buttons */
button.secondary {
    background: transparent !important;
    color: #666 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
}
button.secondary:hover {
    border-color: #444 !important;
    color: #e8e0d0 !important;
}

/* Status box */
.status-box textarea {
    font-size: 11.5px !important;
    color: #6b8e4e !important;
    background: #0d0d0d !important;
    border: none !important;
    border-top: 1px solid #1a1a1a !important;
    padding-top: 8px !important;
}

/* Score box */
.score-box textarea {
    font-size: 11.5px !important;
    color: #8a7a62 !important;
    background: #111 !important;
    line-height: 1.8 !important;
}

/* Exemplar box */
.exemplar-box textarea {
    font-size: 11px !important;
    color: #555 !important;
    background: #0d0d0d !important;
    border-color: #1a1a1a !important;
}

/* Output box */
.output-box textarea {
    font-size: 13px !important;
    line-height: 1.7 !important;
    color: #e8e0d0 !important;
    background: #111 !important;
}

/* Divider */
.divider {
    height: 1px;
    background: #1a1a1a;
    margin: 24px 0;
}

/* Accordion */
.accordion {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 2px !important;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Prompt Forge", css=CSS) as demo:

    # Header
    with gr.Group(elem_classes="header-block"):
        gr.Markdown("""# Prompt Forge
Transform messy prompts into structured, model-optimized instructions. Powered by RAG — 2158 real prompts from Claude Code, GPT-4, Cursor, and Gemini.""")

    with gr.Row(equal_height=False):

        # ── Left column ───────────────────────────────────────────────────────
        with gr.Column(scale=1):

            raw_input = gr.Textbox(
                label       = "Raw prompt",
                placeholder = "Paste your messy prompt here...",
                lines       = 10,
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
                "⚡ Forge",
                variant = "primary",
                size    = "lg",
            )

            with gr.Accordion("Retrieved exemplars", open=False, elem_classes="accordion"):
                exemplar_display = gr.Textbox(
                    label       = "Sources",
                    lines       = 4,
                    interactive = False,
                    elem_classes= "exemplar-box",
                )

            with gr.Accordion("Quality score", open=False, elem_classes="accordion"):
                score_display = gr.Textbox(
                    label       = "Score breakdown",
                    lines       = 8,
                    interactive = False,
                    elem_classes= "score-box",
                )

        # ── Right column ──────────────────────────────────────────────────────
        with gr.Column(scale=1):

            status_display = gr.Textbox(
                label       = "Status",
                lines       = 1,
                interactive = False,
                elem_classes= "status-box",
            )

            output_display = gr.Textbox(
                label       = "Transformed prompt",
                lines       = 24,
                interactive = True,
                elem_classes= "output-box",
                info        = "Editable — tweak before copying.",
            )

    # Footer
    gr.Markdown(
        "Prompt Forge · BGE-small · LanceDB · Groq llama-3.3-70b · "
        "[huggingface.co/spaces/Becher-zribi/prompt-forge-rag]"
        "(https://huggingface.co/spaces/Becher-zribi/prompt-forge-rag)",
    )

    # Wire up
    forge_btn.click(
        fn      = forge,
        inputs  = [raw_input, model_dropdown, depth_dropdown],
        outputs = [output_display, status_display, exemplar_display, score_display],
    )

if __name__ == "__main__":
    demo.launch()