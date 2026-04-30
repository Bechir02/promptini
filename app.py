from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from rag import run_pipeline
from ingest import build_index
from pathlib import Path
from scorer import score_transformation, format_score_for_ui

# ── Build index on startup if missing ─────────────────────────────────────────
print("Checking index...")
if not Path("lancedb_store").exists():
    print("Index not found — building on startup...")
    build_index(force=True)
else:
    print("Index found — skipping rebuild.")

MODELS = [
    "claude-code",
    "claude",
    "gpt-4",
    "cursor",
    "gemini",
    "llama",
    "mistral",
    "copilot",
    "general",
]
DEPTHS = ["concise", "standard", "comprehensive"]

# ── Core function ─────────────────────────────────────────────────────────────
def forge(raw_prompt: str, target_model: str, depth: str):
    if not raw_prompt or not raw_prompt.strip():
        return "", "", "", "", ""
    if len(raw_prompt.strip()) < 15:
        return "", "⚠ Prompt too short — add more detail.", "", "", ""
    if target_model not in MODELS:
        return "", "⚠ Invalid model.", "", "", ""
    if depth not in DEPTHS:
        return "", "⚠ Invalid depth.", "", "", ""

    result = run_pipeline(
        raw_prompt   = raw_prompt,
        target_model = target_model,
        depth        = depth,
    )

    if result["error"]:
        return "", f"✗ {result['error']}", "", "", ""

    score_result = score_transformation(
        raw_prompt   = raw_prompt,
        output       = result["transformed"],
        target_model = target_model,
        task_type    = result["task_type"],
    )

    exemplar_text = ""
    if result["exemplars"]:
        exemplar_text = "Matched prompt examples used for style + structure:\n\n"
        for i, ex in enumerate(result["exemplars"], 1):
            exemplar_text += (
                f"{i:02d}  [{ex.get('target_model')}]  "
                f"{ex.get('task_type')}  ·  "
                f"{ex.get('source_repo')}\n"
            )
    else:
        exemplar_text = "No exemplars retrieved."

    score_text = format_score_for_ui(score_result)
    meta = (
        f"{len(result['transformed'])} chars · "
        f"{len(result['transformed'].split())} words"
    )

    status = (
        f"✓ {result['provider']} · "
        f"task: {result['task_type']} · "
        f"score: {score_result['overall']}/10"
    )

    reasoning_text = f"REASONING:\n{score_result['breakdown']['llm_judge']['note']}\n\n"
    if score_result['breakdown']['llm_judge']['strengths']:
        reasoning_text += "STRENGTHS:\n- " + "\n- ".join(score_result['breakdown']['llm_judge']['strengths']) + "\n\n"
    if score_result['breakdown']['llm_judge']['weaknesses']:
        reasoning_text += "WEAKNESSES:\n- " + "\n- ".join(score_result['breakdown']['llm_judge']['weaknesses'])
    
    score_display_text = format_score_for_ui(score_result)
    full_score_text = f"{score_display_text}\n\n{reasoning_text}"

    return (
        result["transformed"],
        status,
        exemplar_text,
        full_score_text,
        meta,
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }

body, .gradio-container, .main, .wrap {
    background: #edecea !important;
    font-family: 'DM Sans', sans-serif !important;
}

footer, .footer { display: none !important; }
.hide-footer { display: none !important; }

/* Header */
.pf-header {
    text-align: center;
    padding: 32px 24px 28px;
    background: transparent;
}

/* Panels */
.panel-card {
    background: white !important;
    border-radius: 22px !important;
    border: none !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.055), 0 1px 3px rgba(0,0,0,0.04) !important;
    padding: 22px !important;
}

.bottom-card {
    background: white !important;
    border-radius: 20px !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.045), 0 1px 3px rgba(0,0,0,0.03) !important;
    padding: 20px 22px !important;
}

/* Labels */
label > span, .form > label > span {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #9d9991 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Textareas */
textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: #2a2a28 !important;
    background: white !important;
    border: 0.5px solid #ebebea !important;
    border-radius: 14px !important;
    line-height: 1.75 !important;
    padding: 14px 16px !important;
    min-height: 320px !important;
    resize: none !important;
}

textarea:focus {
    border-color: #c4633e !important;
    box-shadow: 0 0 0 3px rgba(196,99,62,0.07) !important;
    outline: none !important;
}

textarea::placeholder { color: #bbb7af !important; }

/* Output textarea — monospace */
.output-text textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    color: #333 !important;
    line-height: 1.85 !important;
}

/* Score and exemplar textareas */
.score-text textarea, .exemplar-text textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    color: #555 !important;
    min-height: 140px !important;
    background: #fafaf9 !important;
    border-color: #f0f0ee !important;
    line-height: 1.8 !important;
}

/* Status box */
.status-text textarea {
    font-size: 12px !important;
    color: #5a9e4a !important;
    background: #f8faf7 !important;
    border-color: #e8f0e5 !important;
    min-height: 44px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Meta box */
.meta-text textarea {
    font-size: 11.5px !important;
    color: #9d9991 !important;
    background: transparent !important;
    border: none !important;
    min-height: 24px !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* Dropdowns */
.gr-dropdown {
    background: #f7f7f5 !important;
    border: 0.5px solid #ebebea !important;
    border-radius: 20px !important;
    font-size: 13px !important;
    color: #333 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
}

/* Primary button */
button.primary, .primary {
    background: #c4633e !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: background 0.15s, transform 0.15s !important;
    box-shadow: none !important;
}

button.primary:hover { background: #d97040 !important; transform: translateY(-1px); }

/* Secondary button */
button.secondary, .secondary {
    background: #f7f7f5 !important;
    color: #555 !important;
    border: 0.5px solid #ebebea !important;
    border-radius: 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: none !important;
    transition: background 0.15s, transform 0.15s !important;
}

button.secondary:hover { background: #efefed !important; transform: translateY(-1px); }

/* Remove all gradio default borders and backgrounds */
.form, .block, .gap, .padded {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gr-group, .gr-box {
    background: transparent !important;
    border: none !important;
}

/* Accordion */
.gr-accordion {
    background: white !important;
    border-radius: 16px !important;
    border: 0.5px solid #f0f0ee !important;
    box-shadow: none !important;
}

@media (max-width: 768px) {
    .pf-header {
        padding: 26px 16px 22px;
    }

    .panel-card, .bottom-card {
        padding: 18px !important;
    }

    textarea {
        min-height: 260px !important;
    }

    .score-text textarea, .exemplar-text textarea {
        min-height: 130px !important;
    }
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Prompt Forge", css=CSS) as demo:

    # Header
    gr.Markdown("""
<div class="pf-header">
<div style="display:inline-flex;align-items:center;gap:5px;font-size:11.5px;font-weight:500;color:#c4633e;background:white;border:0.5px solid #f0d0c0;padding:5px 14px;border-radius:20px;margin-bottom:18px;">⚡ RAG · 8,435 prompts</div>
<h1 style="font-size:32px;font-weight:600;color:#1c1c1a;letter-spacing:-0.04em;line-height:1.18;margin-bottom:12px;font-family:'DM Sans',sans-serif;">Turn messy prompts into <span style="color:#c4633e;">model-ready instructions</span></h1>
<p style="font-size:14px;color:#999;line-height:1.65;max-width:460px;margin:0 auto;font-family:'DM Sans',sans-serif;">RAG-powered prompt rewriting for Claude Code, GPT-4, Cursor, Gemini, and general use.</p>
</div>
""")

    # Main two columns
    with gr.Row(equal_height=False):

        # Left — input
        with gr.Column(scale=1, elem_classes="panel-card"):

            raw_input = gr.Textbox(
                label       = "Messy prompt",
                placeholder = "Paste rough notes, a vague request, or a messy first draft...",
                lines       = 16,
                max_lines   = 24,
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
                "⚡ Transform prompt",
                variant = "primary",
                size    = "lg",
            )

        # Right — output
        with gr.Column(scale=1, elem_classes="panel-card"):

            status_display = gr.Textbox(
                label       = "Status",
                lines       = 1,
                interactive = False,
                elem_classes= "status-text",
            )

            output_display = gr.Textbox(
                label       = "Structured prompt",
                lines       = 16,
                max_lines   = 24,
                interactive = True,
                elem_classes= "output-text",
            )

            output_meta = gr.Textbox(
                label       = "",
                lines       = 1,
                interactive = False,
                elem_classes= "meta-text",
            )

            copy_btn = gr.Button(
                "Copy to clipboard",
                variant = "secondary",
                size    = "lg",
            )

    # Bottom row
    with gr.Row(equal_height=True):

        with gr.Column(scale=1, elem_classes="bottom-card"):
            exemplar_display = gr.Textbox(
                label       = "Matched exemplars",
                lines       = 6,
                interactive = False,
                elem_classes= "exemplar-text",
            )

        with gr.Column(scale=1, elem_classes="bottom-card"):
            score_display = gr.Textbox(
                label       = "Quality score & Reasoning",
                lines       = 8,
                interactive = False,
                elem_classes= "score-text",
            )

    # Event handlers
    forge_btn.click(
        fn=forge,
        inputs=[raw_input, model_dropdown, depth_dropdown],
        outputs=[output_display, status_display, exemplar_display, score_display, output_meta]
    )

    )

    copy_btn.click(
        fn=None,
        inputs=output_display,
        js="(v) => { navigator.clipboard.writeText(v); alert('Copied to clipboard!'); }"
    )

    # Inject JS to handle messages from VS Code
    demo.load(None, None, None, js="""
    () => {
        window.addEventListener('message', (event) => {
            if (event.data.type === 'setPrompt') {
                const textarea = document.querySelector('textarea[data-testid="textbox"]');
                if (textarea) {
                    textarea.value = event.data.text;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        });
    }
    """)

if __name__ == "__main__":
    demo.launch()