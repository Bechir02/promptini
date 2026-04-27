from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import tempfile
import gradio as gr

from rag import run_pipeline
from ingest import build_index
from scorer import score_transformation, format_score_for_ui


# ── Startup ───────────────────────────────────────────────────────────────────
print("Checking index...")
if not Path("lancedb_store").exists():
    print("Index not found — building on startup...")
    build_index(force=True)
else:
    print("Index found — skipping rebuild.")


MODELS = ["claude-code", "gpt-4", "cursor", "gemini", "general"]
DEPTHS = ["concise", "standard", "comprehensive"]


# ── Small UI formatters ───────────────────────────────────────────────────────
def grade_label(score: float) -> str:
    if score >= 9.0:
        return "A — Excellent"
    if score >= 7.5:
        return "B — Good"
    if score >= 6.0:
        return "C — Acceptable"
    if score >= 4.0:
        return "D — Needs Work"
    return "F — Poor"


def render_score_card(score_result: dict) -> str:
    """Render the scorer output as a compact, readable Markdown/HTML card."""
    overall = score_result["overall"]
    grade = score_result.get("grade", grade_label(overall))
    breakdown = score_result["breakdown"]

    rows = []
    labels = {
        "structure": "Structure",
        "specificity": "Specificity",
        "model_aware": "Model-aware",
        "task_cover": "Task coverage",
        "improvement": "Improvement",
    }

    for key, label in labels.items():
        item = breakdown[key]
        score = float(item["score"])
        width = max(0, min(100, score * 10))
        note = item.get("note", "")
        rows.append(
            f"""
            <div class="score-row">
                <div class="score-row-top">
                    <span>{label}</span>
                    <strong>{score:.1f}/10</strong>
                </div>
                <div class="score-track"><div class="score-fill" style="width:{width}%"></div></div>
                <p>{note}</p>
            </div>
            """
        )

    return f"""
    <div class="score-card">
        <div class="score-hero">
            <div>
                <div class="eyebrow">Quality score</div>
                <h3>{overall}/10</h3>
            </div>
            <div class="grade-pill">{grade}</div>
        </div>
        {''.join(rows)}
    </div>
    """


def format_exemplars(exemplars: list[dict]) -> str:
    """Summarize retrieved exemplars as trust signals."""
    if not exemplars:
        return "No exemplars retrieved."

    lines = [
        "Matched examples that shaped this output:",
        "",
    ]

    for i, ex in enumerate(exemplars, 1):
        lines.append(
            f"{i:02d}  [{ex.get('target_model', 'unknown')}]  "
            f"{ex.get('task_type', 'general')}  ·  "
            f"{ex.get('source_repo', 'unknown source')}"
        )

    return "\n".join(lines)


def render_project_facts(result: dict, score_result: dict, transformed: str) -> str:
    usage = result.get("usage") or {}
    usage_bits = []

    if isinstance(usage, dict):
        for key in ["input_tokens", "output_tokens", "total_tokens", "estimated_cost"]:
            if key in usage and usage[key] is not None:
                usage_bits.append(f"{key.replace('_', ' ')}: {usage[key]}")

    usage_text = " · ".join(usage_bits) if usage_bits else "Usage data unavailable"

    return f"""
    <div class="facts-grid">
        <div class="fact-card">
            <span>Detected task</span>
            <strong>{result.get("task_type", "general")}</strong>
        </div>
        <div class="fact-card">
            <span>Provider</span>
            <strong>{result.get("provider", "unknown")}</strong>
        </div>
        <div class="fact-card">
            <span>Output size</span>
            <strong>{len(transformed)} chars · {len(transformed.split())} words</strong>
        </div>
        <div class="fact-card">
            <span>Score</span>
            <strong>{score_result["overall"]}/10</strong>
        </div>
    </div>
    <div class="usage-line">{usage_text}</div>
    """


def write_prompt_file(prompt_text: str):
    """Create a downloadable .txt file for the transformed prompt."""
    if not prompt_text.strip():
        return None

    path = Path(tempfile.gettempdir()) / "prompt_forge_output.txt"
    path.write_text(prompt_text, encoding="utf-8")
    return str(path)


# ── Core function ─────────────────────────────────────────────────────────────
def forge(raw_prompt: str, target_model: str, depth: str, top_k: int):
    """
    Transform a rough prompt into a model-aware structured prompt.

    Output order must match the Gradio outputs:
    1. output_display
    2. status_display
    3. exemplar_display
    4. score_display
    5. facts_display
    6. output_meta
    """
    empty_score = "<div class='empty-state'>Run a transformation to see the score breakdown.</div>"
    empty_facts = "<div class='empty-state'>Detected task, provider, and usage will appear here.</div>"

    if not raw_prompt or not raw_prompt.strip():
        return "", "", "", empty_score, empty_facts, ""

    if len(raw_prompt.strip()) < 15:
        return (
            "",
            "⚠ Prompt too short — add more detail before transforming.",
            "",
            empty_score,
            empty_facts,
            "",
        )

    if target_model not in MODELS:
        return "", "⚠ Invalid model.", "", empty_score, empty_facts, ""

    if depth not in DEPTHS:
        return "", "⚠ Invalid depth.", "", empty_score, empty_facts, ""

    result = run_pipeline(
        raw_prompt=raw_prompt,
        target_model=target_model,
        depth=depth,
        top_k=int(top_k),
    )

    if result.get("error"):
        return (
            "",
            f"✗ {result['error']}",
            "",
            empty_score,
            empty_facts,
            "",
        )

    transformed = result["transformed"]

    score_result = score_transformation(
        raw_prompt=raw_prompt,
        output=transformed,
        target_model=target_model,
        task_type=result["task_type"],
    )

    exemplar_text = format_exemplars(result.get("exemplars", []))
    score_card = render_score_card(score_result)
    facts = render_project_facts(result, score_result, transformed)

    meta = f"{len(transformed)} chars · {len(transformed.split())} words"
    status = (
        f"✓ {result['provider']} · "
        f"task: {result['task_type']} · "
        f"score: {score_result['overall']}/10"
    )

    return (
        transformed,
        status,
        exemplar_text,
        score_card,
        facts,
        meta,
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #edecea;
    --panel: #ffffff;
    --ink: #1f1f1d;
    --muted: #8a8781;
    --soft: #f7f6f3;
    --line: #ece8e1;
    --accent: #c4633e;
    --accent-dark: #a94f31;
    --accent-soft: #fbefe9;
    --green: #4f9850;
}

* {
    font-family: 'DM Sans', sans-serif !important;
    box-sizing: border-box;
}

body,
.gradio-container,
.main,
.wrap {
    background: var(--bg) !important;
    color: var(--ink) !important;
}

footer,
.footer {
    display: none !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 0 18px 18px !important;
}

/* Header */
.pf-header {
    text-align: center;
    padding: 34px 18px 28px;
}

.pf-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    font-weight: 700;
    color: var(--accent);
    background: white;
    border: 1px solid #f0d0c0;
    padding: 6px 14px;
    border-radius: 999px;
    margin-bottom: 18px;
}

.pf-header h1 {
    font-size: clamp(34px, 5vw, 56px);
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.06em;
    line-height: 0.98;
    margin: 0 0 14px;
}

.pf-header h1 span {
    color: var(--accent);
}

.pf-header p {
    font-size: 15px;
    color: var(--muted);
    line-height: 1.7;
    max-width: 620px;
    margin: 0 auto;
}

/* Cards */
.panel-card,
.bottom-card {
    background: var(--panel) !important;
    border-radius: 26px !important;
    border: 1px solid rgba(255,255,255,0.7) !important;
    box-shadow:
        0 18px 48px rgba(40, 34, 28, 0.08),
        0 2px 8px rgba(40, 34, 28, 0.04) !important;
    padding: 24px !important;
}

.bottom-card {
    border-radius: 22px !important;
    padding: 22px !important;
}

/* Section headings */
.section-title {
    margin-bottom: 14px;
}

.section-title h2 {
    font-size: 17px;
    line-height: 1.25;
    margin: 0 0 5px;
    letter-spacing: -0.02em;
}

.section-title p {
    margin: 0;
    color: var(--muted);
    font-size: 12.5px;
    line-height: 1.5;
}

/* Labels */
label > span,
.form > label > span {
    font-size: 10.5px !important;
    font-weight: 700 !important;
    color: #9d9991 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Textareas */
textarea {
    font-size: 14px !important;
    color: #2a2a28 !important;
    background: white !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    line-height: 1.75 !important;
    padding: 15px 16px !important;
    resize: vertical !important;
    box-shadow: none !important;
}

textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(196,99,62,0.09) !important;
    outline: none !important;
}

textarea::placeholder {
    color: #bbb7af !important;
}

.output-text textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.85 !important;
    background: #fffdfa !important;
}

.exemplar-text textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    color: #4b4945 !important;
    background: #fbfaf7 !important;
    border-color: #f0eee8 !important;
    min-height: 150px !important;
}

/* Status + meta */
.status-text textarea {
    font-size: 12.5px !important;
    color: var(--green) !important;
    background: #f7fbf5 !important;
    border-color: #e2eedc !important;
    min-height: 45px !important;
    resize: none !important;
}

.copy-status textarea,
.meta-text textarea {
    font-size: 11.5px !important;
    color: #9c978e !important;
    background: transparent !important;
    border: none !important;
    min-height: 26px !important;
    padding: 0 !important;
    box-shadow: none !important;
    resize: none !important;
}

/* Controls */
.gr-dropdown,
select {
    background: #f8f7f4 !important;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    font-size: 13px !important;
    color: #333 !important;
    font-weight: 600 !important;
}

button.primary,
.primary {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 13px 22px !important;
    transition: transform 0.15s, background 0.15s !important;
    box-shadow: 0 10px 24px rgba(196, 99, 62, 0.22) !important;
}

button.primary:hover {
    background: var(--accent-dark) !important;
    transform: translateY(-1px);
}

button.secondary,
.secondary {
    background: #f8f7f4 !important;
    color: #4f4b45 !important;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 12px 20px !important;
    box-shadow: none !important;
}

button.secondary:hover {
    background: #f0eee8 !important;
}

/* Remove noisy Gradio chrome */
.form,
.block,
.gap,
.padded,
.gr-group,
.gr-box {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gr-accordion {
    background: #fbfaf7 !important;
    border-radius: 18px !important;
    border: 1px solid var(--line) !important;
    box-shadow: none !important;
}

/* Score card */
.score-card {
    background: #fbfaf7;
    border: 1px solid #f0eee8;
    border-radius: 18px;
    padding: 16px;
}

.score-hero {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: center;
    margin-bottom: 14px;
}

.eyebrow {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9d9991;
    font-weight: 800;
}

.score-hero h3 {
    margin: 2px 0 0;
    font-size: 30px;
    letter-spacing: -0.05em;
    line-height: 1;
}

.grade-pill {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid #f0d0c0;
    border-radius: 999px;
    padding: 7px 11px;
    font-size: 12px;
    font-weight: 800;
    white-space: nowrap;
}

.score-row {
    padding: 10px 0;
    border-top: 1px solid #efebe3;
}

.score-row-top {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    color: #45423c;
    margin-bottom: 7px;
}

.score-row-top strong {
    color: #1f1f1d;
}

.score-track {
    height: 7px;
    background: #ece8df;
    border-radius: 999px;
    overflow: hidden;
}

.score-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 999px;
}

.score-row p {
    margin: 7px 0 0;
    color: #8d887f;
    font-size: 11.5px;
    line-height: 1.45;
}

/* Facts */
.facts-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
}

.fact-card {
    background: white;
    border: 1px solid #f0eee8;
    border-radius: 16px;
    padding: 13px 14px;
}

.fact-card span {
    display: block;
    color: #9d9991;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.fact-card strong {
    color: #272420;
    font-size: 13px;
    line-height: 1.3;
}

.usage-line {
    margin-top: 10px;
    color: #918c84;
    font-size: 11.5px;
}

.empty-state {
    background: #fbfaf7;
    border: 1px dashed #e5ded4;
    color: #9a948b;
    border-radius: 16px;
    padding: 18px;
    font-size: 13px;
}

/* Examples */
.examples-wrap {
    margin-top: 14px;
}

/* Mobile */
@media (max-width: 800px) {
    .gradio-container {
        padding: 0 12px 14px !important;
    }

    .panel-card,
    .bottom-card {
        padding: 18px !important;
        border-radius: 22px !important;
    }

    .facts-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .pf-header {
        padding-top: 28px;
    }
}

@media (max-width: 520px) {
    .facts-grid {
        grid-template-columns: 1fr;
    }
}
"""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Prompt Forge RAG", css=CSS) as demo:
    gr.Markdown(
        """
        <div class="pf-header">
            <div class="pf-badge">⚡ Prompt Forge RAG · model-aware prompt optimizer</div>
            <h1>Turn messy prompts into <span>structured instructions</span></h1>
            <p>
                Transform rough ideas into model-specific prompts using retrieval,
                exemplar matching, LLM rewriting, and a built-in quality scorer.
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown(
                """
                <div class="section-title">
                    <h2>1. Draft input</h2>
                    <p>Paste a messy prompt, rough notes, or a half-formed request.</p>
                </div>
                """
            )

            raw_input = gr.Textbox(
                label="Your rough prompt",
                placeholder=(
                    "Example: review this python script and make the prompt better "
                    "for claude code, include exact output format and constraints..."
                ),
                lines=15,
                max_lines=24,
            )

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=MODELS,
                    value="general",
                    label="Target model",
                )
                depth_dropdown = gr.Dropdown(
                    choices=DEPTHS,
                    value="standard",
                    label="Depth",
                )

            with gr.Accordion("Advanced retrieval settings", open=False):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=6,
                    value=3,
                    step=1,
                    label="Retrieved exemplars",
                    info="How many similar prompts to retrieve from LanceDB.",
                )

            forge_btn = gr.Button(
                "⚡ Transform prompt",
                variant="primary",
                size="lg",
            )

            with gr.Accordion("Try an example", open=False, elem_classes="examples-wrap"):
                gr.Examples(
                    examples=[
                        [
                            "Review this Python code for security issues and tell me what to fix.",
                            "claude-code",
                            "standard",
                            3,
                        ],
                        [
                            "Make an agent that monitors a repo and checks for bugs.",
                            "general",
                            "comprehensive",
                            3,
                        ],
                        [
                            "Fix the bug in my login function. It crashes on missing email.",
                            "cursor",
                            "standard",
                            2,
                        ],
                        [
                            "Summarize this meeting transcript for my product team.",
                            "gpt-4",
                            "concise",
                            3,
                        ],
                    ],
                    inputs=[raw_input, model_dropdown, depth_dropdown, top_k_slider],
                )

        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown(
                """
                <div class="section-title">
                    <h2>2. Structured output</h2>
                    <p>The transformed prompt is editable, copyable, and downloadable.</p>
                </div>
                """
            )

            status_display = gr.Textbox(
                label="Status",
                lines=1,
                interactive=False,
                elem_classes="status-text",
            )

            output_display = gr.Textbox(
                label="Structured prompt",
                lines=15,
                max_lines=26,
                interactive=True,
                elem_classes="output-text",
            )

            output_meta = gr.Textbox(
                label="",
                lines=1,
                interactive=False,
                elem_classes="meta-text",
            )

            with gr.Row():
                copy_btn = gr.Button(
                    "Copy",
                    variant="secondary",
                    size="lg",
                )
                download_btn = gr.Button(
                    "Download .txt",
                    variant="secondary",
                    size="lg",
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary",
                    size="lg",
                )

            copy_status = gr.Textbox(
                label="",
                lines=1,
                interactive=False,
                elem_classes="copy-status",
            )

            download_file = gr.File(
                label="Download",
                visible=False,
                interactive=False,
            )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes="bottom-card"):
            gr.Markdown(
                """
                <div class="section-title">
                    <h2>3. Quality score</h2>
                    <p>Checks structure, specificity, model fit, task coverage, and improvement.</p>
                </div>
                """
            )
            score_display = gr.HTML(
                value="<div class='empty-state'>Run a transformation to see the score breakdown.</div>"
            )

        with gr.Column(scale=1, elem_classes="bottom-card"):
            gr.Markdown(
                """
                <div class="section-title">
                    <h2>4. Retrieval trace</h2>
                    <p>Shows the prompt examples used as style and structure signals.</p>
                </div>
                """
            )
            exemplar_display = gr.Textbox(
                label="Matched exemplars",
                lines=8,
                interactive=False,
                elem_classes="exemplar-text",
            )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes="bottom-card"):
            gr.Markdown(
                """
                <div class="section-title">
                    <h2>Run details</h2>
                    <p>Useful for understanding what the pipeline detected and which provider answered.</p>
                </div>
                """
            )
            facts_display = gr.HTML(
                value="<div class='empty-state'>Detected task, provider, and usage will appear here.</div>"
            )

    gr.Markdown(
        """
        <div style="text-align:center;padding:18px 0 6px;">
            <span style="font-size:11.5px;color:#9d9991;font-family:'DM Sans',sans-serif;">
                BGE-small · LanceDB · Groq llama-3.3-70b · Cerebras fallback · Gradio
            </span>
        </div>
        """
    )

    # Copy to clipboard
    copy_btn.click(
        fn=None,
        inputs=[output_display],
        outputs=[copy_status],
        js="""
        async (text) => {
            if (!text || !text.trim()) {
                return "Nothing to copy yet.";
            }
            await navigator.clipboard.writeText(text);
            return "Copied to clipboard.";
        }
        """,
    )

    # Download prompt
    download_btn.click(
        fn=write_prompt_file,
        inputs=[output_display],
        outputs=[download_file],
    )

    # Clear workspace
    clear_btn.click(
        fn=lambda: (
            "",
            "",
            "",
            "",
            "<div class='empty-state'>Run a transformation to see the score breakdown.</div>",
            "<div class='empty-state'>Detected task, provider, and usage will appear here.</div>",
            "",
            None,
        ),
        inputs=[],
        outputs=[
            raw_input,
            output_display,
            status_display,
            exemplar_display,
            score_display,
            facts_display,
            output_meta,
            download_file,
        ],
    )

    # Forge
    forge_btn.click(
        fn=forge,
        inputs=[raw_input, model_dropdown, depth_dropdown, top_k_slider],
        outputs=[
            output_display,
            status_display,
            exemplar_display,
            score_display,
            facts_display,
            output_meta,
        ],
    )


if __name__ == "__main__":
    demo.launch()
