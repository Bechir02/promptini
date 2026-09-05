from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from rag import run_pipeline, run_pipeline_stream
from ingest import build_index
from pathlib import Path
from scorer import score_transformation, format_score_html
from core.constants import MODELS, DEPTHS
from core.concurrency import map_ordered
from core.ratelimit import RateLimiter
from core.config import get_settings

# Fail fast with a clear message if no provider key is configured.
get_settings().require_provider()

# ── Build index on startup if missing ─────────────────────────────────────────
print("Checking index...")
if not Path("lancedb_store").exists():
    print("Index not found — building on startup...")
    build_index(force=True)
else:
    print("Index found — skipping rebuild.")

# ── Status messages ───────────────────────────────────────────────────────────
EMPTY_PROMPT_MSG = "Paste a rough prompt above, pick a model, and hit Forge."
NO_MODEL_MSG = "Select at least one target model to forge for."
RATE_LIMIT_MSG = "You're forging quickly — give it a few seconds and try again."

_settings = get_settings()
_LIMITER = RateLimiter(_settings.rate_limit_calls, _settings.rate_limit_window)


def _empty_return(message: str):
    """Return the 11-tuple for a no-op (empty input) with a helpful status."""
    return (
        "", message, "",
        "", "", "",
        gr.update(visible=False), "",
        "", "", "",
    )


def _begin(raw_prompt: str, target_models: list[str]):
    """Immediate visual feedback before the (slow) LLM calls run."""
    if not raw_prompt or not raw_prompt.strip() or not target_models:
        return gr.update(), gr.update(), gr.update(), gr.update(visible=False)
    two = len(target_models) > 1
    return (
        "Forging Version A…",
        ("Forging Version B…" if two else ""),
        gr.update(visible=two),      # col_2
        gr.update(visible=False),    # arena_note_row
    )


def _fmt_exemplars(exemplars) -> str:
    if not exemplars:
        return "No exemplars found for this query."
    return "\n\n".join(
        f"EXAMPLE {i+1}  [{ex['target_model']}]\n{ex['prompt'][:200]}…"
        for i, ex in enumerate(exemplars)
    )


def _run_one(raw_prompt: str, model_name: str, depth: str, language: str, chain: bool) -> dict:
    res = run_pipeline(raw_prompt=raw_prompt, target_model=model_name, depth=depth, language=language, chain=chain)
    score_res = score_transformation(raw_prompt, res["transformed"], model_name, res["task_type"])
    return {"res": res, "score": score_res, "model": model_name}


def _arena_outputs(results, raw_prompt, selected_models):
    out1 = results[0]["res"]["transformed"]
    status1 = f"✓ {results[0]['model']} · {results[0]['score']['overall']}/10 · {results[0]['res']['task_type']}"
    score1 = format_score_html(results[0]["score"], results[0]["score"]["breakdown"]["llm_judge"]["note"])
    meta1 = f"{len(out1)} chars · {len(out1.split())} words"

    out2 = results[1]["res"]["transformed"]
    status2 = f"✓ {results[1]['model']} · {results[1]['score']['overall']}/10 · {results[1]['res']['task_type']}"
    score2 = format_score_html(results[1]["score"], results[1]["score"]["breakdown"]["llm_judge"]["note"])
    meta2 = f"{len(out2)} chars · {len(out2.split())} words"

    from scorer import judge_battle
    battle = judge_battle(raw_prompt, out1, selected_models[0], out2, selected_models[1])
    winner_model = selected_models[0] if battle["winner"] == "A" else selected_models[1]
    battle_note = f"### 🏆 Winner: Version {battle['winner']} — {winner_model}\n\n{battle['reasoning']}"

    return (
        out1, status1, score1,
        out2, status2, score2,
        gr.update(visible=True), battle_note,
        meta1, meta2, _fmt_exemplars(results[0]["res"].get("exemplars")),
    )


def forge(raw_prompt: str, target_models: list[str], depth: str, language: str = "english", chain: bool = False, request: gr.Request | None = None):
    if not raw_prompt or not raw_prompt.strip():
        yield _empty_return(EMPTY_PROMPT_MSG)
        return
    if not target_models:
        yield _empty_return(NO_MODEL_MSG)
        return

    client_id = request.client.host if (request and request.client) else "anon"
    if not _LIMITER.allow(client_id):
        yield _empty_return(RATE_LIMIT_MSG)
        return

    selected_models = target_models[:2]

    # Arena (two models): run in parallel and emit once.
    if len(selected_models) > 1:
        results = map_ordered(
            lambda m: _run_one(raw_prompt, m, depth, language, chain),
            selected_models, max_workers=2,
        )
        yield _arena_outputs(results, raw_prompt, selected_models)
        return

    # Single model: stream token-by-token (B8).
    model = selected_models[0]
    acc, task_type, exemplars = "", "", []
    for part in run_pipeline_stream(raw_prompt, model, depth, language, chain):
        acc = part["transformed"]
        task_type = part["task_type"]
        exemplars = part.get("exemplars") or []
        if part.get("error"):
            yield _empty_return(f"⚠️ {part['error']}")
            return
        if part.get("done"):
            break
        yield (
            acc, f"⚡ streaming · {model}…", "",
            "", "", "",
            gr.update(visible=False), "",
            f"{len(acc)} chars", "", _fmt_exemplars(exemplars),
        )

    # Final: score the completed output.
    score_res = score_transformation(raw_prompt, acc, model, task_type)
    status = f"✓ {model} · {score_res['overall']}/10 · {task_type}"
    score_html = format_score_html(score_res, score_res["breakdown"]["llm_judge"]["note"])
    meta = f"{len(acc)} chars · {len(acc.split())} words"
    yield (
        acc, status, score_html,
        "", "", "",
        gr.update(visible=False), "",
        meta, "", _fmt_exemplars(exemplars),
    )


# ── CSS — "paper & clay" design system ────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --pf-bg:          #f4f2ee;
  --pf-surface:     #ffffff;
  --pf-surface-2:   #faf9f7;
  --pf-ink:         #1c1b19;
  --pf-ink-soft:    #605c55;
  --pf-ink-faint:   #9a958c;
  --pf-accent:      #c15f3c;
  --pf-accent-hi:   #a94e2e;
  --pf-accent-soft: rgba(193,95,60,0.08);
  --pf-accent-line: rgba(193,95,60,0.22);
  --pf-border:      #e7e4de;
  --pf-border-soft: #efece7;
  --pf-good:        #5f8a4c;
  --pf-warn:        #c58a2c;
  --pf-bad:         #c0503f;
  --pf-radius:      14px;
  --pf-radius-lg:   20px;
  --pf-shadow:      0 1px 2px rgba(28,27,25,0.04), 0 10px 30px rgba(28,27,25,0.05);
}

/* ── Base ── */
* { box-sizing: border-box; }
body, .gradio-container, .main, .wrap, .app {
  background: var(--pf-bg) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  color: var(--pf-ink) !important;
}
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer, .footer, .hide-footer { display: none !important; }

/* ── Header ── */
.pf-header { text-align: center; padding: 40px 24px 8px; }
.pf-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11.5px; font-weight: 600; letter-spacing: 0.01em;
  color: var(--pf-accent); background: var(--pf-surface);
  border: 1px solid var(--pf-accent-line);
  padding: 5px 13px; border-radius: 100px; margin-bottom: 18px;
}
.pf-title {
  font-size: 34px; font-weight: 700; color: var(--pf-ink);
  letter-spacing: -0.035em; line-height: 1.15; margin: 0 0 10px;
}
.pf-title .accent { color: var(--pf-accent); }
.pf-title-ar { font-size: 22px; font-weight: 600; color: var(--pf-ink-soft); margin-inline-start: 8px; vertical-align: 2px; }
.pf-sub {
  font-size: 14.5px; color: var(--pf-ink-soft); line-height: 1.6;
  max-width: 480px; margin: 0 auto;
}

/* ── Cards ── */
.panel-card {
  background: var(--pf-surface) !important;
  border: 1px solid var(--pf-border) !important;
  border-radius: var(--pf-radius-lg) !important;
  box-shadow: var(--pf-shadow) !important;
  padding: 22px !important;
}
.bottom-card {
  background: var(--pf-surface) !important;
  border: 1px solid var(--pf-border) !important;
  border-radius: var(--pf-radius) !important;
  box-shadow: var(--pf-shadow) !important;
  padding: 18px 20px !important;
}

/* ── Labels ── */
label > span, .form > label > span {
  font-size: 11px !important; font-weight: 600 !important;
  color: var(--pf-ink-faint) !important;
  text-transform: uppercase !important; letter-spacing: 0.07em !important;
}

/* ── Textareas ── */
textarea {
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important; color: var(--pf-ink) !important;
  background: var(--pf-surface) !important;
  border: 1px solid var(--pf-border) !important;
  border-radius: 12px !important; line-height: 1.7 !important;
  padding: 13px 15px !important; resize: none !important;
  transition: border-color .15s, box-shadow .15s !important;
}
textarea:focus {
  border-color: var(--pf-accent) !important;
  box-shadow: 0 0 0 3px var(--pf-accent-soft) !important; outline: none !important;
}
textarea::placeholder { color: var(--pf-ink-faint) !important; }

.output-text textarea {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12.5px !important; color: #2b2a27 !important;
  line-height: 1.8 !important; background: var(--pf-surface-2) !important;
  min-height: 340px !important;
}
.exemplar-text textarea {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important; color: var(--pf-ink-soft) !important;
  background: var(--pf-surface-2) !important; border-color: var(--pf-border-soft) !important;
  line-height: 1.75 !important; min-height: 130px !important;
}

/* ── Status + meta ── */
.status-text textarea {
  font-size: 12px !important; font-weight: 500 !important;
  color: var(--pf-good) !important; background: rgba(95,138,76,0.07) !important;
  border-color: rgba(95,138,76,0.18) !important; min-height: 40px !important;
  font-family: 'Inter', sans-serif !important;
}
.meta-text textarea {
  font-size: 11.5px !important; color: var(--pf-ink-faint) !important;
  background: transparent !important; border: none !important;
  min-height: 20px !important; padding: 2px 2px !important; box-shadow: none !important;
}

/* ── Dropdowns ── */
.gr-dropdown, .wrap-inner, .secondary-wrap {
  border-radius: 12px !important;
}
.gr-dropdown {
  background: var(--pf-surface-2) !important;
  border: 1px solid var(--pf-border) !important;
  font-size: 13px !important; color: var(--pf-ink) !important;
  font-weight: 500 !important; padding: 8px 12px !important;
}

/* ── Buttons ── */
button.primary, .primary {
  background: var(--pf-accent) !important; color: #fff !important;
  border: none !important; border-radius: 12px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13.5px !important; font-weight: 600 !important;
  padding: 12px 22px !important; box-shadow: none !important;
  transition: background .15s, transform .12s !important;
}
button.primary:hover { background: var(--pf-accent-hi) !important; transform: translateY(-1px); }
button.primary:active { transform: translateY(0); }

button.secondary, .secondary {
  background: var(--pf-surface) !important; color: var(--pf-ink-soft) !important;
  border: 1px solid var(--pf-border) !important; border-radius: 10px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 12.5px !important; font-weight: 500 !important;
  padding: 9px 16px !important; box-shadow: none !important;
  transition: background .15s, border-color .15s !important;
}
button.secondary:hover { background: var(--pf-surface-2) !important; border-color: var(--pf-ink-faint) !important; }

/* ── Strip default gradio chrome ── */
.form, .block, .gap, .padded, .gr-group, .gr-box {
  background: transparent !important; border: none !important; box-shadow: none !important;
}

/* ── Accordion ── */
.gr-accordion {
  background: var(--pf-surface-2) !important; border: 1px solid var(--pf-border-soft) !important;
  border-radius: 12px !important; box-shadow: none !important;
}

/* ── Battle note ── */
.bottom-card h3 { margin: 0 0 8px; font-size: 15px; font-weight: 700; color: var(--pf-ink); letter-spacing: -0.01em; }
.bottom-card p  { margin: 0; font-size: 13.5px; color: var(--pf-ink-soft); line-height: 1.6; }

/* ── Score card ── */
.pf-score-card {
  background: var(--pf-surface-2); border: 1px solid var(--pf-border-soft);
  border-radius: 12px; padding: 14px 15px; margin-top: 4px;
}
.pf-score-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.pf-grade-chip {
  display: inline-block; color: #fff; font-size: 11.5px; font-weight: 600;
  padding: 4px 11px; border-radius: 100px; letter-spacing: 0.01em;
}
.pf-score-overall { font-size: 22px; font-weight: 700; color: var(--pf-ink); letter-spacing: -0.02em; }
.pf-score-overall-max { font-size: 13px; font-weight: 500; color: var(--pf-ink-faint); }
.pf-score-bars { display: flex; flex-direction: column; gap: 7px; }
.pf-score-row { display: flex; align-items: center; gap: 10px; }
.pf-score-label { flex: 0 0 96px; font-size: 11px; font-weight: 500; color: var(--pf-ink-soft); }
.pf-score-track { flex: 1; height: 6px; background: var(--pf-border); border-radius: 100px; overflow: hidden; }
.pf-score-fill  { display: block; height: 100%; border-radius: 100px; transition: width .5s ease; }
.pf-score-val   { flex: 0 0 30px; text-align: right; font-size: 11.5px; font-weight: 600; color: var(--pf-ink); font-family: 'JetBrains Mono', monospace; }
.pf-score-reason {
  margin-top: 12px; padding-top: 11px; border-top: 1px solid var(--pf-border-soft);
  font-size: 12px; color: var(--pf-ink-soft); line-height: 1.55;
}
.pf-score-hints { margin-top: 10px; }
.pf-hint-label { display: block; font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pf-accent); margin-bottom: 4px; }
.pf-score-hints ul { margin: 0; padding-left: 16px; }
.pf-score-hints li { font-size: 11.5px; color: var(--pf-ink-soft); line-height: 1.5; }

/* ── Toasts ── */
#pf-toast-root {
  position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%);
  z-index: 9999; display: flex; flex-direction: column; gap: 8px; align-items: center;
  pointer-events: none;
}
.pf-toast {
  background: var(--pf-ink); color: #fff; font-family: 'Inter', sans-serif;
  font-size: 13px; font-weight: 500; padding: 10px 18px; border-radius: 100px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.18);
  opacity: 0; transform: translateY(8px); transition: opacity .25s, transform .25s;
}
.pf-toast.pf-toast-in { opacity: 1; transform: translateY(0); }

/* ── Library cards ── */
#library-container .lib-empty { color: var(--pf-ink-faint); font-size: 13px; }

@media (max-width: 768px) {
  .pf-header { padding: 30px 16px 6px; }
  .pf-title { font-size: 27px; }
  .panel-card, .bottom-card { padding: 16px !important; }
  .output-text textarea { min-height: 260px !important; }
}
"""


# ── Voice dictation (free, browser-native Web Speech API) ─────────────────────
MIC_JS = """
(lang) => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { if (window.pfToast) window.pfToast('Dictation needs Chrome or Edge'); return; }
    try {
        const rec = new SR();
        rec.lang = lang || 'ar-TN';
        rec.interimResults = true;
        rec.continuous = false;
        if (window.pfToast) window.pfToast('Listening… speak now');
        let finalText = '';
        rec.onresult = (e) => {
            let interim = '';
            for (let i = e.resultIndex; i < e.results.length; i++) {
                const t = e.results[i][0].transcript;
                if (e.results[i].isFinal) finalText += t; else interim += t;
            }
            const areas = document.querySelectorAll('textarea[data-testid="textbox"]');
            if (areas.length) {
                areas[0].value = (finalText + interim).trim();
                areas[0].dispatchEvent(new Event('input', { bubbles: true }));
            }
        };
        rec.onerror = (ev) => { if (window.pfToast) window.pfToast('Mic: ' + ev.error); };
        rec.onend = () => { if (window.pfToast) window.pfToast('Captured'); };
        rec.start();
    } catch (err) { if (window.pfToast) window.pfToast('Mic unavailable'); }
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="M3allem · المْعلّم") as demo:

    # Header
    gr.HTML(f"""
<div class="pf-header">
  <div class="pf-badge">⚡ المْعلّم · Derja-first · {len(MODELS)} models · 8,435 exemplars</div>
  <h1 class="pf-title">M<span class="accent">3</span>allem <span class="pf-title-ar">المْعلّم</span></h1>
  <p class="pf-sub">خلّي البرومبت متاعك m3allem — speak or type a rough idea in Derja, Arabic, French or English, and get a precise, model-optimized prompt.</p>
</div>
""")

    # Input Section
    with gr.Row(elem_classes="panel-card"):
        with gr.Column(scale=2):
            raw_input = gr.Textbox(
                label       = "Messy prompt",
                placeholder = "Paste — or dictate 🎙 — your rough draft here…",
                lines       = 6,
            )
            with gr.Row():
                mic_lang = gr.Dropdown(
                    choices = [("Derja · تونسي", "ar-TN"), ("Français", "fr-FR"), ("English", "en-US")],
                    value   = "ar-TN",
                    label   = "Dictation language",
                    scale   = 2,
                )
                mic_btn = gr.Button("🎙 Dictate", scale=1, min_width=120)
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices     = MODELS,
                value       = ["general"],
                multiselect = True,
                max_choices = 2,
                label       = "Models (select up to 2)",
            )
            depth_dropdown = gr.Dropdown(
                choices = DEPTHS,
                value   = "standard",
                label   = "Depth",
            )
            language_dropdown = gr.Dropdown(
                choices = [
                    ("Auto — match input", "auto"),
                    ("English", "english"),
                    ("العربية · Arabic", "arabic"),
                    ("Derja · تونسي", "derja"),
                    ("Français", "french"),
                ],
                value   = "english",
                label   = "Output language",
            )
            chain_toggle = gr.Checkbox(value=False, label="Decompose into steps (chain)")
            forge_btn = gr.Button("⚡ Forge & Battle", variant="primary")

    # Battle Result Note
    with gr.Row(visible=False) as arena_note_row:
        battle_note = gr.Markdown(elem_classes="bottom-card")

    # Main Arena Row
    with gr.Row(equal_height=False):

        # Column 1
        with gr.Column(scale=1, elem_classes="panel-card"):
            status_1 = gr.Textbox(label="Version A", lines=1, interactive=False, elem_classes="status-text")
            output_1 = gr.Textbox(label="Structured prompt (A)", lines=16, interactive=True, elem_classes="output-text")
            meta_1 = gr.Textbox(label="", lines=1, interactive=False, elem_classes="meta-text")
            score_1 = gr.HTML(elem_classes="score-card-wrap")
            with gr.Row():
                copy_btn_1 = gr.Button("Copy A", size="sm")
                save_btn_1 = gr.Button("⭐ Save A", size="sm")

        # Column 2 (Arena)
        with gr.Column(scale=1, elem_classes="panel-card", visible=False) as col_2:
            status_2 = gr.Textbox(label="Version B", lines=1, interactive=False, elem_classes="status-text")
            output_2 = gr.Textbox(label="Structured prompt (B)", lines=16, interactive=True, elem_classes="output-text")
            meta_2 = gr.Textbox(label="", lines=1, interactive=False, elem_classes="meta-text")
            score_2 = gr.HTML(elem_classes="score-card-wrap")
            with gr.Row():
                copy_btn_2 = gr.Button("Copy B", size="sm")
                save_btn_2 = gr.Button("⭐ Save B", size="sm")

    # Library & Exemplars
    with gr.Row():
        with gr.Column(elem_classes="bottom-card"):
            with gr.Accordion("📚 Your prompt library", open=False):
                library_list = gr.HTML(
                    "<div id='library-container'><p class='lib-empty'>No saved prompts yet. "
                    "Use the VS Code sidebar to save prompts.</p></div>"
                )
        with gr.Column(elem_classes="bottom-card"):
            exemplar_display = gr.Textbox(
                label       = "Source exemplars",
                lines       = 5,
                interactive = False,
                elem_classes= "exemplar-text",
            )

    # ── Event handlers ────────────────────────────────────────────────────────
    forge_btn.click(
        fn=_begin,
        inputs=[raw_input, model_dropdown],
        outputs=[status_1, status_2, col_2, arena_note_row],
    ).then(
        fn=forge,
        inputs=[raw_input, model_dropdown, depth_dropdown, language_dropdown, chain_toggle],
        outputs=[
            output_1, status_1, score_1,
            output_2, status_2, score_2,
            col_2, battle_note,
            meta_1, meta_2, exemplar_display,
        ],
    )

    mic_btn.click(fn=None, inputs=[mic_lang], js=MIC_JS)

    battle_note.change(lambda x: gr.update(visible=bool(x)), inputs=battle_note, outputs=arena_note_row)

    # Copy: native clipboard on the standalone page + postMessage bridge for VS Code.
    copy_js = """
    (v) => {
        if (v && navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(v)
                .then(() => { if (window.pfToast) window.pfToast('📋 Copied to clipboard'); })
                .catch(() => {});
        }
        try { window.top.postMessage({ type: 'copyText', text: v }, '*'); } catch (e) {}
    }
    """
    copy_btn_1.click(fn=None, inputs=output_1, js=copy_js)
    copy_btn_2.click(fn=None, inputs=output_2, js=copy_js)

    def get_save_js():
        return """
        (prompt, model, status) => {
            const targetModel = Array.isArray(model) ? model[0] : model;
            const entry = { prompt: prompt, model: targetModel, status: status };
            try { window.top.postMessage({ type: 'savePrompt', entry: entry }, '*'); } catch (e) {}
            if (window.pfToast) window.pfToast('⭐ Saved to library');
        }
        """
    save_btn_1.click(fn=None, inputs=[output_1, model_dropdown, status_1], js=get_save_js())
    save_btn_2.click(fn=None, inputs=[output_2, model_dropdown, status_2], js=get_save_js())

    # Toast helper + message bridge with the VS Code extension.
    demo.load(None, None, None, js="""
    () => {
        // Toast helper (replaces the old activeElement button-text hack).
        window.pfToast = (msg) => {
            let root = document.getElementById('pf-toast-root');
            if (!root) { root = document.createElement('div'); root.id = 'pf-toast-root'; document.body.appendChild(root); }
            const t = document.createElement('div');
            t.className = 'pf-toast';
            t.textContent = msg;
            root.appendChild(t);
            requestAnimationFrame(() => t.classList.add('pf-toast-in'));
            setTimeout(() => { t.classList.remove('pf-toast-in'); setTimeout(() => t.remove(), 300); }, 2200);
        };

        const setInput = (text) => {
            const areas = document.querySelectorAll('textarea[data-testid="textbox"]');
            if (areas.length > 0) {
                areas[0].value = text;
                areas[0].dispatchEvent(new Event('input', { bubbles: true }));
            }
        };

        window.addEventListener('message', (event) => {
            const data = event.data;
            if (!data || !data.type) return;

            if (data.type === 'setPrompt') {
                setInput(data.text);
            } else if (data.type === 'syncLibrary') {
                const container = document.getElementById('library-container');
                if (!container) return;
                const lib = data.library || [];
                if (lib.length === 0) {
                    container.innerHTML = "<p class='lib-empty'>No saved prompts yet.</p>";
                    return;
                }
                let html = "<div style='display:grid;gap:10px;'>";
                lib.forEach(item => {
                    const date = new Date(item.date).toLocaleDateString();
                    html += `
                        <div style="background:#faf9f7;padding:12px;border-radius:12px;border:1px solid #efece7;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                                <span style="font-size:11px;font-weight:600;color:#c15f3c;text-transform:uppercase;letter-spacing:.04em;">${item.model}</span>
                                <span style="font-size:11px;color:#9a958c;">${date}</span>
                            </div>
                            <div style="font-size:13px;color:#605c55;margin-bottom:10px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.5;">${item.prompt}</div>
                            <button onclick="window.loadFavorite('${item.id}')" style="background:#fff;border:1px solid #e7e4de;border-radius:8px;padding:5px 11px;font-size:11px;cursor:pointer;font-weight:600;color:#605c55;">Load prompt</button>
                        </div>`;
                });
                html += "</div>";
                container.innerHTML = html;

                window.loadFavorite = (id) => {
                    const entry = lib.find(i => i.id === id);
                    if (entry) setInput(entry.prompt);
                };
            }
        });

        window.top.postMessage({ type: 'ready' }, '*');
    }
    """)

if __name__ == "__main__":
    demo.queue().launch(css=CSS, ssr_mode=False)
