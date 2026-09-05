from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from llm import transform_prompt
from scorer import pick_best
from core.tasks import detect_task_type
from core.concurrency import map_ordered
from core.config import get_settings
from core.ratelimit import RateLimiter

BEST_OF = 3   # generate N candidates; an internal judge keeps only the best

_settings = get_settings()
_LIMITER = RateLimiter(_settings.rate_limit_calls, _settings.rate_limit_window)

EMPTY_PROMPT_MSG = "Paste or dictate a rough idea, then hit Forge."
RATE_LIMIT_MSG   = "You're forging quickly — give it a few seconds and try again."

# ── Design (clean monochrome + one blue accent) ───────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Noto+Naskh+Arabic:wght@700&display=swap');
:root{
  --bg:#f7f7f8; --card:#ffffff; --ink:#0a0a0a; --soft:#6b7280; --faint:#9ca3af;
  --line:#eaeaec; --accent:#2563eb; --accent-soft:rgba(37,99,235,.10); --good:#16a34a;
  --radius:14px; --shadow:0 1px 3px rgba(0,0,0,.04),0 10px 34px rgba(0,0,0,.06);
}
*{box-sizing:border-box;}
body,.gradio-container,.app,.main,.wrap{background:var(--bg)!important;color:var(--ink)!important;
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;}
.gradio-container{max-width:100%!important;margin:0 auto!important;padding:8px 40px 46px!important;}
footer,.footer{display:none!important;}
.gradio-container .block,.gradio-container .form{border:none!important;background:transparent!important;box-shadow:none!important;}

.pf-head{text-align:center;padding:26px 10px 22px;}
.pf-logo{display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;border-radius:13px;
  background:var(--ink);color:#fff;font-size:23px;margin-bottom:12px;box-shadow:0 8px 20px rgba(0,0,0,.18);}
.pf-brand{font-size:40px;font-weight:800;letter-spacing:-.045em;color:var(--ink);}
.pf-brand span{color:var(--accent);}
.pf-tag{margin-top:8px;font-size:14px;color:var(--soft);line-height:1.6;}

.pf-panel{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:18px!important;
  box-shadow:var(--shadow)!important;padding:22px!important;}

textarea,input[type=text]{border:1px solid var(--line)!important;border-radius:12px!important;background:#fcfcfd!important;
  color:var(--ink)!important;font-size:14.5px!important;line-height:1.6!important;}
textarea:focus,input:focus{outline:none!important;border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-soft)!important;}
label span{color:var(--soft)!important;font-size:12px!important;font-weight:600!important;letter-spacing:.01em;}
.pf-ctl input{background:#fcfcfd!important;border:1px solid var(--line)!important;border-radius:11px!important;font-weight:500!important;}
.pf-ctl input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-soft)!important;}

.pf-forge button{background:var(--ink)!important;color:#fff!important;border:none!important;border-radius:12px!important;
  font-size:15px!important;font-weight:700!important;padding:14px!important;width:100%!important;
  box-shadow:0 8px 20px rgba(0,0,0,.16)!important;transition:transform .06s ease,opacity .2s;}
.pf-forge button:hover{opacity:.90;} .pf-forge button:active{transform:translateY(1px);}

.pf-mic-native{width:100%;background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:11px;
  font-weight:600;font-size:13.5px;font-family:'Inter',sans-serif;padding:11px;cursor:pointer;transition:.15s;
  display:flex;align-items:center;justify-content:center;gap:7px;}
.pf-mic-native:hover{border-color:var(--accent);color:var(--accent);}
.pf-mic-native.rec{border-color:#dc2626;color:#dc2626;animation:pfpulse 1s infinite;}
@keyframes pfpulse{0%,100%{opacity:1}50%{opacity:.55}}

.pf-copy button{background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:11px!important;
  font-weight:700!important;font-size:13.5px!important;width:100%!important;box-shadow:0 6px 16px rgba(37,99,235,.28)!important;}
.pf-copy button:hover{opacity:.92;}

.pf-output textarea{font-family:'JetBrains Mono',ui-monospace,monospace!important;font-size:12.5px!important;line-height:1.75!important;
  background:#fafafa!important;}
.pf-status{font-size:12.5px!important;color:var(--soft)!important;text-align:center;min-height:16px;}
.pf-status:empty{display:none;}
"""

COPY_JS = "(t)=>{ if(!t) return; navigator.clipboard.writeText(t).then(()=>{ if(window.pfToast) window.pfToast('Copied to clipboard'); }); }"

# One load script: toast helper + native Web Speech dictation wired straight to the DOM
# (no gr.Audio recorder, no Gradio js= event — both are unreliable in Gradio 6).
INIT_JS = """
() => {
  window.pfToast = (m) => {
    let r=document.getElementById('pf-toast');
    if(!r){ r=document.createElement('div'); r.id='pf-toast';
      r.style.cssText='position:fixed;bottom:24px;left:50%;transform:translateX(-50%);z-index:9999;display:flex;flex-direction:column;gap:8px;align-items:center;';
      document.body.appendChild(r); }
    const t=document.createElement('div'); t.textContent=m;
    t.style.cssText='background:#0a0a0a;color:#fff;padding:9px 16px;border-radius:10px;font-size:13px;font-family:Inter,sans-serif;box-shadow:0 8px 22px rgba(0,0,0,.28);';
    r.appendChild(t); setTimeout(()=>t.remove(),2000);
  };
}
"""

# ── Forge (single model, internal best-of-N, no visible metrics) ──────────────
def forge(raw_prompt, model, depth, request: gr.Request | None = None):
    if not raw_prompt or not raw_prompt.strip():
        yield "", EMPTY_PROMPT_MSG
        return
    client_id = request.client.host if (request and request.client) else "anon"
    if not _LIMITER.allow(client_id):
        yield "", RATE_LIMIT_MSG
        return

    yield "", "⚡ forging & evaluating…"
    task_type = detect_task_type(raw_prompt)

    def _gen(_i):
        try:
            txt, _prov, _u = transform_prompt(raw_prompt, model, task_type, depth, [], "english", False)
            return (txt or "").strip()
        except Exception as e:
            return f"__ERR__{e}"

    results = map_ordered(_gen, list(range(BEST_OF)), max_workers=BEST_OF)
    cands = [r for r in results if r and not r.startswith("__ERR__")]
    if not cands:
        err = next((r[7:] for r in results if r.startswith("__ERR__")), "generation failed")
        yield "", f"⚠️ {err}"
        return

    best_i = pick_best(raw_prompt, cands, model) if len(cands) > 1 else 0
    yield cands[best_i], ""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Promptini") as demo:
    gr.HTML("""
<div class="pf-head">
  <div class="pf-logo">⚡</div>
  <div class="pf-brand">Prompt<span>ini</span></div>
  <div class="pf-tag">Type a rough idea in Derja, Arabic, French or English — get a clean, model-ready prompt.</div>
</div>
""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="pf-panel"):
            raw_input = gr.Textbox(
                label="Your rough idea", elem_id="pf-raw",
                placeholder="Type your rough idea… e.g. اعملّي fonction python باش تقرا csv",
                lines=8,
            )
            with gr.Row(elem_classes="pf-ctl"):
                model_dropdown = gr.Dropdown(
                    [("Claude Code", "claude-code"), ("Claude", "claude"), ("ChatGPT", "gpt-4"),
                     ("Gemini", "gemini"), ("Cursor", "cursor"), ("Any model", "general")],
                    value="claude-code", label="Model",
                )
            with gr.Row(elem_classes="pf-ctl"):
                depth_dropdown = gr.Dropdown(
                    [("Concise", "concise"), ("Balanced", "standard"), ("Detailed", "comprehensive")],
                    value="standard", label="Depth",
                )
            forge_btn = gr.Button("⚡ Forge", variant="primary", elem_classes="pf-forge")
            status = gr.Markdown("", elem_classes="pf-status")

        with gr.Column(scale=1, elem_classes="pf-panel"):
            output = gr.Textbox(label="Optimized prompt", lines=18, interactive=True, elem_classes="pf-output")
            copy_btn = gr.Button("📋 Copy prompt", elem_classes="pf-copy")

    forge_btn.click(fn=forge,
                    inputs=[raw_input, model_dropdown, depth_dropdown],
                    outputs=[output, status])
    copy_btn.click(fn=None, inputs=output, js=COPY_JS)
    demo.load(None, None, None, js=INIT_JS)

if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Soft(), ssr_mode=False)
