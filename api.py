"""FastAPI engine endpoint (C9).

Exposes the pipeline over HTTP so the VS Code extension and other clients can
call the engine directly, not only through the Gradio iframe.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from rag import run_pipeline
from scorer import score_transformation
from core.config import get_settings
from core.metrics import metrics
from core.constants import DEFAULT_MODEL, DEFAULT_DEPTH, DEFAULT_LANGUAGE

app = FastAPI(title="Prompt Forge Engine", version="1.0")


class ForgeRequest(BaseModel):
    prompt: str
    target_model: str = DEFAULT_MODEL
    depth: str = DEFAULT_DEPTH
    language: str = DEFAULT_LANGUAGE
    score: bool = True


@app.get("/health")
def health():
    s = get_settings()
    return {"ok": s.has_any_provider, "metrics": metrics.snapshot()}


@app.post("/forge")
def forge(req: ForgeRequest):
    res = run_pipeline(
        raw_prompt=req.prompt,
        target_model=req.target_model,
        depth=req.depth,
        language=req.language,
    )
    out = {
        "transformed": res["transformed"],
        "provider":    res["provider"],
        "task_type":   res["task_type"],
        "error":       res["error"],
    }
    if req.score and res["transformed"]:
        out["score"] = score_transformation(
            req.prompt, res["transformed"], req.target_model, res["task_type"]
        )
    return out
