# Running it live (streaming + API + extension)

Do this on your machine (or HF), where outbound LLM/HF calls work — the cloud
sandbox blocks them, so streaming can only be seen here.

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# .env needs at least GROQ_API_KEY (CEREBRAS_API_KEY optional; OPENROUTER/TOGETHER/GOOGLE optional)
```

First run downloads the embedding model and builds the index (1–3 min); after
that it's fast.

## 1. Gradio app — see streaming (B8)

```bash
python app.py     # http://127.0.0.1:7860
```

Pick **one** model → the optimized prompt streams in token-by-token (status shows
"⚡ streaming · <model>…", then the score card appears at the end). Pick **two**
models → arena runs both in parallel and shows the battle winner (batch, not
streamed). Try a Derja prompt + Output language = Derja, and the "Decompose into
steps" toggle.

## 2. Engine API (C9) + streaming endpoint (B8)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/forge \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"write a csv dedupe function","target_model":"claude-code"}'

# streaming (Server-Sent Events) — watch tokens arrive:
curl -N -X POST localhost:8000/forge/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"write a csv dedupe function","target_model":"claude-code"}'
```

## 3. VS Code extension → API command

1. Rebuild + install the extension:
   ```bash
   cd vscode-extension && npm install && npm run compile && npm run package
   code --install-extension prompt-forge-vscode-0.8.0.vsix
   ```
2. Start the engine (step 2). In VS Code Settings, `Prompt Forge: Api Url`
   defaults to `http://127.0.0.1:8000` (and `Api Model` to `general`).
3. Select text in any file → right-click → **Prompt Forge: Forge Selection via
   API** → the selection is replaced with the optimized prompt.

## Notes
- Streaming shows for single-model runs; the arena stays batch-parallel by design.
- If no streaming-capable provider is configured, streaming falls back to a single
  final chunk automatically.
