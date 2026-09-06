# ⚡ Promptini — Derja Prompt Studio

Turn a rough idea — in **Tunisian Derja, Arabizi, Arabic, French, or English** — into a clean, **model-ready prompt**, right inside VS Code.

Promptini understands messy, code-switched developer input (the way people in Tunisia actually type) and rewrites it into a precise, structured, model-optimized prompt you can paste straight into Claude, ChatGPT, Gemini, Cursor, or Copilot.

## Features

- **Derja-first** — understands authentic Tunisian Derja and Arabizi (`3→ع`, `7→ح`, `9→ق`).
- **Model-aware** — formats the prompt for your target tool (Claude, ChatGPT, Gemini, Cursor, or Any).
- **Always-English output** — the messy input can be any language; the optimized prompt comes out clean and English.
- **One-click sidebar panel** — type a rough idea, hit **⚡ Promptini**, and copy the result.
- **Forge Selection** — turn highlighted editor text into an optimized prompt via the engine API.

## Usage

1. Click the **⚡ Promptini** icon in the Activity Bar.
2. Type or paste a rough idea, pick a model, and hit **⚡ Promptini**.
3. **Copy prompt** and paste it into your AI tool.

## Configuration

- `promptForge.hfUrl` — the Promptini engine URL (defaults to the hosted cloud app).
- `promptForge.mode` — `cloud` (hosted engine) or `local` (your own server).
- `promptForge.apiUrl` / `promptForge.apiModel` — endpoint + target model for **Forge Selection**.

Built on a 100% free / open-source stack (Groq inference, Hugging Face hosting).
