# Promptini

Turn rough ideas — in Tunisian Derja, Arabic, French, or English into clean, model-ready prompts, without leaving VS Code.

Promptini reads messy, code-switched input and rewrites it into a structured prompt you can paste into Claude, ChatGPT, Gemini, or Cursor. The optimized prompt always comes out in English.

## Features

- Understands Tunisian Derja and Arabizi (3→ع, 7→ح, 9→ق).
- Formats the prompt for your target model: Claude, ChatGPT, Gemini, Cursor, or Any.
- English output regardless of input language.
- Sidebar panel: type an idea, run Promptini, copy the result.
- Optimize Selection: turn highlighted editor text into an optimized prompt.

## Usage

1. Open the Promptini panel from the Activity Bar.
2. Type or paste an idea, pick a target model, and run it.
3. Copy the prompt into your AI tool.

## Settings

- `promptini.hfUrl` — engine URL (defaults to the hosted app).
- `promptini.mode` — `cloud` (hosted) or `local` (your own server).
- `promptini.apiUrl` / `promptini.apiModel` — endpoint and target model for Optimize Selection.

Runs on a free, open-source stack: Groq inference, Hugging Face hosting.
