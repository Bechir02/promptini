# Free stack — $0 to run

Only constraint: money. Everything M3allem uses is free or open-source.

| Layer | Tool | Cost |
|---|---|---|
| Hosting | Hugging Face Spaces (grandfathered free CPU) | Free |
| LLM (primary) | Groq — Llama-3.3-70B free tier | Free |
| LLM (fallbacks) | Cerebras · OpenRouter · Together · Google AI Studio free tiers | Free (opt-in per key) |
| Embeddings | BAAI/bge-small (or bge-m3 for Derja) via sentence-transformers | Open-source |
| Vector store | LanceDB (embedded) | Open-source |
| **Voice input** | **Browser Web Speech API** (webkitSpeechRecognition) — supports ar-TN | **Free, no server model** |
| UI | Gradio | Open-source |
| Engine API | FastAPI + Uvicorn | Open-source |
| Design | Claude Design canvas | Free |

## Voice dictation
The microphone uses the browser's built-in speech recognition — no Whisper
server, no API cost. It supports Tunisian Arabic (ar-TN), French (fr-FR) and
English (en-US), selectable next to the Dictate button. Works in Chrome/Edge and
on the HF Space (HTTPS); other browsers fall back to typing with a friendly note.

## Optional free upgrades (no cost)
- Better Derja retrieval: PF_EMBEDDING_MODEL=BAAI/bge-m3 (multilingual, open-source).
- More resilience: add OPENROUTER_API_KEY / TOGETHER_API_KEY / GOOGLE_API_KEY
  (all free tiers) — the provider chain uses them automatically.
- Read answers aloud: the browser speechSynthesis API (free) could voice the
  optimized prompt back — a natural next addition.
