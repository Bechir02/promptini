# Deployment & Hosting

How to ship Prompt Forge Rag and where to run it for free.

---

## 1. Deploy to Hugging Face Spaces (current, recommended)

The `origin` remote **is** the live Space:

```
origin  https://huggingface.co/spaces/Becher-zribi/prompt-forge-rag
```

A push to `main` triggers an automatic rebuild + redeploy. The cleaned code is
already committed locally — to ship it:

```bash
git push origin main
```

You'll be prompted for your Hugging Face **username + a write token**
(Settings → Access Tokens on huggingface.co). After the push, watch the build
logs on the Space page; first boot builds the vector index (see §3).

### ⚠️ Do NOT delete or recreate the Space

As of ~July 2026, Hugging Face **no longer offers free CPU for new Gradio
Spaces** — creating one now requires PRO ($9/mo); new free accounts are steered
to ZeroGPU only. **Existing free CPU Spaces are grandfathered.** Your Space is
one of them, so **update it in place (push), never delete-and-recreate** — a
recreate would lose the free CPU seat.

### Rollback

```bash
git revert HEAD        # safe: new commit that undoes the last one, then push
# or, before pushing, undo the local commit entirely:
git reset --soft HEAD~1
```

---

## 2. Config / secrets

The app needs at least one provider key, set as **Space secrets**
(Settings → Variables and secrets), not committed:

| Secret | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes (primary) | Llama-3.3-70B transform + judge |
| `CEREBRAS_API_KEY` | optional | fallback provider |

Optional overrides (all have sane defaults in `core/config.py`): `GROQ_MODEL`,
`CEREBRAS_MODEL`, `PF_MAX_TOKENS`, `PF_TEMPERATURE`, `PF_EMBEDDING_MODEL`.
`app.py` calls `require_provider()` at startup and fails fast with a clear
message if neither key is present.

---

## 3. Cold start (and how to make it fast)

On boot the app builds a LanceDB index by embedding 8,435 prompts with
`bge-small` on CPU — roughly **1–3 minutes**. HF's free disk is **ephemeral**
(wiped on every rebuild / sleep-wake), so this rebuild happens on each cold
start unless the embeddings ship *inside* the repo.

**Optional fast-start:** commit a prebuilt embedding cache so boot skips the
re-embed. This adds a ~6–29 MB binary via Git LFS, so do it on your own machine
(this repo's tooling can't run LFS):

```bash
# one-time, on your machine, with the app deps + git-lfs installed
git lfs install
python -c "from ingest import build_index; build_index(force=True)"   # writes embedding_cache.pkl
# remove the ignore line for it, then:
git add .gitattributes embedding_cache.pkl
git commit -m "chore: ship embedding cache for fast cold starts"
git push origin main
```

`*.pkl` is already routed to LFS in `.gitattributes`. The cache key is content-
hashed against `prompts.json`, so it self-invalidates if the corpus changes.
Trade-off: faster boots vs. a binary + LFS bandwidth in the repo. Skip it and
first load is just slower — the app still works.

---

## 4. Free hosting comparison (2026)

The app is CPU-only but heavy: ~2–3 GB of deps (torch dominates), needs
~1.5–2 GB RAM, and has an expensive cold start.

| Platform | Free RAM / vCPU | Fits torch + 2 GB model? | Sleep / cold start | Persistent disk (free) | Card? / first paid |
|---|---|---|---|---|---|
| **HF Spaces (CPU Basic)** | **16 GB / 2 vCPU** | ✅ easily | sleeps idle; wake re-runs build | ❌ ephemeral 50 GB | no card; Gradio-on-CPU now needs **PRO $9/mo** for *new* Spaces |
| **Google Cloud Run** | up to 2 GiB | ✅ (Dockerfile) | scale-to-zero → rebuild every wake | ❌ (GCS FUSE only) | **card required**; ~50 inst-hrs/mo free |
| **Oracle Cloud (Always Free VM)** | **12–24 GB / 2–4 ARM** | ✅ | ❌ never sleeps | ✅ **~200 GB persistent** | card required; free-forever |
| Render (free web) | 512 MB / 0.1 CPU | ❌ OOMs | spins down 15 min | ❌ | no card; $7/mo |
| Koyeb (free) | 512 MB / 0.1 vCPU | ❌ OOMs | scale-to-zero 1 h | ❌ | maybe card; $29/mo |
| Railway | 1 GB (trial) | ⚠️ borderline | metered | volume deleted post-trial | $5 one-time / then $1 credit/mo |
| Fly.io | none in 2026 | — | — | $0.15/GB/mo | **card; no free tier** |

### Recommendation

- **Primary — stay on Hugging Face Spaces.** Only free box that swallows torch +
  the model with zero tuning, and it runs Gradio natively. Your Space is
  grandfathered free — keep it, push to it.
- **Backup — Google Cloud Run.** The only *always-free* tier that can allocate
  ~2 GiB. Needs a Dockerfile (bake the model + prebuilt index into the image so
  it doesn't re-embed on every scale-from-zero) and a billing account.
- **If you want free-forever + persistent disk** (no re-embed ever) and can run a
  raw VM — **Oracle Cloud Always Free** (12–24 GB RAM, ~200 GB persistent). More
  setup (you own the OS, HTTPS, process supervision) and a card at signup.

### Traps
- **Render / Koyeb free** (512 MB RAM) → torch OOMs on load. Don't.
- **Railway** free is ~$1 credit/mo — a few hours of runtime, not a home.
- **Fly.io** has no free tier in 2026 — it's pay-as-you-go (~$11/mo for 2 GB).

---

*Sources: HF Spaces overview + pricing forums, Cloud Run pricing/memory docs,
Render/Koyeb/Railway/Fly pricing pages, Oracle Always Free (all verified 2026-09).*
