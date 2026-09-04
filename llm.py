import logging
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from cerebras.cloud.sdk import Cerebras

from core.config import get_settings
from core.templates import OUTPUT_FORMATS, TASK_GUIDANCE, ANTI_GENERIC, DERJA_INPUT_NOTE, build_language_block

logger = logging.getLogger(__name__)
_settings = get_settings()

# ── Provider routing ──────────────────────────────────────────────────────────
# Only Groq (primary) and Cerebras (fallback) are supported, by design.
GROQ_MODEL     = _settings.groq_model
CEREBRAS_MODEL = _settings.cerebras_model

# ── Build system prompt ───────────────────────────────────────────────────────
def build_system_prompt(
    target_model: str,
    task_type:    str,
    depth:        str,
    exemplars:    list[dict],
    language:     str = "english",
) -> str:

    fmt        = OUTPUT_FORMATS.get(target_model, OUTPUT_FORMATS["general"])
    task_guide = TASK_GUIDANCE.get(task_type, TASK_GUIDANCE["general"])
    language_block = build_language_block(language)

    depth_guidance = {
        "concise": (
            "CONCISE MODE: Output under 180 words. "
            "Include only role + task + the single most important constraint. "
            "Cut everything else. No examples."
        ),
        "standard": (
            "STANDARD MODE: 250-450 words. "
            "Include role, task, key context, constraints, output format. "
            "No examples needed unless the task is ambiguous."
        ),
        "comprehensive": (
            "COMPREHENSIVE MODE: 450+ words. "
            "Include role, task, detailed context, full constraints, "
            "output format with schema, edge cases, and one concrete example."
        ),
    }.get(depth, "STANDARD MODE: 250-450 words.")

    exemplar_block = ""
    if exemplars:
        exemplar_block = (
            "\n\nSTUDY these high-quality exemplars for this model and task. "
            "Learn their structure, specificity, and tone — do NOT copy verbatim:\n"
        )
        for i, ex in enumerate(exemplars, 1):
            exemplar_block += (
                f"\n--- Exemplar {i} "
                f"[{ex.get('target_model')} / {ex.get('task_type')}] ---\n"
                f"{ex.get('prompt', '')[:500]}\n"
            )

    return f"""You are a world-class prompt engineer. Transform messy, vague user prompts into precise, model-optimized, task-specific prompts that produce dramatically better results.

TARGET MODEL: {target_model}
TASK TYPE: {task_type}
DEPTH: {depth}
{DERJA_INPUT_NOTE}
━━━ OUTPUT LANGUAGE ━━━
{language_block}

━━━ OUTPUT FORMAT FOR {target_model.upper()} ━━━
{fmt['description']}

EXAMPLE of correct format for {target_model}:
{fmt['example']}

━━━ TASK-SPECIFIC REQUIREMENTS FOR {task_type.upper()} ━━━
{task_guide}

━━━ DEPTH REQUIREMENT ━━━
{depth_guidance}

━━━ QUALITY RULES ━━━
1. Every section must earn its place — if it adds no value, cut it
2. Be specific to THIS task — no generic boilerplate
3. Preserve all {{placeholders}} from the original prompt
4. Preserve the user's intent exactly — restructure, never redirect
5. Use the correct format for {target_model} — see example above
6. Make constraints concrete and testable, not vague
7. The transformed prompt must be dramatically more useful than the input
{ANTI_GENERIC}
{exemplar_block}

━━━ ABSOLUTE OUTPUT RULES ━━━
- Output ONLY the transformed prompt — nothing else
- NO preamble like "Here is..." or "I have transformed..."
- NO explanation of what you changed
- NO markdown code fences around the output
- NO commentary after the prompt ends
- First character of response = first character of the prompt"""


# ── Groq call ─────────────────────────────────────────────────────────────────
def call_groq(system_prompt: str, user_prompt: str, model: str | None = None) -> tuple[str, dict]:
    if not _settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not set.")

    client   = Groq(api_key=_settings.groq_api_key)
    response = client.chat.completions.create(
        model    = model or GROQ_MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens  = _settings.max_tokens,
        temperature = _settings.temperature,
    )
    usage = {
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }
    return response.choices[0].message.content.strip(), usage


# ── Cerebras fallback ─────────────────────────────────────────────────────────
def call_cerebras(system_prompt: str, user_prompt: str, model: str | None = None) -> tuple[str, dict]:
    if not _settings.cerebras_api_key:
        raise ValueError("CEREBRAS_API_KEY not set.")

    client   = Cerebras(api_key=_settings.cerebras_api_key)
    response = client.chat.completions.create(
        model    = model or CEREBRAS_MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens = _settings.max_tokens,
    )
    usage = {
        "prompt_tokens":     getattr(response.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
        "total_tokens":      getattr(response.usage, "total_tokens", 0),
    }
    return response.choices[0].message.content.strip(), usage


# ── Main transform ────────────────────────────────────────────────────────────
def transform_prompt(
    raw_prompt:   str,
    target_model: str,
    task_type:    str,
    depth:        str,
    exemplars:    list[dict],
    language:     str = "english",
) -> tuple[str, str, dict]:
    """
    Returns (transformed_prompt, provider_name, usage_dict).
    Tries Groq first, falls back to Cerebras.
    """
    system_prompt = build_system_prompt(
        target_model = target_model,
        task_type    = task_type,
        depth        = depth,
        exemplars    = exemplars,
        language     = language,
    )
    user_message = f"Raw prompt to transform:\n\n{raw_prompt}"

    try:
        result, usage = call_groq(system_prompt, user_message)
        return result, f"Groq ({GROQ_MODEL})", usage
    except Exception as e:
        logger.warning("Groq failed: %s — falling back to Cerebras...", e)

    try:
        result, usage = call_cerebras(system_prompt, user_message)
        return result, f"Cerebras ({CEREBRAS_MODEL})", usage
    except Exception as e:
        raise RuntimeError(
            f"Both Groq and Cerebras failed.\n"
            f"Last error: {e}\n"
            f"Check your API keys in HF Space Secrets."
        )


def _route_model(model: str | None) -> tuple[str, str | None]:
    """Parse a model spec into (provider, model_name).

    Accepts ``None``/``"groq"``/``"cerebras"`` or a ``"provider/model"`` form
    such as ``"groq/llama-3.3-70b-versatile"``. Unknown providers default to
    Groq. Returns the explicit model name (or None to use the provider default).
    """
    if not model:
        return "groq", None
    spec = model.strip().lower()
    provider, _, name = spec.partition("/")
    if provider not in ("groq", "cerebras"):
        # No provider prefix — treat the whole thing as a provider keyword.
        return ("cerebras" if provider == "cerebras" else "groq"), None
    return provider, (name or None)


def call_llm(prompt: str, model: str = "groq", response_format: str = "text") -> str:
    """Unified helper used by scorer.py for LLM-as-a-Judge evaluations.

    Honors the requested ``model`` (provider and optional model name) and still
    falls back to the other provider on failure.
    """
    system = "You are a helpful assistant."
    if "json" in response_format:
        system += " You must respond in valid JSON format."

    provider, model_name = _route_model(model)
    primary, secondary = (
        (call_groq, call_cerebras) if provider == "groq" else (call_cerebras, call_groq)
    )

    try:
        res, _ = primary(system, prompt, model=model_name)
        return res
    except Exception as e:
        logger.warning("Primary provider (%s) failed: %s — trying fallback.", provider, e)
        res, _ = secondary(system, prompt)
        return res