import os
from google import genai
from google.genai import types
from cerebras.cloud.sdk import Cerebras

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# ── Model config ──────────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-2.0-flash"
CEREBRAS_MODEL = "llama3.1-8b"

# ── Model-specific rules ──────────────────────────────────────────────────────
MODEL_RULES = {
    "claude-code": """
- Use XML tags to separate sections: <role>, <task>, <context>, <constraints>, <output_format>
- Be explicit about file paths, tools, and step-by-step reasoning
- Claude Code operates in an agentic loop — write the prompt assuming multi-step execution
- Specify verification steps after each action
""",
    "gpt-4": """
- Use clear markdown headers (##) to separate sections
- Be direct and task-focused — GPT-4 responds well to explicit instructions
- Include a clear output format specification
- Chain-of-thought is optional but helpful for complex tasks
""",
    "cursor": """
- Cursor operates inside an IDE with file context — reference files and lines explicitly
- Keep instructions minimal and action-oriented
- One logical change at a time
- Always end with a verification step
""",
    "gemini": """
- Gemini handles multimodal input — specify input type clearly if not text
- Use numbered steps for sequential tasks
- Be explicit about output format and length
- Gemini responds well to role + task + format structure
""",
    "general": """
- Use clear role + task + context + output format structure
- Be explicit about constraints and edge cases
- Include output format specification
- Add examples if the task is ambiguous
""",
}

# ── Build system prompt ───────────────────────────────────────────────────────
def build_system_prompt(
    target_model: str,
    task_type:    str,
    depth:        str,
    exemplars:    list[dict],
) -> str:

    model_rules = MODEL_RULES.get(target_model, MODEL_RULES["general"])

    depth_guidance = {
        "concise":       "Keep the output tight and minimal. Under 180 words. Only what is essential.",
        "standard":      "Balanced structure. Cover role, task, context, format, constraints. 250-450 words.",
        "comprehensive": "Full structure with XML tags, examples, edge cases, and reasoning guidance. 450+ words.",
    }.get(depth, "Balanced structure. 250-450 words.")

    exemplar_block = ""
    if exemplars:
        exemplar_block = "\n\nHere are high-quality reference prompts for this model and task type. Use their structure and style as inspiration — do not copy them verbatim:\n"
        for i, ex in enumerate(exemplars, 1):
            exemplar_block += f"\n--- Exemplar {i} [{ex.get('target_model')} / {ex.get('task_type')}] ---\n"
            exemplar_block += ex.get("prompt", "")[:400]
            exemplar_block += "\n"

    return f"""You are an expert prompt engineer specializing in {target_model} prompts.

Transform the user's raw prompt into a structured, effective prompt optimized for {target_model}.

Model-specific rules for {target_model}:
{model_rules}

Depth requirement:
{depth_guidance}

General prompt engineering principles:
1. Open with a clear role definition
2. State the task precisely with success criteria
3. Include only context the model actually needs
4. Specify output format explicitly
5. List constraints and edge cases
6. Preserve any {{placeholders}} the user wrote
7. Encourage step-by-step reasoning for complex tasks
8. Preserve the user's original intent — improve structure, do not change the goal
{exemplar_block}

ABSOLUTE OUTPUT RULES:
- Output ONLY the transformed prompt
- NO preamble like "Here is..." or "I've transformed..."
- NO explanation of what you changed
- NO markdown code fences
- NO commentary anywhere
- First character of your response = first character of the prompt
- Ready to copy-paste immediately"""


# ── Gemini call ───────────────────────────────────────────────────────────────
def call_gemini(system_prompt: str, user_prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set.")

    client   = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model  = GEMINI_MODEL,
        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            max_output_tokens  = 1500,
            temperature        = 0.7,
        ),
        contents = user_prompt,
    )
    return response.text.strip()


# ── Cerebras fallback ─────────────────────────────────────────────────────────
def call_cerebras(system_prompt: str, user_prompt: str) -> str:
    if not CEREBRAS_API_KEY:
        raise ValueError("CEREBRAS_API_KEY not set.")

    client   = Cerebras(api_key=CEREBRAS_API_KEY)
    response = client.chat.completions.create(
        model    = CEREBRAS_MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens = 1500,
    )
    return response.choices[0].message.content.strip()


# ── Main transform function ───────────────────────────────────────────────────
def transform_prompt(
    raw_prompt:   str,
    target_model: str,
    task_type:    str,
    depth:        str,
    exemplars:    list[dict],
) -> tuple[str, str]:
    """
    Transform a raw prompt into a structured one.
    Returns (transformed_prompt, provider_used).
    Tries Gemini first, falls back to Cerebras on failure.
    """
    system_prompt = build_system_prompt(
        target_model = target_model,
        task_type    = task_type,
        depth        = depth,
        exemplars    = exemplars,
    )
    user_message = f"Raw prompt to transform:\n\n{raw_prompt}"

    # Try Gemini first
    try:
        result = call_gemini(system_prompt, user_message)
        return result, "Gemini 2.0 Flash"
    except Exception as e:
        print(f"Gemini failed: {e} — falling back to Cerebras...")

    # Fallback to Cerebras
    try:
        result = call_cerebras(system_prompt, user_message)
        return result, "Cerebras (llama3.1-8b)"
    except Exception as e:
        raise RuntimeError(
            f"Both Gemini and Cerebras failed.\n"
            f"Last error: {e}\n"
            f"Check your API keys in HF Space Secrets."
        )