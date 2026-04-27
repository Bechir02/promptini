import os
from groq import Groq
from cerebras.cloud.sdk import Cerebras

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# ── Model config ──────────────────────────────────────────────────────────────
GROQ_MODEL     = "llama-3.3-70b-versatile"
CEREBRAS_MODEL = "llama3.1-8b"

# ── Output format per model ───────────────────────────────────────────────────
OUTPUT_FORMATS = {
    "claude-code": {
        "format":      "xml",
        "description": "XML tags only — <role>, <task>, <context>, <constraints>, <output_format>. No markdown headers. No bullet points outside tags.",
        "example": """<role>
You are a senior Python engineer working inside Claude Code.
</role>
<task>
Implement a CSV duplicate finder that returns a DataFrame of duplicate rows with counts.
</task>
<context>
- File path: {csv_path}
- Expected columns: {columns}
</context>
<constraints>
- Use pandas for all operations
- Handle encoding errors gracefully
- Return empty DataFrame if no duplicates found
- Do not modify the original file
</constraints>
<output_format>
DataFrame with columns: ['duplicate_row', 'count']
Followed by a one-line summary: "Found X duplicate rows."
</output_format>""",
    },

    "gpt-4": {
        "format":      "markdown",
        "description": "Markdown headers only — ## Role, ## Task, ## Instructions, ## Output Format. No XML tags. Instructions as numbered list.",
        "example": """## Role
You are a senior Python engineer.

## Task
Build a CSV duplicate finder that returns duplicate rows with their counts.

## Instructions
1. Accept a file path as input
2. Load the CSV using pandas
3. Identify all duplicate rows
4. Return a DataFrame with columns: duplicate_row, count

## Output Format
- A pandas DataFrame
- Print summary: "Found X duplicate rows."
- Raise ValueError if file not found""",
    },

    "cursor": {
        "format":      "minimal",
        "description": "Minimal, action-first. No headers, no XML. Short paragraphs. One clear task per sentence. You are working inside Cursor IDE. End with a concrete verification step.",
        "example": """You are working inside Cursor IDE on a Python project.

Find all duplicate rows in {csv_path} using pandas. Return a DataFrame with columns duplicate_row and count. Handle missing files with a clear error message.

Verify: run the function on a test CSV with known duplicates and confirm the output matches.""",
    },

    "gemini": {
        "format":      "numbered",
        "description": "Role statement, then numbered steps, then explicit output section. No XML. Clean and direct.",
        "example": """You are a data processing assistant specialized in Python and pandas.

Task: Find duplicate rows in a CSV file and return them with counts.

Steps:
1. Load the CSV from {csv_path} using pandas
2. Identify all fully duplicate rows
3. Count occurrences of each duplicate
4. Return results as a DataFrame

Output: DataFrame with columns [duplicate_row, count]. Print "Found X duplicate rows." as summary.

Edge cases: empty file returns empty DataFrame, missing file raises ValueError.""",
    },

    "general": {
        "format":      "structured",
        "description": "Clear sections with bold headers. Role, Task, Context, Constraints, Output Format. Readable and model-agnostic.",
        "example": """**Role:** You are a Python data engineer.

**Task:** Find duplicate rows in a CSV file and return them with occurrence counts.

**Context:** Input is a CSV file at {csv_path}. Caller expects a pandas DataFrame back.

**Constraints:**
- Use pandas only
- Handle missing files gracefully
- Do not modify the input file

**Output Format:** DataFrame with columns [duplicate_row, count] plus a summary string.""",
    },
}

# ── Task-specific guidance ────────────────────────────────────────────────────
TASK_GUIDANCE = {
    "code_generation": """
- Specify the exact function/class signature expected
- List input types and output types explicitly
- Include error handling requirements
- Mention performance constraints if relevant (large files, speed, memory)
- Add at least one concrete usage example
""",
    "debugging": """
- ALWAYS include these four elements — no exceptions:
  1. The exact error message or exception type
  2. Root cause analysis BEFORE any fix is proposed
  3. The concrete fix with corrected code
  4. Prevention strategy for this class of bug
- Keep scope narrow — fix this specific bug only, do not refactor
- End with a concrete verification step — not "ensure it works" but exactly how to test
- Use the words: error, root cause, fix, prevent
""",
    "code_review": """
- Define the review criteria explicitly (security, performance, style, correctness)
- Specify severity levels for findings (critical / warning / suggestion)
- Ask for positive observations too, not just problems
- Set scope — what NOT to review
- Request actionable fixes, not just observations
""",
    "refactoring": """
- State the refactoring goal precisely (readability / performance / testability)
- Explicitly require: preserve all existing behavior
- Ask for before/after comparison
- Limit scope — one concern at a time
- Request explanation of each change made
""",
    "documentation": """
- Specify doc format (docstring / JSDoc / README / inline comments)
- Define the audience (junior dev / API consumer / end user)
- List what must be covered (params, returns, exceptions, examples)
- Specify length constraints
- Ask for usage examples in the docs
""",
    "analysis": """
- Define what "interesting" means — patterns, anomalies, trends, outliers
- Specify output structure (executive summary + findings + recommendations)
- Ask for confidence levels on conclusions
- Request that assumptions be stated explicitly
- Define the audience for the analysis
""",
    "extraction": """
- List every field to extract with its expected type
- Specify null handling — what to return when a field is missing
- Prohibit hallucination explicitly
- Define output format (JSON schema preferred)
- Handle edge cases: nested fields, arrays, ambiguous values
""",
    "summarization": """
- Specify target length (word count or sentence count)
- Define what must be preserved (key decisions / action items / numbers)
- Define what to exclude (filler / repetition / obvious statements)
- Specify audience and reading level
- Prohibit starting with "This document..."
""",
    "system_prompt": """
- Define the persona with specific traits, not generic ones
- List behavioral rules as explicit do/don't pairs
- Define the scope boundary — what topics are in/out
- Specify tone with concrete adjectives
- Add an out-of-scope redirect behavior
""",
    "writing": """
- Specify genre, tone, and target audience explicitly
- Define length in word count
- List stylistic constraints (avoid passive voice, use short sentences, etc.)
- Prohibit meta-commentary — start writing immediately
- Give one concrete example of the desired style if possible
""",
    "general": """
- Make the task as specific as possible
- Add at least one concrete constraint
- Define the output format explicitly
- Specify what success looks like
""",
}

# ── Anti-generic instructions ─────────────────────────────────────────────────
ANTI_GENERIC = """
WHAT TO AVOID — these make prompts weak and generic:
- Vague quality words: "good", "nice", "appropriate", "proper", "clear", "relevant"
- Redundant phrases: "please", "if possible", "as needed", "when applicable"
- Obvious instructions: "make sure it works", "test your code", "be accurate"
- Over-scaffolding reasoning models with verbose chain-of-thought scripts
- Copying exemplar text verbatim
- Adding sections that add no value for this specific task
"""


# ── Build system prompt ───────────────────────────────────────────────────────
def build_system_prompt(
    target_model: str,
    task_type:    str,
    depth:        str,
    exemplars:    list[dict],
) -> str:

    fmt        = OUTPUT_FORMATS.get(target_model, OUTPUT_FORMATS["general"])
    task_guide = TASK_GUIDANCE.get(task_type, TASK_GUIDANCE["general"])

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
            "output format with schema, edge cases, and one concrete example. "
            "This is the most complete version possible."
        ),
    }.get(depth, "STANDARD MODE: 250-450 words.")

    exemplar_block = ""
    if exemplars:
        exemplar_block = (
            "\n\nSTUDY these high-quality exemplars for this exact model "
            "and task. Learn their structure, specificity, and tone "
            "— do NOT copy them verbatim:\n"
        )
        for i, ex in enumerate(exemplars, 1):
            exemplar_block += (
                f"\n--- Exemplar {i} "
                f"[{ex.get('target_model')} / {ex.get('task_type')}] ---\n"
                f"{ex.get('prompt', '')[:500]}\n"
            )

    return f"""You are a world-class prompt engineer. Your job is to transform messy, vague user prompts into precise, model-optimized, task-specific prompts that produce dramatically better results.

TARGET MODEL: {target_model}
TASK TYPE: {task_type}
DEPTH: {depth}

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
- NO "Here is your prompt:" or any preamble whatsoever
- NO explanation of what you changed
- NO markdown code fences around the output
- NO commentary after the prompt ends
- First character of response = first character of prompt
- Last character of response = last character of prompt"""


# ── Groq call ─────────────────────────────────────────────────────────────────
def call_groq(system_prompt: str, user_prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")

    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens  = 1500,
        temperature = 0.4,
    )
    return response.choices[0].message.content.strip()


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
    Tries Groq first, falls back to Cerebras.
    """
    system_prompt = build_system_prompt(
        target_model = target_model,
        task_type    = task_type,
        depth        = depth,
        exemplars    = exemplars,
    )
    user_message = f"Raw prompt to transform:\n\n{raw_prompt}"

    try:
        result = call_groq(system_prompt, user_message)
        return result, "Groq (llama-3.3-70b)"
    except Exception as e:
        print(f"Groq failed: {e} — falling back to Cerebras...")

    try:
        result = call_cerebras(system_prompt, user_message)
        return result, "Cerebras (llama3.1-8b)"
    except Exception as e:
        raise RuntimeError(
            f"Both Groq and Cerebras failed.\n"
            f"Last error: {e}\n"
            f"Check your API keys in HF Space Secrets."
        )