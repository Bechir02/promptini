"""Prompt-engineering templates — the product's domain knowledge in one place.

Previously these dicts lived inside llm.py. They are extracted here so prompt
iteration is a single, reviewable location, and so ``validate_templates()`` can
guarantee every model and task type is covered (run by the test suite and the
``__main__`` linter below).

Each OUTPUT_FORMATS entry needs: ``format``, ``description``, ``example``.
Each TASK_GUIDANCE entry is a guidance string. Coverage is checked against
core.constants.MODELS / TASK_TYPES.
"""

from __future__ import annotations

from .constants import MODELS, TASK_TYPES, OUTPUT_LANGUAGES, ALLOWED_LANGUAGES, DEFAULT_LANGUAGE

REQUIRED_FORMAT_KEYS = ("format", "description", "example")

# ── Output format per model ───────────────────────────────────────────────────
OUTPUT_FORMATS: dict[str, dict[str, str]] = {
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

    "claude": {
        "format":      "xml",
        "description": "XML tags — <role>, <task>, <context>, <constraints>, <output_format>. Clear role. Explicit output format. Step-by-step reasoning for complex tasks.",
        "example": """<role>
You are an expert data analyst.
</role>
<task>
Analyze the provided dataset and identify the top 3 trends.
</task>
<context>
- Dataset: {dataset_path}
- Time period: {time_period}
</context>
<constraints>
- Cite specific data points for each trend
- Avoid speculation beyond the data
- Use plain language suitable for non-technical stakeholders
</constraints>
<output_format>
Three numbered trends, each with: trend name, supporting data, business implication.
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
        "description": "Role statement, then numbered steps, then explicit output section. No XML. Clean and direct. Put the most important instruction at the end.",
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

    "llama": {
        "format":      "structured",
        "description": "Clear role at top. Numbered steps for sequential tasks. Explicit constraints list. Output format section. Direct and explicit — Llama models respond well to clear delimiters.",
        "example": """Role: You are a Python data engineer.

Task: Find all duplicate rows in a CSV file and return them with occurrence counts.

Steps:
1. Accept file path as input parameter
2. Load CSV using pandas read_csv
3. Identify fully duplicate rows using duplicated()
4. Count occurrences and return as DataFrame

Constraints:
- Use only pandas and standard library
- Handle FileNotFoundError explicitly
- Do not modify the input file

Output: DataFrame with columns [duplicate_row, count] plus summary string.""",
    },

    "mistral": {
        "format":      "markdown",
        "description": "Markdown headers for structure. Direct and concise. Clear delimiters between sections. Specify output format explicitly.",
        "example": """## Role
Python data engineer specializing in data quality.

## Task
Find duplicate rows in a CSV file and return them with counts.

## Requirements
- Input: file path as string
- Use pandas for all operations
- Return DataFrame with columns: duplicate_row, count
- Raise ValueError for missing files

## Output Format
DataFrame + one-line summary: "Found X duplicates." """,
    },

    "copilot": {
        "format":      "minimal",
        "description": "Code-focused and file-aware. Reference specific files, functions, line numbers. Action-oriented. One task at a time. Always specify the programming language.",
        "example": """Language: Python

In the file {file_path}, implement a function called find_duplicates(csv_path: str) -> pd.DataFrame.

The function should read the CSV, find all duplicate rows, and return them with occurrence counts as a DataFrame with columns [duplicate_row, count].

Handle FileNotFoundError. Do not modify existing functions in the file.""",
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
TASK_GUIDANCE: dict[str, str] = {
    "code_generation": """
- Specify the exact function/class signature expected
- List input types and output types explicitly
- Include error handling requirements
- Mention performance constraints if relevant
- Add at least one concrete usage example
""",
    "debugging": """
- ALWAYS include these four elements — no exceptions:
  1. The exact error message or exception type
  2. Root cause analysis BEFORE any fix is proposed
  3. The concrete fix with corrected code
  4. Prevention strategy for this class of bug
- Keep scope narrow — fix this specific bug only
- End with a concrete verification step
- Use the words: error, root cause, fix, prevent
""",
    "code_review": """
- Define review criteria explicitly (security, performance, style, correctness)
- Specify severity levels (critical / warning / suggestion)
- Ask for positive observations too
- Set scope — what NOT to review
- Request actionable fixes, not just observations
""",
    "refactoring": """
- State the refactoring goal precisely (e.g., reduce cognitive complexity, improve DRY, add type hints)
- Explicitly require: preserve all existing behavior and pass all current tests
- Ask for a "Before/After" comparison table or summary
- Limit scope — focus on one architectural concern at a time
- Request an explanation of EACH pattern applied (e.g., "Extracted Method", "Replaced Magic Number")
- Require that no new dependencies be added unless specified
""",
    "documentation": """
- Specify doc format (docstring / JSDoc / README / inline)
- Define the audience
- List what must be covered (params, returns, exceptions, examples)
- Specify length constraints
- Ask for usage examples
""",
    "analysis": """
- Define what "interesting" means — patterns, anomalies, trends
- Specify output structure (summary + findings + recommendations)
- Ask for confidence levels on conclusions
- Request that assumptions be stated explicitly
- Define the audience
""",
    "extraction": """
- List every field to extract with its expected type
- Specify null handling explicitly
- Prohibit hallucination
- Define output format (JSON schema preferred)
- Handle edge cases: nested fields, arrays, ambiguous values
""",
    "summarization": """
- Specify target length EXACTLY (e.g., "between 100 and 150 words" or "3-5 bullet points")
- Define what "Key Information" must be preserved (e.g., names, dates, core argument)
- Define what to exclude (e.g., examples, preamble, meta-commentary)
- Specify the reading level (e.g., "Grade 10", "Executive Summary", "Layperson")
- Prohibit generic openings like "This document discusses..." or "The text covers..."
- Ask for a "TL;DR" one-liner at the very top
""",
    "system_prompt": """
- Define the persona with specific traits, not generic ones
- List behavioral rules as explicit do/don't pairs
- Define the scope boundary — what topics are in/out
- Specify tone with concrete adjectives
- Add an out-of-scope redirect behavior
""",
    "writing": """
- Specify genre, tone, and target audience with descriptive adjectives (e.g., "Professional yet witty", "Technical but accessible")
- Define length in word count range
- List stylistic constraints (e.g., "No passive voice", "Use short paragraphs", "Include a punchy headline")
- Prohibit meta-commentary — do not talk about the writing, just WRITE
- Give one concrete example of the desired style and one example of a style to avoid
- Specify the perspective (1st person, 3rd person objective, etc.)
""",
    "general": """
- Make the task as specific as possible
- Add at least one concrete constraint
- Define the output format explicitly
- Specify what success looks like
""",
}

# ── Anti-generic rules ────────────────────────────────────────────────────────
ANTI_GENERIC = """
WHAT TO AVOID — these make prompts weak:
- Vague words: "good", "nice", "appropriate", "proper", "clear", "relevant"
- Redundant phrases: "please", "if possible", "as needed", "when applicable"
- Obvious instructions: "make sure it works", "test your code", "be accurate"
- Copying exemplar text verbatim
- Adding sections that add no value for this specific task
"""


# ── Input: Tunisian Derja / Arabizi understanding (always injected) ────────────
DERJA_INPUT_NOTE = """
INPUT LANGUAGE NOTE:
The user's raw prompt may be written in Tunisian Derja (Tunisian Arabic), or in
"Arabizi" / franco-arabe (Derja written in Latin letters, often code-switched with
French and English). Interpret it faithfully before optimizing. Common Arabizi
digit substitutions: 3=ع, 7=ح, 9=ق, 5=خ, 2=ء (hamza), 8=غ, 6=ط. Treat mixed
Derja + French + English in one sentence as normal. Never refuse or ask for a
translation — infer the intent and proceed.
"""

# ── Output language directives ────────────────────────────────────────────────
LANGUAGE_GUIDANCE: dict[str, str] = {
    "auto": (
        "Write the optimized prompt in the SAME language and script the user used in "
        "their raw prompt (mirror their language and register). If they wrote in "
        "Arabizi, prefer Arabic-script Derja unless they clearly want Latin script."
    ),
    "english": "Write the optimized prompt in clear, professional English.",
    "arabic": (
        "Write the optimized prompt in Modern Standard Arabic (الفصحى). Keep code, "
        "placeholders like {var}, API names and technical identifiers in their "
        "original English / Latin form."
    ),
    "derja": (
        "Write the optimized prompt in natural, idiomatic Tunisian Derja (Tunisian "
        "Arabic) in Arabic script. Keep code, placeholders like {var}, API names and "
        "technical identifiers in their original English / Latin form."
    ),
    "french": "Write the optimized prompt in clear, professional French.",
}


def normalize_language(language: str | None) -> str:
    """Return a valid output language, defaulting when unknown/empty."""
    if not language:
        return DEFAULT_LANGUAGE
    lang = str(language).strip().lower()
    return lang if lang in ALLOWED_LANGUAGES else DEFAULT_LANGUAGE


def build_language_block(language: str | None) -> str:
    """Render the OUTPUT LANGUAGE directive for the system prompt."""
    lang = normalize_language(language)
    return LANGUAGE_GUIDANCE.get(lang, LANGUAGE_GUIDANCE[DEFAULT_LANGUAGE])


def validate_templates() -> list[str]:
    """Return a list of coverage/shape problems. Empty list == healthy.

    Checks every model in MODELS has an OUTPUT_FORMATS entry with the required
    keys, and every task type in TASK_TYPES has TASK_GUIDANCE.
    """
    problems: list[str] = []

    for model in MODELS:
        entry = OUTPUT_FORMATS.get(model)
        if entry is None:
            problems.append(f"OUTPUT_FORMATS missing model: {model}")
            continue
        for key in REQUIRED_FORMAT_KEYS:
            if not entry.get(key):
                problems.append(f"OUTPUT_FORMATS[{model}] missing/empty key: {key}")

    for task in TASK_TYPES:
        if not TASK_GUIDANCE.get(task):
            problems.append(f"TASK_GUIDANCE missing task type: {task}")

    for lang in OUTPUT_LANGUAGES:
        if not LANGUAGE_GUIDANCE.get(lang):
            problems.append(f"LANGUAGE_GUIDANCE missing language: {lang}")

    return problems


if __name__ == "__main__":
    issues = validate_templates()
    if issues:
        print("Template problems found:")
        for p in issues:
            print(f"  - {p}")
        raise SystemExit(1)
    print(f"✅ Templates valid: {len(OUTPUT_FORMATS)} models, {len(TASK_GUIDANCE)} task types, {len(LANGUAGE_GUIDANCE)} languages.")
