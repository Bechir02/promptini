import re

# ── Vague words that weaken prompts ──────────────────────────────────────────
VAGUE_WORDS = [
    "good", "nice", "appropriate", "proper", "clear", "relevant",
    "please", "if possible", "as needed", "when applicable",
    "make sure", "ensure that", "try to", "attempt to",
    "should be", "must be good", "high quality", "best practices",
]

# ── Required sections per model ───────────────────────────────────────────────
REQUIRED_SECTIONS = {
    "claude-code": ["<role>", "<task>", "<constraints>", "<output_format>"],
    "gpt-4":       ["## role", "## task", "## output"],
    "cursor":      [],
    "gemini":      ["task:", "output:"],
    "general":     ["role", "task", "output"],
}

# ── Required elements per task type ──────────────────────────────────────────
TASK_REQUIREMENTS = {
    "code_generation": ["input", "output", "error", "return"],
    "debugging":       ["error", "expected", "actual", "root cause"],
    "code_review":     ["critical", "warning", "suggest"],
    "extraction":      ["null", "json", "field", "missing"],
    "summarization":   ["length", "audience", "exclude"],
    "system_prompt":   ["role", "scope", "tone", "behavior"],
    "analysis":        ["finding", "recommend", "evidence"],
    "refactoring":     ["goal", "behavior", "change"],
    "documentation":   ["param", "return", "example"],
    "writing":         ["tone", "audience", "length"],
    "general":         [],
}


def score_structure(output: str, target_model: str) -> dict:
    """Check if required sections are present for the target model."""
    required = REQUIRED_SECTIONS.get(target_model, [])
    if not required:
        return {"score": 10.0, "found": [], "missing": [], "note": "No structure requirements for this model."}

    output_lower = output.lower()
    found   = [s for s in required if s.lower() in output_lower]
    missing = [s for s in required if s.lower() not in output_lower]
    score   = (len(found) / len(required)) * 10.0

    return {
        "score":   round(score, 1),
        "found":   found,
        "missing": missing,
        "note":    f"{len(found)}/{len(required)} required sections present.",
    }


def score_specificity(output: str) -> dict:
    """Penalize vague words and reward concrete language."""
    output_lower = output.lower()
    found_vague  = [w for w in VAGUE_WORDS if w in output_lower]
    word_count   = len(output.split())

    # Start at 10, deduct for vague words
    deduction = min(len(found_vague) * 1.5, 6.0)
    score     = max(10.0 - deduction, 4.0)

    # Bonus for concrete patterns
    concrete_patterns = [
        r"\{[\w_]+\}",           # placeholders like {variable}
        r"\d+",                   # specific numbers
        r"ValueError|TypeError",  # specific error types
        r"DataFrame|JSON|CSV",    # specific formats
        r"step \d|^\d\.",         # numbered steps
    ]
    bonuses = sum(
        1 for p in concrete_patterns
        if re.search(p, output, re.IGNORECASE | re.MULTILINE)
    )
    score = min(score + bonuses * 0.5, 10.0)

    return {
        "score":       round(score, 1),
        "vague_words": found_vague,
        "note": (
            f"Found {len(found_vague)} vague words: {', '.join(found_vague[:3])}"
            if found_vague else "No vague words detected."
        ),
    }


def score_model_awareness(output: str, target_model: str) -> dict:
    """Check if the output uses the correct format for the target model."""
    output_lower = output.lower()

    checks = {
        "claude-code": {
            "good":  ["<role>", "<task>", "<constraints>"],
            "bad":   ["## role", "## task"],
            "note":  "Should use XML tags, not markdown headers.",
        },
        "gpt-4": {
            "good":  ["##"],
            "bad":   ["<role>", "<task>"],
            "note":  "Should use markdown headers, not XML tags.",
        },
        "cursor": {
            "good":  ["verify", "step", "file"],
            "bad":   ["<role>", "## role"],
            "note":  "Should be minimal and action-oriented.",
        },
        "gemini": {
            "good":  ["task:", "steps:", "output:"],
            "bad":   ["<role>", "<task>"],
            "note":  "Should use numbered steps format.",
        },
        "general": {
            "good":  ["role", "task", "output"],
            "bad":   [],
            "note":  "Should have clear sections.",
        },
    }

    check    = checks.get(target_model, checks["general"])
    good_hit = sum(1 for g in check["good"] if g in output_lower)
    bad_hit  = sum(1 for b in check["bad"]  if b in output_lower)
    score    = min((good_hit / max(len(check["good"]), 1)) * 10.0, 10.0)
    score    = max(score - bad_hit * 3.0, 0.0)

    return {
        "score": round(score, 1),
        "note":  check["note"],
    }


def score_task_coverage(output: str, task_type: str) -> dict:
    """Check if task-specific requirements are covered."""
    required     = TASK_REQUIREMENTS.get(task_type, [])
    if not required:
        return {"score": 10.0, "found": [], "missing": [], "note": "No task requirements."}

    output_lower = output.lower()
    found        = [r for r in required if r in output_lower]
    missing      = [r for r in required if r not in output_lower]
    score        = (len(found) / len(required)) * 10.0

    return {
        "score":   round(score, 1),
        "found":   found,
        "missing": missing,
        "note":    f"{len(found)}/{len(required)} task-specific elements present.",
    }


def score_improvement(raw: str, output: str) -> dict:
    """Check if the output is meaningfully better than the input."""
    raw_words    = len(raw.split())
    output_words = len(output.split())
    ratio        = output_words / max(raw_words, 1)

    # Ideal ratio is 2x-5x longer
    if ratio < 1.2:
        score = 3.0
        note  = "Output is barely longer than input — likely too thin."
    elif ratio < 2.0:
        score = 6.0
        note  = "Moderate improvement over input."
    elif ratio <= 5.0:
        score = 10.0
        note  = f"Good expansion ({ratio:.1f}x longer than input)."
    else:
        score = 7.0
        note  = f"Very long ({ratio:.1f}x) — may have unnecessary padding."

    return {
        "score":       round(score, 1),
        "input_words": raw_words,
        "output_words": output_words,
        "ratio":       round(ratio, 2),
        "note":        note,
    }


# ── Master scorer ─────────────────────────────────────────────────────────────
def score_transformation(
    raw_prompt:   str,
    output:       str,
    target_model: str,
    task_type:    str,
) -> dict:
    """
    Score a prompt transformation across 5 dimensions.
    Returns overall score out of 10 with full breakdown.
    """
    structure    = score_structure(output, target_model)
    specificity  = score_specificity(output)
    model_aware  = score_model_awareness(output, target_model)
    task_cover   = score_task_coverage(output, task_type)
    improvement  = score_improvement(raw_prompt, output)

    # Weighted average
    weights = {
        "structure":   0.25,
        "specificity": 0.25,
        "model_aware": 0.20,
        "task_cover":  0.20,
        "improvement": 0.10,
    }

    overall = (
        structure["score"]   * weights["structure"]   +
        specificity["score"] * weights["specificity"] +
        model_aware["score"] * weights["model_aware"] +
        task_cover["score"]  * weights["task_cover"]  +
        improvement["score"] * weights["improvement"]
    )

    # Grade
    if overall >= 9.0:
        grade = "A — Excellent"
    elif overall >= 7.5:
        grade = "B — Good"
    elif overall >= 6.0:
        grade = "C — Acceptable"
    elif overall >= 4.0:
        grade = "D — Needs Work"
    else:
        grade = "F — Poor"

    return {
        "overall":     round(overall, 1),
        "grade":       grade,
        "breakdown": {
            "structure":   structure,
            "specificity": specificity,
            "model_aware": model_aware,
            "task_cover":  task_cover,
            "improvement": improvement,
        },
    }


def format_score_for_ui(score_result: dict) -> str:
    """Format score result as a readable string for Gradio UI."""
    s = score_result
    b = s["breakdown"]

    lines = [
        f"Overall: {s['overall']}/10  |  Grade: {s['grade']}",
        f"",
        f"Structure     {b['structure']['score']:4.1f}/10  — {b['structure']['note']}",
        f"Specificity   {b['specificity']['score']:4.1f}/10  — {b['specificity']['note']}",
        f"Model-aware   {b['model_aware']['score']:4.1f}/10  — {b['model_aware']['note']}",
        f"Task coverage {b['task_cover']['score']:4.1f}/10  — {b['task_cover']['note']}",
        f"Improvement   {b['improvement']['score']:4.1f}/10  — {b['improvement']['note']}",
    ]
    return "\n".join(lines)