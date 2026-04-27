import re

# ── Vague words that weaken prompts ──────────────────────────────────────────
VAGUE_WORDS = [
    "good", "nice", "appropriate", "proper", "relevant",
    "if possible", "as needed", "when applicable",
    "make sure", "try to", "attempt to",
    "should be good", "high quality", "best practices",
    "be accurate", "be helpful",
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
    "debugging":       ["error", "root cause", "fix", "prevent"],
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

# ── Model awareness checks ────────────────────────────────────────────────────
MODEL_CHECKS = {
    "claude-code": {
        "good": ["<role>", "<task>", "<constraints>", "<output_format>"],
        "bad":  ["## role", "## task", "## output"],
        "note": "Should use XML tags — <role>, <task>, <constraints>, <output_format>.",
    },
    "gpt-4": {
        "good": ["##", "## role", "## task"],
        "bad":  ["<role>", "<task>", "<constraints>"],
        "note": "Should use markdown headers (##), not XML tags.",
    },
    "cursor": {
        "good": ["you are working inside cursor", "verify", "inside cursor"],
        "bad":  ["<role>", "<task>", "## role", "## task"],
        "note": "Should be minimal and action-oriented with no XML or markdown headers.",
    },
    "gemini": {
        "good": ["task:", "steps:", "output:", "step 1", "1."],
        "bad":  ["<role>", "<task>"],
        "note": "Should use numbered steps format with Task/Output sections.",
    },
    "general": {
        "good": ["**role**", "**task**", "**output**", "role:", "task:"],
        "bad":  [],
        "note": "Should have clear bold or labeled sections.",
    },
}


def score_structure(output: str, target_model: str) -> dict:
    """Check if required sections are present for the target model."""
    required = REQUIRED_SECTIONS.get(target_model, [])
    if not required:
        return {
            "score": 10.0,
            "found": [],
            "missing": [],
            "note": "No strict structure requirements for this model.",
        }

    output_lower = output.lower()
    found        = [s for s in required if s.lower() in output_lower]
    missing      = [s for s in required if s.lower() not in output_lower]
    score        = (len(found) / len(required)) * 10.0

    return {
        "score":   round(score, 1),
        "found":   found,
        "missing": missing,
        "note":    f"{len(found)}/{len(required)} required sections present.",
    }


def score_specificity(raw: str, output: str) -> dict:
    """Penalize vague words, reward concrete language, and preserve placeholders."""
    output_lower = output.lower()
    found_vague  = [w for w in VAGUE_WORDS if w in output_lower]

    deduction = min(len(found_vague) * 1.5, 6.0)
    score     = max(10.0 - deduction, 4.0)

    concrete_patterns = [
        r"\{[\w_]+\}",
        r"\b\d+\b",
        r"ValueError|TypeError|KeyError|NullPointer",
        r"DataFrame|JSON|CSV|XML|dict|list",
        r"step \d|^\d\.\s",
        r"raise|return|assert",
    ]
    bonuses = sum(
        1 for p in concrete_patterns
        if re.search(p, output, re.IGNORECASE | re.MULTILINE)
    )
    score = min(score + bonuses * 0.4, 10.0)

    placeholders = re.findall(r"\{[\w_-]+\}", raw)
    missing_placeholders = [p for p in placeholders if p not in output]
    if missing_placeholders:
        deduction += min(len(missing_placeholders) * 2.0, 6.0)
        score = max(10.0 - deduction, 0.0)

    note = []
    if found_vague:
        note.append(f"Found {len(found_vague)} vague words")
    if missing_placeholders:
        note.append(f"missing placeholders: {', '.join(missing_placeholders)}")
    if not note:
        note = ["No vague wording or placeholder drift detected. ✓"]

    return {
        "score":       round(score, 1),
        "vague_words": found_vague,
        "note":        "; ".join(note),
    }


def score_model_awareness(output: str, target_model: str) -> dict:
    """Check if output uses correct format for the target model."""
    output_lower = output.lower()
    check        = MODEL_CHECKS.get(target_model, MODEL_CHECKS["general"])

    good_hit = sum(1 for g in check["good"] if g.lower() in output_lower)
    bad_hit  = sum(1 for b in check["bad"]  if b.lower() in output_lower)

    score = (good_hit / max(len(check["good"]), 1)) * 10.0
    score = max(score - bad_hit * 2.5, 0.0)

    return {
        "score": round(score, 1),
        "note":  check["note"],
    }


def score_task_coverage(output: str, task_type: str) -> dict:
    """Check if task-specific required elements are present."""
    required = TASK_REQUIREMENTS.get(task_type, [])
    if not required:
        return {
            "score":   10.0,
            "found":   [],
            "missing": [],
            "note":    "No specific task requirements.",
        }

    output_lower = output.lower()
    found        = [r for r in required if r.lower() in output_lower]
    missing      = [r for r in required if r.lower() not in output_lower]
    score        = (len(found) / len(required)) * 10.0

    return {
        "score":   round(score, 1),
        "found":   found,
        "missing": missing,
        "note":    f"{len(found)}/{len(required)} task elements present.",
    }


def score_improvement(raw: str, output: str) -> dict:
    """Check if output is meaningfully better than input."""
    raw_words    = len(raw.split())
    output_words = len(output.split())
    ratio        = output_words / max(raw_words, 1)

    if ratio < 1.2:
        score = 3.0
        note  = "Output barely longer than input — likely too thin."
    elif ratio < 1.8:
        score = 5.0
        note  = "Small improvement over input."
    elif ratio <= 3.5:
        score = 9.0
        note  = f"Good expansion — {ratio:.1f}x longer than input. ✓"
    elif ratio <= 5.0:
        score = 8.0
        note  = f"Very detailed output — review for conciseness."
    else:
        score = 6.0
        note  = f"Very long ({ratio:.1f}x) — may have unnecessary padding."

    return {
        "score":        round(score, 1),
        "input_words":  raw_words,
        "output_words": output_words,
        "ratio":        round(ratio, 2),
        "note":         note,
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
    structure   = score_structure(output, target_model)
    specificity = score_specificity(raw_prompt, output)
    model_aware = score_model_awareness(output, target_model)
    task_cover  = score_task_coverage(output, task_type)
    improvement = score_improvement(raw_prompt, output)

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
        "overall":   round(overall, 1),
        "grade":     grade,
        "breakdown": {
            "structure":   structure,
            "specificity": specificity,
            "model_aware": model_aware,
            "task_cover":  task_cover,
            "improvement": improvement,
        },
    }


def format_score_for_ui(score_result: dict) -> str:
    """Format score result as readable string for Gradio UI."""
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