import os
import json
import time
import requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OUTPUT_FILE  = "prompts.json"
HEADERS      = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}

# ── Repos to fetch ────────────────────────────────────────────────────────────
REPOS = [
    {
        "repo":         "Piebald-AI/claude-code-system-prompts",
        "target_model": "claude-code",
        "license":      "MIT",
        "paths":        ["prompts", "tools", "subagents", "utility"],
        "extensions":   [".md", ".txt", ".json"],
    },
    {
        "repo":         "repowise-dev/claude-code-prompts",
        "target_model": "claude-code",
        "license":      "MIT",
        "paths":        ["prompts", "patterns", "skills", "complete_prompts"],
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "DVC2/cursor_prompts",
        "target_model": "cursor",
        "license":      "MIT",
        "paths":        [".cursor/rules", "prompts", "examples"],
        "extensions":   [".md", ".mdc", ".txt"],
    },
    {
        "repo":         "awesome-chatgpt-prompts/awesome-chatgpt-prompts-github",
        "target_model": "gpt-4",
        "license":      "CC0-1.0",
        "paths":        [""],
        "extensions":   [".csv", ".md"],
    },
    {
        "repo":         "YouMind-OpenLab/awesome-gemini-3-prompts",
        "target_model": "gemini",
        "license":      "CC BY 4.0",
        "paths":        ["prompts", ""],
        "extensions":   [".md", ".txt", ".json"],
    },
]

# ── Keyword-based task type detector ─────────────────────────────────────────
def detect_task_type(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["extract", "pull out", "parse json", "get fields"]):
        return "extraction"
    if any(w in t for w in ["system prompt", "persona", "act as", "you are a"]):
        return "system_prompt"
    if any(w in t for w in ["review", "audit", "check for bugs", "scan for"]):
        return "code_review"
    if any(w in t for w in ["fix", "debug", "error", "bug", "not working"]):
        return "debugging"
    if any(w in t for w in ["refactor", "clean up", "optimize", "restructure"]):
        return "refactoring"
    if any(w in t for w in ["document", "docstring", "readme", "explain"]):
        return "documentation"
    if any(w in t for w in ["analyze", "analysis", "compare", "research"]):
        return "analysis"
    if any(w in t for w in ["summarize", "summary", "tldr", "overview"]):
        return "summarization"
    if any(w in t for w in ["story", "essay", "blog", "creative", "poem"]):
        return "writing"
    if any(w in t for w in ["write", "create", "build", "implement",
                             "function", "class", "script"]):
        return "code_generation"
    return "general"


# ── Quality filter ────────────────────────────────────────────────────────────
def quality_score(text: str) -> float:
    """
    Simple heuristic quality scorer.
    Returns a score between 0-10.
    Filters out jailbreaks, very short prompts, and generic fluff.
    """
    score = 5.0
    t     = text.lower()

    # Too short — useless
    if len(text) < 50:
        return 0.0

    # Jailbreak / DAN patterns — discard
    bad_patterns = [
        "dan ", "jailbreak", "ignore previous instructions",
        "ignore all instructions", "you are now", "pretend you are",
        "act as if you have no", "disregard", "bypass"
    ]
    if any(p in t for p in bad_patterns):
        return 0.0

    # Positive signals
    if any(w in t for w in ["<role>", "<task>", "<context>", "<constraints>"]):
        score += 2.0
    if any(w in t for w in ["output format", "output_format", "format:"]):
        score += 1.0
    if any(w in t for w in ["example", "e.g.", "for instance"]):
        score += 0.5
    if any(w in t for w in ["step by step", "step-by-step", "first", "then", "finally"]):
        score += 0.5
    if len(text) > 300:
        score += 1.0
    if len(text) > 600:
        score += 0.5

    # Negative signals
    if text.count("\n") < 2:
        score -= 1.0
    if any(w in t for w in ["lol", "haha", "omg", "wtf"]):
        score -= 2.0

    return min(max(score, 0.0), 10.0)


# ── GitHub API helpers ────────────────────────────────────────────────────────
def get_repo_tree(repo: str) -> list:
    """Get full file tree for a repo."""
    url      = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("tree", [])
    print(f"  ⚠️  Could not fetch tree for {repo}: {response.status_code}")
    return []


def get_file_content(repo: str, path: str) -> str:
    """Fetch raw file content from GitHub."""
    url      = f"https://raw.githubusercontent.com/{repo}/main/{path}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        # Try master branch
        url      = f"https://raw.githubusercontent.com/{repo}/master/{path}"
        response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.text
    return ""


def parse_csv_prompts(content: str, target_model: str, license: str) -> list:
    """Parse awesome-chatgpt-prompts style CSV."""
    prompts = []
    lines   = content.strip().split("\n")
    for i, line in enumerate(lines[1:], 1):  # skip header
        parts = line.split('","')
        if len(parts) >= 2:
            act    = parts[0].replace('"', '').strip()
            prompt = parts[1].replace('"', '').strip()
            if prompt and len(prompt) > 50:
                score = quality_score(prompt)
                if score >= 5.0:
                    prompts.append({
                        "id":           f"{target_model}_csv_{i:04d}",
                        "target_model": target_model,
                        "task_type":    detect_task_type(prompt),
                        "prompt":       prompt,
                        "source_repo":  "awesome-chatgpt-prompts",
                        "license":      license,
                        "quality_score": round(score, 1),
                        "act":          act,
                    })
    return prompts


def parse_markdown_prompts(
    content:      str,
    target_model: str,
    license:      str,
    source_repo:  str,
    file_path:    str,
    idx:          int,
) -> list:
    """
    Parse markdown files — extract code blocks and substantial paragraphs
    as prompt candidates.
    """
    prompts = []
    blocks  = []

    # Extract fenced code blocks
    import re
    code_blocks = re.findall(r"```(?:[\w]*)\n([\s\S]*?)```", content)
    blocks.extend(code_blocks)

    # Extract paragraphs longer than 100 chars
    paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 100]
    blocks.extend(paragraphs)

    for j, block in enumerate(blocks):
        block = block.strip()
        score = quality_score(block)
        if score >= 6.0:
            slug = Path(file_path).stem[:20].replace(" ", "_")
            prompts.append({
                "id":           f"{target_model}_{slug}_{idx:03d}_{j:02d}",
                "target_model": target_model,
                "task_type":    detect_task_type(block),
                "prompt":       block,
                "source_repo":  source_repo,
                "license":      license,
                "quality_score": round(score, 1),
            })

    return prompts


# ── Main fetch function ───────────────────────────────────────────────────────
def fetch_all_prompts() -> list:
    all_prompts = []

    for repo_config in REPOS:
        repo         = repo_config["repo"]
        target_model = repo_config["target_model"]
        license      = repo_config["license"]
        extensions   = repo_config["extensions"]

        print(f"\n📦 Fetching: {repo}")
        tree = get_repo_tree(repo)

        if not tree:
            print(f"  Skipping — could not fetch tree.")
            continue

        # Filter files by extension
        files = [
            item["path"] for item in tree
            if item["type"] == "blob"
            and any(item["path"].endswith(ext) for ext in extensions)
        ]

        print(f"  Found {len(files)} files to process.")

        for i, file_path in enumerate(files):
            print(f"  [{i+1}/{len(files)}] {file_path}")
            content = get_file_content(repo, file_path)

            if not content:
                continue

            # CSV files (awesome-chatgpt-prompts style)
            if file_path.endswith(".csv"):
                parsed = parse_csv_prompts(content, target_model, license)
                all_prompts.extend(parsed)
                print(f"    → {len(parsed)} prompts from CSV")

            # JSON files
            elif file_path.endswith(".json"):
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for j, item in enumerate(data):
                            text = item.get("prompt", item.get("content", ""))
                            if text:
                                score = quality_score(text)
                                if score >= 6.0:
                                    all_prompts.append({
                                        "id":           f"{target_model}_json_{i:03d}_{j:02d}",
                                        "target_model": target_model,
                                        "task_type":    detect_task_type(text),
                                        "prompt":       text,
                                        "source_repo":  repo,
                                        "license":      license,
                                        "quality_score": round(score, 1),
                                    })
                except json.JSONDecodeError:
                    pass

            # Markdown / text / mdc files
            else:
                parsed = parse_markdown_prompts(
                    content, target_model, license, repo, file_path, i
                )
                all_prompts.extend(parsed)
                if parsed:
                    print(f"    → {len(parsed)} prompts extracted")

            # Rate limit protection
            time.sleep(0.3)

    return all_prompts


# ── Merge with existing seed prompts ─────────────────────────────────────────
def merge_with_seed(fetched: list, seed_file: str = "prompts.json") -> list:
    """Keep existing hand-crafted seed prompts and add fetched ones."""
    try:
        with open(seed_file, "r") as f:
            seed = json.load(f)
        print(f"\nLoaded {len(seed)} seed prompts from {seed_file}.")
    except Exception:
        seed = []

    # Deduplicate by prompt text (first 200 chars)
    seen     = {p["prompt"][:200] for p in seed}
    new_only = []
    for p in fetched:
        key = p["prompt"][:200]
        if key not in seen:
            new_only.append(p)
            seen.add(key)

    merged = seed + new_only
    print(f"Merged: {len(seed)} seed + {len(new_only)} new = {len(merged)} total prompts.")
    return merged


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Starting corpus fetch...")
    print(f"GitHub token: {'✅ set' if GITHUB_TOKEN else '⚠️  not set — rate limited to 60 req/hour'}")

    fetched = fetch_all_prompts()
    print(f"\n✅ Fetched {len(fetched)} raw prompts.")

    merged = merge_with_seed(fetched)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved {len(merged)} prompts to {OUTPUT_FILE}")

    # Stats
    from collections import Counter
    models = Counter(p["target_model"] for p in merged)
    tasks  = Counter(p["task_type"]    for p in merged)

    print("\n── By model ─────────────────────────────")
    for model, count in models.most_common():
        print(f"  {model:20s} {count}")

    print("\n── By task type ─────────────────────────")
    for task, count in tasks.most_common():
        print(f"  {task:20s} {count}")