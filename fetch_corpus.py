import os
import json
import time
import requests
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OUTPUT_FILE  = "prompts.json"
HEADERS      = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}

# ── Repositories ──────────────────────────────────────────────────────────────
REPOS = [

    # ── Claude / Claude Code ──────────────────────────────────────────────────
    {
        "repo":         "Piebald-AI/claude-code-system-prompts",
        "target_model": "claude-code",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".json"],
    },
    {
        "repo":         "repowise-dev/claude-code-prompts",
        "target_model": "claude-code",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "anthropics/prompt-eng-interactive-tutorial",
        "target_model": "claude",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".ipynb"],
    },
    {
        "repo":         "langgptai/awesome-claude-prompts",
        "target_model": "claude",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "langgptai/wonderful-prompts",
        "target_model": "claude",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "oxbshw/System-Prompt-Agent-Prompts",
        "target_model": "claude-code",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".json"],
    },
    {
        "repo":         "Comfy-Org/comfy-claude-prompt-library",
        "target_model": "claude",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".json"],
    },
    {
        "repo":         "VoltAgent/awesome-claude-code-subagents",
        "target_model": "claude-code",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },

    # ── GPT-4 / GPT-4o ───────────────────────────────────────────────────────
    {
        "repo":         "f/awesome-chatgpt-prompts",
        "target_model": "gpt-4",
        "license":      "CC0-1.0",
        "extensions":   [".md", ".csv"],
    },
    {
        "repo":         "awesome-chatgpt-prompts/awesome-chatgpt-prompts-github",
        "target_model": "gpt-4",
        "license":      "CC0-1.0",
        "extensions":   [".md", ".csv"],
    },
    {
        "repo":         "bharatkalluri/awesome-prompts",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "brexhq/prompt-engineering",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "microsoft/promptbase",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".json"],
    },
    {
        "repo":         "openai/openai-cookbook",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".ipynb"],
    },
    {
        "repo":         "mustvlad/ChatGPT-System-Prompts",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".json"],
    },
    {
        "repo":         "linexjlin/GPTs",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "LouisShark/chatgpt_system_prompt",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "mattnigh/ChatGPT-Free-Prompt-List",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "PickleBoxer/play-with-chatgpt",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },

    # ── Cursor ────────────────────────────────────────────────────────────────
    {
        "repo":         "DVC2/cursor_prompts",
        "target_model": "cursor",
        "license":      "MIT",
        "extensions":   [".md", ".mdc", ".txt"],
    },
    {
        "repo":         "instructa/ai-prompts",
        "target_model": "cursor",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".mdc"],
    },
    {
        "repo":         "PatrickJS/awesome-cursorrules",
        "target_model": "cursor",
        "license":      "CC0-1.0",
        "extensions":   [".md", ".mdc", ".txt"],
    },
    {
        "repo":         "pontusab/cursor.directory",
        "target_model": "cursor",
        "license":      "MIT",
        "extensions":   [".md", ".mdc", ".ts", ".tsx"],
    },

    # ── Gemini ────────────────────────────────────────────────────────────────
    {
        "repo":         "YouMind-OpenLab/awesome-gemini-3-prompts",
        "target_model": "gemini",
        "license":      "CC BY 4.0",
        "extensions":   [".md", ".txt", ".json"],
    },
    {
        "repo":         "ZeroLu/awesome-gemini-ai",
        "target_model": "gemini",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "langgptai/awesome-gemini-prompts",
        "target_model": "gemini",
        "license":      "CC0-1.0",
        "extensions":   [".md", ".txt"],
    },

    # ── Llama ─────────────────────────────────────────────────────────────────
    {
        "repo":         "langgptai/awesome-llama-prompts",
        "target_model": "llama",
        "license":      "Apache-2.0",
        "extensions":   [".md", ".txt"],
    },

    # ── Mistral ───────────────────────────────────────────────────────────────
    {
        "repo":         "samouraiworld/awesome-mistral",
        "target_model": "mistral",
        "license":      "CC0-1.0",
        "extensions":   [".md", ".txt"],
    },

    # ── GitHub Copilot ────────────────────────────────────────────────────────
    {
        "repo":         "pnp/copilot-prompts",
        "target_model": "copilot",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".json"],
    },

    # ── Agentic ───────────────────────────────────────────────────────────────
    {
        "repo":         "e2b-dev/awesome-ai-agents",
        "target_model": "general",
        "license":      "Apache-2.0",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "kyrolabs/awesome-agents",
        "target_model": "general",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "e2b-dev/e2b-cookbook",
        "target_model": "general",
        "license":      "Apache-2.0",
        "extensions":   [".md", ".txt", ".ipynb"],
    },
    {
        "repo":         "microsoft/TypeChat",
        "target_model": "gpt-4",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },

    # ── General Prompt Engineering ────────────────────────────────────────────
    {
        "repo":         "promptslab/Awesome-Prompt-Engineering",
        "target_model": "general",
        "license":      "Apache-2.0",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "dair-ai/Prompt-Engineering-Guide",
        "target_model": "general",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
    {
        "repo":         "ai-boost/awesome-prompts",
        "target_model": "general",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".json"],
    },
    {
        "repo":         "NirPolak/promptify",
        "target_model": "general",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },

    # ── Domain — Data Science ─────────────────────────────────────────────────
    {
        "repo":         "dataprofessor/prompt-engineering",
        "target_model": "general",
        "license":      "MIT",
        "extensions":   [".md", ".txt", ".ipynb"],
    },
    {
        "repo":         "microsoft/Data-Science-For-Beginners",
        "target_model": "general",
        "license":      "MIT",
        "extensions":   [".md"],
    },

    # ── Domain — Security ─────────────────────────────────────────────────────
    {
        "repo":         "TakSec/chatgpt-prompts-bug-bounty",
        "target_model": "general",
        "license":      "MIT",
        "extensions":   [".md", ".txt"],
    },
]

# ── Task type detection ───────────────────────────────────────────────────────
# Canonical classifier shared with the live pipeline (rag.py). Previously this
# file had its own divergent keyword lists, so corpus labels disagreed with
# query-time labels. Now there is a single source of truth.
from core.tasks import detect_task_type


# ── Quality scoring ───────────────────────────────────────────────────────────
def quality_score(text: str) -> float:
    if len(text) < 50:
        return 0.0

    t = text.lower()

    bad_patterns = [
        "dan ", "jailbreak", "ignore previous instructions",
        "ignore all instructions", "disregard all",
        "bypass", "pretend you have no",
        "act as if you have no restrictions",
        "you are now", "unlock mode",
        "developer mode", "no restrictions",
    ]
    if any(p in t for p in bad_patterns):
        return 0.0

    score = 5.0

    # Structure signals
    if any(w in t for w in ["<role>", "<task>", "<context>", "<constraints>", "<output_format>"]):
        score += 2.0
    if any(w in t for w in ["## role", "## task", "## context", "## output"]):
        score += 1.5
    if any(w in t for w in ["output format", "output_format", "format:", "return format"]):
        score += 1.0
    if any(w in t for w in ["example", "e.g.", "for instance", "sample"]):
        score += 0.5
    if any(w in t for w in ["step by step", "step-by-step", "first", "then", "finally"]):
        score += 0.5
    if any(w in t for w in ["constraint", "rule:", "must not", "do not", "never"]):
        score += 0.5
    if any(w in t for w in ["verify", "confirm", "test", "validate"]):
        score += 0.5

    # Length signals
    if len(text) > 300:
        score += 1.0
    if len(text) > 600:
        score += 0.5
    if len(text) > 1000:
        score += 0.3

    # Negative signals
    if text.count("\n") < 2:
        score -= 1.0
    if any(w in t for w in ["lol", "haha", "omg", "wtf", "idk"]):
        score -= 2.0
    if len(text) < 100:
        score -= 1.0

    return min(max(score, 0.0), 10.0)


# ── GitHub API helpers ────────────────────────────────────────────────────────
def get_repo_tree(repo: str) -> list:
    url      = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("tree", [])
    print(f"  ⚠️  Could not fetch tree for {repo}: {response.status_code}")
    return []


def get_file_content(repo: str, path: str) -> str:
    for branch in ["main", "master"]:
        url      = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.text
    return ""


# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_csv_prompts(
    content: str, target_model: str,
    license: str, repo: str,
) -> list:
    prompts = []
    lines   = content.strip().split("\n")
    for i, line in enumerate(lines[1:], 1):
        parts = line.split('","')
        if len(parts) >= 2:
            act    = parts[0].strip('" ')
            prompt = parts[1].strip('" ')
            if prompt and len(prompt) > 50:
                score = quality_score(prompt)
                if score >= 5.0:
                    prompts.append({
                        "id":            f"{target_model}_csv_{i:04d}",
                        "target_model":  target_model,
                        "task_type":     detect_task_type(prompt),
                        "prompt":        prompt,
                        "source_repo":   repo,
                        "license":       license,
                        "quality_score": round(score, 1),
                        "act":           act,
                    })
    return prompts


def parse_ipynb_prompts(
    content: str, target_model: str,
    license: str, repo: str, idx: int,
) -> list:
    prompts = []
    try:
        nb    = json.loads(content)
        cells = nb.get("cells", [])
        for j, cell in enumerate(cells):
            if cell.get("cell_type") in ["markdown", "code"]:
                src = "".join(cell.get("source", []))
                if len(src) > 100:
                    score = quality_score(src)
                    if score >= 6.0:
                        prompts.append({
                            "id":            f"{target_model}_nb_{idx:03d}_{j:02d}",
                            "target_model":  target_model,
                            "task_type":     detect_task_type(src),
                            "prompt":        src[:1000],
                            "source_repo":   repo,
                            "license":       license,
                            "quality_score": round(score, 1),
                        })
    except Exception:
        pass
    return prompts


def parse_json_prompts(
    content: str, target_model: str,
    license: str, repo: str,
    file_idx: int,
) -> list:
    prompts = []
    try:
        data = json.loads(content)
        items = data if isinstance(data, list) else []
        for j, item in enumerate(items):
            text = item.get("prompt", item.get("content", item.get("text", "")))
            if text and len(text) > 50:
                score = quality_score(text)
                if score >= 6.0:
                    prompts.append({
                        "id":            f"{target_model}_json_{file_idx:03d}_{j:02d}",
                        "target_model":  target_model,
                        "task_type":     detect_task_type(text),
                        "prompt":        text[:1000],
                        "source_repo":   repo,
                        "license":       license,
                        "quality_score": round(score, 1),
                    })
    except Exception:
        pass
    return prompts


def parse_markdown_prompts(
    content: str, target_model: str, license: str,
    source_repo: str, file_path: str, idx: int,
) -> list:
    import re
    prompts = []
    blocks  = []

    code_blocks = re.findall(r"```(?:[\w]*)\n([\s\S]*?)```", content)
    blocks.extend(code_blocks)

    paragraphs = [
        p.strip() for p in content.split("\n\n")
        if len(p.strip()) > 100
    ]
    blocks.extend(paragraphs)

    for j, block in enumerate(blocks):
        block = block.strip()
        score = quality_score(block)
        if score >= 6.0:
            slug = Path(file_path).stem[:20].replace(" ", "_")
            prompts.append({
                "id":            f"{target_model}_{slug}_{idx:03d}_{j:02d}",
                "target_model":  target_model,
                "task_type":     detect_task_type(block),
                "prompt":        block[:1000],
                "source_repo":   source_repo,
                "license":       license,
                "quality_score": round(score, 1),
            })

    return prompts


# ── Main fetch ────────────────────────────────────────────────────────────────
def fetch_all_prompts() -> list:
    all_prompts = []

    for repo_config in REPOS:
        repo         = repo_config["repo"]
        target_model = repo_config["target_model"]
        license      = repo_config["license"]
        extensions   = repo_config["extensions"]

        print(f"\n📦 {repo}")
        tree = get_repo_tree(repo)

        if not tree:
            print(f"  Skipping.")
            continue

        files = [
            item["path"] for item in tree
            if item["type"] == "blob"
            and any(item["path"].endswith(ext) for ext in extensions)
        ]

        print(f"  {len(files)} files found.")
        repo_count = 0

        for i, file_path in enumerate(files):
            content = get_file_content(repo, file_path)
            if not content:
                continue

            if file_path.endswith(".csv"):
                parsed = parse_csv_prompts(
                    content, target_model, license, repo)
            elif file_path.endswith(".ipynb"):
                parsed = parse_ipynb_prompts(
                    content, target_model, license, repo, i)
            elif file_path.endswith(".json"):
                parsed = parse_json_prompts(
                    content, target_model, license, repo, i)
            else:
                parsed = parse_markdown_prompts(
                    content, target_model, license, repo, file_path, i)

            all_prompts.extend(parsed)
            repo_count += len(parsed)
            time.sleep(0.25)

        print(f"  → {repo_count} prompts")

    return all_prompts


# ── Deduplicate ───────────────────────────────────────────────────────────────
def deduplicate(prompts: list) -> list:
    seen   = set()
    unique = []
    for p in prompts:
        key = p["prompt"][:200].strip().lower()
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Starting corpus fetch...")
    print(f"Token: {'✅ set' if GITHUB_TOKEN else '⚠️  not set'}")
    print(f"Repos: {len(REPOS)}")

    fetched = fetch_all_prompts()
    print(f"\n✅ Fetched {len(fetched)} raw prompts.")

    unique = deduplicate(fetched)
    print(f"✅ After dedup: {len(unique)} unique prompts.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to {OUTPUT_FILE}")

    models = Counter(p["target_model"] for p in unique)
    tasks  = Counter(p["task_type"]    for p in unique)

    print("\n── By model ─────────────────────────────────")
    for model, count in models.most_common():
        print(f"  {model:20s} {count}")

    print("\n── By task type ─────────────────────────────")
    for task, count in tasks.most_common():
        print(f"  {task:20s} {count}")