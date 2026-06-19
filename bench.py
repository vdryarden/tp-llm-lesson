#!/usr/bin/env python3
"""Score guide quality at every git commit using a fixed test set."""
import json
import re
import subprocess
import sys
import requests
from pathlib import Path

API_KEY = (Path(__file__).parent / "api_key").read_text().strip()
MODEL = "deepseek/deepseek-v4-pro"
URL = "https://openrouter.ai/api/v1/chat/completions"
PAIRS_FILE = Path(__file__).parent / "bench_pairs.json"


def load_guide_at(commit):
    root = Path(__file__).parent
    ls = subprocess.run(
        ["git", "ls-tree", "--name-only", commit, "guide/"],
        capture_output=True, text=True, cwd=root,
    ).stdout.strip().split("\n")
    files = sorted(f for f in ls if f.endswith(".md"))
    parts = []
    for f in files:
        content = subprocess.run(
            ["git", "show", f"{commit}:{f}"],
            capture_output=True, text=True, cwd=root,
        ).stdout
        parts.append(f"## {Path(f).name}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def translate_batch(sentences, system):
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = (
        "Translate each of the following English sentences into toki pona. "
        "Reply with only the translations, numbered to match the input. "
        "Do not add any explanation.\n\n" + numbered
    )
    resp = requests.post(
        URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_translations(text, n):
    translations = []
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        m = re.match(r"^\d+\.\s*(.+)", line)
        if m:
            translations.append(re.sub(r"\*+", "", m.group(1)).strip())
    return translations


def jaccard(a_tok, b_tok):
    a = set(re.findall(r"[a-z]+", a_tok.lower()))
    b = set(re.findall(r"[a-z]+", b_tok.lower()))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def get_commits():
    out = subprocess.run(
        ["git", "log", "--oneline", "--reverse"],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    ).stdout.strip()
    return [(line.split()[0], line.split(" ", 1)[1]) for line in out.splitlines()]


def score_at(commit, pairs, retries=3):
    import time
    guide = load_guide_at(commit)
    english = [en for en, _ in pairs]
    for attempt in range(retries):
        try:
            raw = translate_batch(english, guide)
            if raw is None:
                raise ValueError("API returned null content")
            translations = parse_translations(raw, len(pairs))
            scores = [
                jaccard(translations[i] if i < len(translations) else "", ref)
                for i, (_, ref) in enumerate(pairs)
            ]
            return sum(scores) / len(scores)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(10)
            else:
                raise


if __name__ == "__main__":
    if not PAIRS_FILE.exists():
        print("Generating fixed test pairs...", file=sys.stderr)
        from sample_pairs import get_valid_sample
        pairs = get_valid_sample(30)
        PAIRS_FILE.write_text(json.dumps(pairs))
        print(f"Saved {len(pairs)} pairs to {PAIRS_FILE}", file=sys.stderr)

    pairs = json.loads(PAIRS_FILE.read_text())
    commits = get_commits()

    print(f"{'sha':<10} {'score':>6}  message")
    print("-" * 70)
    import time
    for sha, msg in commits:
        try:
            mean = score_at(sha, pairs)
            print(f"{sha:<10} {mean:>6.3f}  {msg}", flush=True)
        except Exception as e:
            print(f"{sha:<10} {'ERR':>6}  {msg}  ({e})", flush=True)
        time.sleep(5)
