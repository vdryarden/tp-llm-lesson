#!/usr/bin/env python3
"""Benchmark all commits using the held-out validation split."""
import json
import re
import subprocess
import sys
import time
import requests
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

API_KEY = (Path(__file__).parent / "api_key").read_text().strip()
MODEL = "deepseek/deepseek-v4-pro"
URL = "https://openrouter.ai/api/v1/chat/completions"
PAIRS_FILE = Path(__file__).parent / "val_pairs.json"


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


def bleu2(hyp_tok, ref_tok):
    hyp = re.findall(r"[a-z]+", hyp_tok.lower())
    ref = re.findall(r"[a-z]+", ref_tok.lower())
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0),
                         smoothing_function=smoothie)


def score_at(commit, pairs, retries=3):
    guide = load_guide_at(commit)
    english = [en for en, _ in pairs]
    for attempt in range(retries):
        try:
            raw = translate_batch(english, guide)
            if not raw:
                raise ValueError("null response")
            translations = []
            for line in (l.strip() for l in raw.splitlines() if l.strip()):
                m = re.match(r"^\d+\.\s*(.+)", line)
                if m:
                    translations.append(re.sub(r"\*+", "", m.group(1)).strip())
            if len(translations) < len(pairs) - 2:
                raise ValueError(f"only {len(translations)} translations")
            scores = [
                bleu2(translations[i] if i < len(translations) else "", ref)
                for i, (_, ref) in enumerate(pairs)
            ]
            return sum(scores) / len(scores)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  retry {attempt+1}: {e}", file=sys.stderr)
                time.sleep(15)
            else:
                raise


def get_commits():
    out = subprocess.run(
        ["git", "log", "--oneline", "--reverse", "code"],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    ).stdout.strip()
    return [(line.split()[0], line.split(" ", 1)[1]) for line in out.splitlines()]


if __name__ == "__main__":
    pairs = json.loads(PAIRS_FILE.read_text())
    commits = get_commits()

    print(f"{'sha':<10} {'score':>6}  message")
    print("-" * 70)
    results = []
    for sha, msg in commits:
        try:
            mean = score_at(sha, pairs)
            print(f"{sha:<10} {mean:>6.3f}  {msg}", flush=True)
            results.append((sha, msg, mean))
        except Exception as e:
            print(f"{sha:<10} {'ERR':>6}  {msg}  ({e})", flush=True)
            results.append((sha, msg, None))
        time.sleep(6)

    Path("val_results.json").write_text(json.dumps(results))
    print("\nSaved val_results.json")
