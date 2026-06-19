#!/usr/bin/env python3
import re
import sys
import requests
from pathlib import Path
from sample_pairs import get_valid_sample

API_KEY = (Path(__file__).parent / "api_key").read_text().strip()
MODEL = "deepseek/deepseek-v4-pro"
URL = "https://openrouter.ai/api/v1/chat/completions"
GUIDE_DIR = Path(__file__).parent / "guide"

def load_guide(files=None):
    if files is None:
        files = sorted(GUIDE_DIR.glob("*.md"))
    else:
        files = [GUIDE_DIR / f for f in files]
    parts = [f"## {f.name}\n\n{f.read_text()}" for f in files]
    return "\n\n---\n\n".join(parts)

def translate_batch(sentences, system):
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = (
        "Translate each of the following English sentences into toki pona. "
        "Reply with only the translations, numbered to match the input. "
        "Do not add any explanation.\n\n"
        + numbered
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
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def score_pair(model_tok, ref_tok):
    a = set(re.findall(r"[a-z]+", model_tok.lower()))
    b = set(re.findall(r"[a-z]+", ref_tok.lower()))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def parse_translations(text, n):
    if not text:
        print(f"Warning: empty response from model", file=sys.stderr)
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    translations = []
    for line in lines:
        m = re.match(r"^\d+\.\s*(.+)", line)
        if m:
            tok = re.sub(r"\*+", "", m.group(1)).strip()  # strip bold markdown
            translations.append(tok)
    if len(translations) != n:
        print(f"Warning: expected {n} translations, got {len(translations)}", file=sys.stderr)
    return translations

guide = load_guide()
pairs = get_valid_sample(30)
english = [en for en, _ in pairs]

print(f"Sending {len(english)} sentences to {MODEL}...\n")
raw = translate_batch(english, system=guide)
translations = parse_translations(raw, len(pairs))

scores = []
for i, (en, ref_tok) in enumerate(pairs):
    model_tok = translations[i] if i < len(translations) else "(missing)"
    sc = score_pair(model_tok, ref_tok)
    scores.append(sc)
    print(f"en:       {en}")
    print(f"model:    {model_tok}")
    print(f"dataset:  {ref_tok}")
    print(f"score:    {sc:.2f}")
    print()

print(f"mean score: {sum(scores)/len(scores):.3f}  (token Jaccard over {len(scores)} sentences)")
