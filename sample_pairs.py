#!/usr/bin/env python3
import random
import re
from pathlib import Path
from datasets import load_dataset

DICT_PATH = Path(__file__).parent / "guide" / "dictionary.md"

def load_tp_words(path):
    words = set()
    for line in path.read_text().splitlines():
        m = re.match(r"^#### \[([^\]]+)\]", line)
        if m:
            for word in m.group(1).split("/"):
                words.add(word.strip())
    return words

def is_valid_tok(text, vocab):
    for token in re.split(r"[\s.,!?;:\"'()]+", text):
        if not token:
            continue
        if token[0].isupper():  # proper noun
            continue
        if token not in vocab:
            return False, token
    return True, None

def get_valid_sample(n=30):
    vocab = load_tp_words(DICT_PATH)
    ds = load_dataset("NetherQuartz/tatoeba-tokipona", split="train")
    valid = [(row["en"], row["tok"]) for row in ds
             if row["en"] and row["tok"] and is_valid_tok(row["tok"], vocab)[0]]
    return random.sample(valid, n)

if __name__ == "__main__":
    sample = get_valid_sample(30)
    for en, tok in sample:
        print(f"en:  {en}")
        print(f"tok: {tok}")
        print()
