# toki pona LLM improvement loop

An iterative experiment in using a language model's translation output as
feedback for improving a toki pona lesson.

## What this is

A [toki pona](https://tokipona.org) learner's guide — a human-readable,
page-by-page course covering all ~120 core words and the grammar of the
language. The guide doubles as the system prompt for a translation model:
the same document that teaches a human reader is fed verbatim to the model,
and the model's translation quality reveals what the guide explains well
and what it is missing.

The hypothesis: **a gap in the model's output is a gap in the guide**.
Fixing it improves both the course and the model's translations.

## How the loop works

```
run translate.py
       │
       ▼
compare model output to dataset reference translations
       │
       ▼
identify a systematic error or missing explanation
       │
       ▼
find the guide page where that concept is first introduced
       │
       ▼
add a natural explanation with examples — written for a human learner
       │
       ▼
run again — did the error disappear without new regressions?
       │
  yes  │  no
  ─────┴──────
commit    revert
```

One change per iteration. Commit only on net improvement.

## Files

| file | purpose |
|------|---------|
| `translate.py` | loads all `guide/*.md` as the system prompt, samples 30 random sentences from the Tatoeba toki pona dataset, sends them to the model, and prints a side-by-side comparison |
| `sample_pairs.py` | pulls from [`NetherQuartz/tatoeba-tokipona`](https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona) and filters to sentences whose reference translations only use known vocabulary |
| `guide/` | the toki pona course — each page introduces new vocabulary and grammar |

## Model

[DeepSeek V4 Pro](https://openrouter.ai/deepseek/deepseek-v4-pro) via OpenRouter (`deepseek/deepseek-v4-pro`).

## What the guide learned to explain

Each commit captures one pattern that was missing from the lesson. Highlights:

| concept | before | after |
|---------|--------|-------|
| likes / dislikes | `mi olin e moku` | `moku li pona tawa mi` |
| buying something | `mi esun e len` | `mi kama jo e len lon esun` |
| speaking a language | `ona li toki e toki Kanse` | `ona li toki pona kepeken toki Kanse` |
| show X to someone | `mi pana e ijo tawa ona` | `mi pana e ijo tawa lukin ona` |
| sitting down | `mi anpa` | `mi anpa e monsi` |
| arrive at a place | `mi kama tawa tomo` | `mi kama lon tomo` |
| afraid | `mi pilin ike` | `mi pilin monsuta` |
| angry | `mi pilin ike` | `mi kama pilin utala` |

## Constraint: context size

All guide pages are concatenated into a single string passed as the system
prompt. DeepSeek V4 Pro stops responding above roughly **122,800 characters**.
Every addition to the guide must be offset by trimming elsewhere — which
forces concision: every word in the lesson has to earn its place.

## Running

```sh
export OPENROUTER_API_KEY=...   # or write key to ./api_key
python3 translate.py
```

Requires: `requests`, `datasets` (HuggingFace).
