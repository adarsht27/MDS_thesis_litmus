# Item validity in translated LLM benchmarks

Detectors for questions that stop working as test items after a benchmark is
translated, even when the translation itself is correct.

This is pilot work for an MDS master's thesis at the Hertie School, Berlin.
Results here cover English, Hindi and German subsets of
[Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU); the full
study extends to all 42 languages.

## The problem

Most evaluation of language models outside English uses benchmarks written in
English and then translated. The usual concern is translation *quality*: does
the translation say what the original said? That misses a different failure.

Consider this item from Global-MMLU (`electrical_engineering/test/17`). The
English asks what GAL abbreviates in digital logic design, with options
*General Advance Logic*, *General Array Logic*, *Generic Advance Logic* and
*Generic Array Logic*. The question works because English distinguishes
"General" from "Generic". Hindi uses one word, सामान्य, for both. In the Hindi
version option A is character-identical to C, and B to D.

Nobody made a mistake. The Hindi is correct. The item simply cannot exist in
Hindi, because it depends on a distinction Hindi does not draw. A reviewer
asked whether the translation is acceptable has no reason to flag it.

## What is detected

**Answer disclosure.** The text of the correct option appears inside the
question stem, so the item can be answered by matching strings. This is a
property of the item, visible in the dataset, and is distinct from *data
contamination*, where test items appear in a model's training data.

**Option collapse.** Two or more options normalise to the same string, so the
item has no unique correct answer.

Each is reported at two levels of severity. Disclosure is *informative* only
when exactly one option appears in the stem; an item naming several of its own
options gives nothing away. Collapse becomes *key corruption* when the
duplication involves the correct option, at which point the item is
unanswerable and the answer key points at the wrong string.

## Results so far

Global-MMLU `test` split, 14,042 items per language.

| Language | Answer disclosed | Rate | Option collapse | Rate |
|---|---|---|---|---|
| English (source) | 81 | 0.58% | 14 | 0.10% |
| Hindi | 57 | 0.41% | 56 | 0.40% |
| German | 71 | 0.51% | 37 | 0.26% |

Item-by-item against the English source:

| Language | Disclosure added | Disclosure removed | Collapse added | Collapse removed |
|---|---|---|---|---|
| Hindi | 11 | 35 | 42 | 0 |
| German | 12 | 22 | 23 | 0 |

Three things to read out of this.

**The English source is itself defective**, with 81 disclosures and 14
collapses before any translation. Every translated version inherits this floor
and adds its own.

**No collapses are removed by translation**, in either language. This is
structural rather than coincidental: translation can merge two distinct options
but cannot separate two merged ones, so the quantity only accumulates.

**The disclosure numbers fall after translation, and should not be believed.**
Two unrelated languages both appearing to clean up disclosure is not plausible.
Exact matching succeeds in English, where the stem word and the standalone
option are the same string, and fails in inflected languages where the stem
uses a different form. The failure is not uniform: in the Humanities category,
detected disclosures fall from 0.64% in English to 0.55% in German but to 0.28%
in Hindi. Detection therefore degrades with morphological distance from
English, which means counts produced this way are biased against exactly the
languages most likely to be of concern.

That bias is the reason these detectors are a starting point rather than a
result. Establishing what proportion of real failures they find requires
someone who reads the target language to check a sample, which is the next
stage of the thesis.

## Worked examples

The first two need no German.

**`elementary_mathematics/test/238`, German.** 7,285 ÷ 4. English options
1,801 / 1,801 R1 / 1,821 / 1,821 R1, answer D. German: 1.801 / 1.801 R1 /
1.821 / 1.801 R1. The English thousands comma was read as a German decimal
separator, so the correct answer became a duplicate of option B and the true
answer, 1,821 remainder 1, is absent from the item.

**`high_school_macroeconomics/test/85`, German.** The correct option, about net
exports falling as goods become more expensive abroad, has been replaced by a
verbatim copy of option A about currency depreciation. The correct content is
not in the item at all.

**`conceptual_physics/test/113`, Hindi.** "Period" is rendered समय, plain
"time", while the stem contains लगने वाला समय, so the correct answer is the
exact word in the question. The correct term is आवर्तकाल.

**`astronomy/test/25`, Hindi, a false positive.** The detector fires because
आयो appears in the stem, but all four moons are named there, so nothing is
given away, and the English item discloses identically. This case produced the
informativeness filter.

## Usage

```bash
pip install -r requirements.txt

# audit Hindi and German against English
python run_audit.py --languages en hi de

# with worked examples of what translation introduced
python run_audit.py --languages en hi de --examples 10

# many languages, from a file
python run_audit.py --languages-file languages.txt --output-dir output
```

Flagged items are written to `output/flagged_<lang>.csv`, one row per item with
the flag columns appended, so they can be read and coded by hand.

To use the detectors directly:

```python
import pandas as pd
from itemvalidity.detectors import flag_items, summarise, compare_to_source

hindi = flag_items(pd.read_parquet("hi_test.parquet").set_index("sample_id"))
print(hindi["key_corrupted"].sum())
```

## Limitations

The detectors use exact string matching after Unicode NFC normalisation. They
find a lower bound, not a rate. They miss disclosure where the stem inflects a
word differently from the standalone option, which is most of it in
morphologically rich languages, and they miss paraphrase entirely. They cannot
see failures that leave the surface strings distinct, such as a mistranslated
fact inside the question.

Counts are therefore not comparable across languages without an estimate of how
much each language's detection is missing. Producing that estimate for Hindi is
the next stage.

## Data

Global-MMLU (Singh et al., ACL 2025), Apache 2.0. The detectors assume the
schema `sample_id`, `question`, `option_a` to `option_d`, `answer` as an option
letter, and optionally `subject` and `subject_category`. Any multiple-choice
dataset with those fields will work.

