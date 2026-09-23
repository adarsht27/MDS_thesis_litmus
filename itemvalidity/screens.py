"""Second-tier screens for item-validity failures.

The detectors in ``detectors.py`` are the conservative floor: exact string
matching within one language. The screens here compare each translated item
against its English source, using the question-by-question alignment that
Global-MMLU provides. They have two jobs:

1. find failure types the floor cannot see (e.g. a changed answer key), and
2. rank unflagged items by suspicion, so that human reading can be spent where
   failures are most likely (stratified sampling for Stage 4).

All screens are deterministic, CPU-only and standard-library only. None is a
verdict: each is a candidate generator whose precision must be established by
reading the items it flags.

Screens
-------
key_divergence
    The target answer letter differs from the source answer letter, although
    the options were not reordered.
empty_option
    An option is empty in the target but not in the source. ``flag_items``
    deliberately ignores empty options when testing for collapse, so these
    items are otherwise invisible.
numeric_mismatch
    A short option's digits differ from the source's, after separators are
    ignored ('1,821 R1' -> '1.801 R1'; '800,000' -> '800').
numeric_gained_text
    A purely numeric source option acquired letters in the target
    ('1/2' -> '2-jan'; '2' -> '2 Jahre').
float_artefact
    A short option gained '.0' ('5' -> '5.0'). A processing artefact, not a
    change of value, so it is kept apart from ``numeric_mismatch``.
near_collapse_gain
    How much more alike the two most similar options became in translation.
    A continuous ranking score, not a flag.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

from itemvalidity.detectors import OPTION_COLUMNS, normalise

__all__ = ["screen_items", "retention", "disclosure_boundary_audit"]

#: Numeric screens only look at short options. Longer options are sentences,
#: where translation legitimately reorders numbers or converts them
#: (750 million -> 75 crore in Hindi).
MAX_NUMERIC_OPTION_LENGTH = 20

#: A source option counts as purely numeric if it matches this pattern:
#: digits with separators, spaces, slashes, signs or a decimal point only.
NUMERIC_ONLY = re.compile(r"[\d.,/ :+\-−]+")


# --- helpers ----------------------------------------------------------------


def _is_word_char(char: str) -> bool:
    """True for letters, digits and combining marks.

    Combining marks matter for Indic scripts. Devanagari vowel signs such as
    'ा' are separate code points of Unicode category M*, and ``str.isalnum``
    returns False for them. Without this, 'समयावधि' would appear to contain a
    word boundary straight after 'समय'.
    """
    return char.isalnum() or unicodedata.category(char).startswith("M")


def _digits(text: str) -> str:
    """Return every decimal digit in ``text`` as ASCII, separators dropped.

    '1,821 R1' and '1.821 R1' both give '18211', so English and German number
    formats compare equal. Non-ASCII digits (e.g. Devanagari '१') are mapped
    through their Unicode decimal value.
    """
    return "".join(
        str(unicodedata.decimal(char)) for char in text if unicodedata.decimal(char, None) is not None
    )


def _has_letters(text: str) -> bool:
    """True if ``text`` contains a letter in any script."""
    return any(unicodedata.category(char).startswith("L") for char in text)


def _is_float_artefact(source: str, target: str) -> bool:
    """True if the target is the source integer with '.0' appended."""
    return re.fullmatch(r"\d+\.?", source) is not None and target == source.rstrip(".") + ".0"


def _similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1]; two empty strings count as 0."""
    if not a and not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# --- screens ----------------------------------------------------------------


def screen_items(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Return one row per shared item with the screen results.

    Both frames must be indexed by ``sample_id`` and contain ``answer`` and
    the four option columns. Extra columns (e.g. flags from ``flag_items``)
    are ignored.
    """
    shared = source.index.intersection(target.index)
    rows = []

    for item_id in shared:
        s, t = source.loc[item_id], target.loc[item_id]
        s_opts = [normalise(s[column]) for column in OPTION_COLUMNS]
        t_opts = [normalise(t[column]) for column in OPTION_COLUMNS]

        key_divergence = str(s["answer"]).strip() != str(t["answer"]).strip()
        empty_option = any(a and not b for a, b in zip(s_opts, t_opts))

        short_numeric = [
            (a, b)
            for a, b in zip(s_opts, t_opts)
            if len(a) <= MAX_NUMERIC_OPTION_LENGTH and _digits(a) and b
        ]
        float_artefact = any(_is_float_artefact(a, b) for a, b in short_numeric)
        numeric_mismatch = any(
            _digits(b) and _digits(a) != _digits(b) and not _is_float_artefact(a, b)
            for a, b in short_numeric
        )
        numeric_gained_text = any(
            NUMERIC_ONLY.fullmatch(a) and _has_letters(b) for a, b in short_numeric
        )

        # Near-collapse: find the most similar pair of target options, then
        # subtract how similar the same pair was in English. Positive values
        # mean translation pulled two options together.
        pairs = list(combinations(range(4), 2))
        i, j = max(pairs, key=lambda p: _similarity(t_opts[p[0]], t_opts[p[1]]))
        gain = _similarity(t_opts[i], t_opts[j]) - _similarity(s_opts[i], s_opts[j])

        rows.append(
            {
                "sample_id": item_id,
                "key_divergence": key_divergence,
                "empty_option": empty_option,
                "numeric_mismatch": numeric_mismatch,
                "numeric_gained_text": bool(numeric_gained_text),
                "float_artefact": float_artefact,
                "near_collapse_gain": round(gain, 4),
                "near_collapse_pair": "abcd"[i] + "abcd"[j],
            }
        )

    return pd.DataFrame(rows).set_index("sample_id")


def retention(source: pd.DataFrame, target: pd.DataFrame, flag: str) -> dict[str, float]:
    """Among items flagged in the source, the share still flagged in the target.

    A faithful translation should usually carry an English disclosure into the
    target language, so low retention mostly measures the detector going
    blind rather than the data improving. That assumption is checked by
    reading the 'lost' items (see ``lost_ids`` in the notebook).
    """
    shared = source.index.intersection(target.index)
    in_source = source.loc[shared, flag].astype(bool)
    kept = int((in_source & target.loc[shared, flag].astype(bool)).sum())
    total = int(in_source.sum())
    return {
        "flagged_in_source": total,
        "kept": kept,
        "retention": kept / total if total else float("nan"),
    }


def disclosure_boundary_audit(frame: pd.DataFrame) -> pd.Series:
    """For items flagged ``answer_disclosed``, did the match respect word boundaries?

    ``detectors.py`` tests disclosure by substring, so option '120' matches
    inside '1200' and 'art' inside 'party'. True means the correct option
    occurs in the stem as a whole word at least once; False means it only
    matched inside a longer word.
    """
    from itemvalidity.detectors import correct_option_text

    flagged = frame[frame["answer_disclosed"].astype(bool)]
    result = {}
    for item_id, row in flagged.iterrows():
        stem, answer = normalise(row["question"]), normalise(correct_option_text(row))
        whole_word = False
        for match in re.finditer(re.escape(answer), stem):
            before = stem[match.start() - 1] if match.start() > 0 else " "
            after = stem[match.end()] if match.end() < len(stem) else " "
            if not _is_word_char(before) and not _is_word_char(after):
                whole_word = True
                break
        result[item_id] = whole_word
    return pd.Series(result, name="whole_word_match", dtype=bool)
