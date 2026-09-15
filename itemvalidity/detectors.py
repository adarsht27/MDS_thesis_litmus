"""Detectors for item-validity failures in translated multiple-choice benchmarks.

An item-validity failure is a question that no longer works as a test after
translation, regardless of whether the translation itself is good. Two such
failures are detected here.

Answer disclosure
    The text of the correct option appears inside the question stem, so the
    item can be answered by matching strings rather than by knowing the fact.
    This is a property of the item, visible in the dataset itself, and is
    distinct from data contamination, where test items appear in a model's
    training data.

Option collapse
    Two or more options normalise to the same string, so the item has no
    unique correct answer. This can happen when every option is translated
    correctly, if the target language does not draw a distinction the English
    item depends on.

Both detectors are deliberately conservative. They use exact string matching
after normalisation, which means they find a lower bound rather than a true
rate, and they lose sensitivity in morphologically rich languages where a
concept appears in the stem with different inflection from the standalone
option. Quantifying that shortfall is a separate task and requires a reader
competent in the target language.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "OPTION_COLUMNS",
    "normalise",
    "correct_option_text",
    "flag_items",
    "summarise",
    "compare_to_source",
]

OPTION_COLUMNS: tuple[str, ...] = ("option_a", "option_b", "option_c", "option_d")

#: Options shorter than this are ignored when testing for disclosure, because
#: very short strings match inside longer words by coincidence.
MIN_OPTION_LENGTH = 3


def normalise(text: object) -> str:
    """Return ``text`` in a canonical form suitable for string comparison.

    Applies Unicode Normalisation Form C (canonical composition), strips
    surrounding whitespace, and case-folds.

    The composition step matters for non-Latin scripts. A character can be
    stored in more than one way: the Devanagari letter क़ may be a single
    precomposed code point, or क followed by a separate nukta mark. The two
    render identically and compare as unequal. NFC rewrites both to the
    composed form, so strings that look the same are treated as the same.

    Without this step the detectors would miss genuine duplicates in Indic
    scripts while catching them in Latin ones, making any cross-language
    comparison meaningless.

    Note that case folding is a no-op for scripts without case distinction,
    including Devanagari. It therefore affects the Latin-script languages
    only. This asymmetry is small but real and worth reporting.

    Non-string input returns an empty string, so missing values propagate
    harmlessly.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text).strip().casefold()


def correct_option_text(row: pd.Series) -> str:
    """Return the raw text of the option named by ``row['answer']``.

    The ``answer`` field holds an option letter rather than the answer text.
    Returns an empty string if the letter does not name a known column, which
    keeps a malformed row from aborting a whole run.
    """
    letter = str(row.get("answer", "")).strip().casefold()
    column = f"option_{letter}"
    if column in row.index:
        return row[column]
    return ""


def _count_options_in_stem(option_values: Sequence[str], stem: str) -> int:
    """Count how many normalised options appear as substrings of ``stem``."""
    return sum(
        1
        for value in option_values
        if len(value) >= MIN_OPTION_LENGTH and value in stem
    )


def flag_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with item-validity flags added.

    ``frame`` must contain ``question``, ``answer`` and the four option
    columns. Six columns are added:

    ``answer_disclosed``
        The correct option appears in the stem.

    ``options_in_stem``
        How many of the four options appear in the stem. Used below.

    ``informative_disclosure``
        Disclosure that actually gives something away, i.e. exactly one option
        appears in the stem. An item naming several of its own options in the
        question discloses nothing, because matching the stem does not single
        out an answer. This distinction came from reading flagged items and
        finding that a share of them were harmless.

    ``collapse``
        Two or more options normalise to the same string.

    ``key_corrupted``
        A collapse involving the correct option. The distinction matters: a
        collapse between two distractors leaves the item answerable but raises
        the random-guess baseline from one in four to one in three, whereas a
        collapse involving the correct option leaves the item unanswerable and
        the answer key pointing at a string that is not the right answer.

    ``severity``
        A readable label: ``"ok"``, ``"disclosure"``, ``"collapse"`` or
        ``"key_corrupted"``, taking the most serious that applies.
    """
    missing = {"question", "answer", *OPTION_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    flagged = frame.copy()

    stems = flagged["question"].map(normalise)
    correct = flagged.apply(correct_option_text, axis=1).map(normalise)
    options = flagged[list(OPTION_COLUMNS)].apply(lambda col: col.map(normalise))

    flagged["answer_disclosed"] = [
        len(answer) >= MIN_OPTION_LENGTH and answer in stem
        for stem, answer in zip(stems, correct)
    ]

    flagged["options_in_stem"] = [
        _count_options_in_stem(row, stem)
        for row, stem in zip(options.to_numpy(), stems)
    ]
    flagged["informative_disclosure"] = flagged["answer_disclosed"] & (
        flagged["options_in_stem"] == 1
    )

    # Empty options are dropped before the duplicate test, so a row with a
    # missing option is not reported as a collapse.
    flagged["collapse"] = [
        len([value for value in row if value]) != len({value for value in row if value})
        for row in options.to_numpy()
    ]
    flagged["key_corrupted"] = [
        collapsed and list(row).count(answer) > 1
        for collapsed, row, answer in zip(
            flagged["collapse"], options.to_numpy(), correct
        )
    ]

    flagged["severity"] = [
        "key_corrupted"
        if key
        else "collapse"
        if collapsed
        else "disclosure"
        if disclosed
        else "ok"
        for key, collapsed, disclosed in zip(
            flagged["key_corrupted"],
            flagged["collapse"],
            flagged["informative_disclosure"],
        )
    ]

    return flagged


FLAG_COLUMNS: tuple[str, ...] = (
    "answer_disclosed",
    "informative_disclosure",
    "collapse",
    "key_corrupted",
)


def summarise(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return per-language counts and rates for each flag.

    ``frames`` maps a language code to a frame already passed through
    :func:`flag_items`.
    """
    rows = []
    for language, frame in frames.items():
        total = len(frame)
        row: dict[str, object] = {"language": language, "items": total}
        for column in FLAG_COLUMNS:
            count = int(frame[column].sum())
            row[column] = count
            row[f"{column}_rate"] = count / total if total else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("language")


def compare_to_source(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: Iterable[str] = FLAG_COLUMNS,
) -> pd.DataFrame:
    """Return, per flag, how many items the translation added and removed.

    Both frames must be indexed by a shared item identifier; only items present
    in both are compared.

    ``added`` counts items clean in the source and flagged in the target, which
    is what the translation introduced. ``removed`` counts the reverse.

    How to read ``removed`` depends on the flag. For collapse it should be near
    zero, because translation can merge two distinct options but cannot
    separate two merged ones, so the quantity only accumulates. For disclosure
    a large ``removed`` count is not evidence that the translation improved
    anything; it is evidence that the detector is losing sensitivity, because
    exact matching fails once the stem inflects the word differently from the
    standalone option.
    """
    shared = source.index.intersection(target.index)
    rows = []
    for column in columns:
        in_source = source.loc[shared, column]
        in_target = target.loc[shared, column]
        rows.append(
            {
                "flag": column,
                "compared": len(shared),
                "added": int((~in_source & in_target).sum()),
                "removed": int((in_source & ~in_target).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("flag")
