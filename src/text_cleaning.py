"""Day 2-3: Regex based text cleaning and delivery-time extraction.

Changes vs. the exploratory notebook (and why):
  1. Contractions are expanded BEFORE punctuation is stripped, so "didn't" becomes
     "did not" instead of "didn t". Negation is essential for complaint detection.
  2. Decimal points between digits are preserved, so "1.5 hours" stays "1.5 hours"
     (before it became "1 5 hours" and was mis-read as 5 hours = 300 min).
  3. Delivery time must appear close to a delivery word (deliver/arrive/zomato/swiggy/...),
     so "ordered, then waited 2 hours for a table" is not counted as a delivery time.
     (Version 1 counted any "order" word and produced 168 matches, ~half of them dine-in waits;
     the stricter rule gives ~86 genuine delivery times.)
"""
import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")

_CONTRACTIONS = [
    (r"\bcan't\b", "can not"),
    (r"\bwon't\b", "will not"),
    (r"\bain't\b", "is not"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'m\b", " am"),
    (r"'ve\b", " have"),
    (r"'ll\b", " will"),
    (r"'d\b", " would"),
    (r"'s\b", ""),
]


def expand_contractions(text: str) -> str:
    for pattern, repl in _CONTRACTIONS:
        text = re.sub(pattern, repl, text)
    return text


def clean_review(text) -> str:
    """Lower-case, strip URLs/@mentions/emojis/punctuation, keep decimals."""
    if pd.isna(text):
        return ""
    t = str(text).lower().replace("\u2019", "'").replace("`", "'")
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = expand_contractions(t)
    t = re.sub(r"[^\w\s.]", " ", t)               # drop emojis & punctuation (keep '.')
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)      # drop '.' unless it sits between digits
    t = t.replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------
# Delivery-time extraction
# ----------------------------------------------------------------------------
# NOTE: plain "order/ordered" is deliberately NOT a delivery word - "we ordered and waited 30 mins at the
# table" is a dine-in wait (handled as a Staff & Service complaint), not a delivery time.
DELIVERY_WORDS = r"(?:deliver\w*|arriv\w*|zomato|swiggy|dispatch\w*|rider|courier|parcel|take ?away|packed|receiv\w*)"
HOUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|hr)\b")
SPACED_HOUR_RE = re.compile(r"\b(\d)\s(\d)\s*(?:hours?|hrs?|hr)\b")  # legacy "1 5 hours"
MIN_RE = re.compile(r"(\d+)\s*(?:mins?|minutes?)\b")
WINDOW = 80  # characters between the delivery word and the time expression


def _near_delivery_word(text: str, start: int, end: int) -> bool:
    left = text[max(0, start - WINDOW):start]
    right = text[end:end + WINDOW]
    return bool(re.search(DELIVERY_WORDS, left) or re.search(DELIVERY_WORDS, right))


def extract_delivery_time(review):
    """Return (matched_text, minutes) or (None, None).

    Sanity limits: 0 < hours <= 6, 0 < minutes <= 180.
    """
    if not isinstance(review, str) or not review:
        return None, None
    text = review.lower()

    for rx, kind in ((SPACED_HOUR_RE, "spaced"), (HOUR_RE, "hour"), (MIN_RE, "min")):
        for m in rx.finditer(text):
            if not _near_delivery_word(text, m.start(), m.end()):
                continue
            if kind == "spaced":
                value = float(f"{m.group(1)}.{m.group(2)}") * 60
            elif kind == "hour":
                value = float(m.group(1)) * 60
            else:
                value = float(m.group(1))
            if 0 < value <= (360 if kind != "min" else 180):
                return m.group(), value
    return None, None
