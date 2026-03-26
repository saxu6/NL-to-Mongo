"""Schema-aware NL preprocessor.

Converts prose queries into semi-structured text that the Lark grammar can parse.

    "give me the list of products whose gross sale values are greater than 1000 in the past month"
    → "find products where gross_sale_value greater than 1000 after 2026-02-09"
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import spacy

from .resolver import SchemaContext

# ---------------------------------------------------------------------------
# spaCy model (loaded once)
# ---------------------------------------------------------------------------

_NLP: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        for model_name in ("en_core_web_md", "en_core_web_sm"):
            try:
                _NLP = spacy.load(model_name, disable=["ner"])
                break
            except OSError:
                continue
        if _NLP is None:
            _NLP = spacy.blank("en")
    return _NLP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FILLER_PHRASES = [
    "can you please", "could you please", "i would like to",
    "i want to see", "i want to", "please show me", "please give me",
    "please list", "can you show me", "can you give me", "can you get me",
    "could you show me", "show me the", "give me the list of",
    "give me the", "give me a list of", "give me a", "give me all",
    "show me all", "show me a list of", "show me", "give me",
    "get me the", "get me all", "get me a", "get me",
    "list of all", "list of the", "list of",
    "the list of", "a list of",
    "i need to", "i need", "i want",
    "please", "kindly",
]

_INTENT_VERBS = {
    "show": "find", "give": "find", "get": "find", "fetch": "find",
    "display": "find", "retrieve": "find", "list": "find", "see": "find",
    "find": "find", "return": "find",
    "count": "count", "total": "count",
    "delete": "delete", "remove": "delete", "drop": "delete",
    "update": "update", "modify": "update", "change": "update",
    "edit": "update", "set": "update",
}

_COPULAS = {"is", "are", "was", "were", "has", "have", "had", "does", "do", "did", "been"}

_RELATIVE_TIMES = {
    "today": 0, "yesterday": 1,
    "last week": 7, "past week": 7, "this week": 7, "previous week": 7,
    "last month": 30, "past month": 30, "this month": 30, "previous month": 30,
    "last year": 365, "past year": 365, "this year": 365, "previous year": 365,
    "last 7 days": 7, "past 7 days": 7, "last 30 days": 30, "past 30 days": 30,
    "last 90 days": 90, "past 90 days": 90,
    "last 24 hours": 1, "past 24 hours": 1,
}

_OPERATOR_PHRASES = {
    "greater than", "more than", "less than", "at least", "at most",
    "equal to", "not equal", "is not", "does not equal",
    "greater than or equal", "less than or equal",
}

# Words that are NEVER part of a field name
_NON_FIELD_WORDS = {
    "and", "or", "where", "with", "that", "whose", "having",
    "greater", "less", "more", "equal", "equals", "contains", "contain",
    "matches", "match", "like", "before", "after", "at", "least", "most",
    "sort", "sorted", "order", "ordered", "by", "limit", "top", "first",
    "asc", "ascending", "desc", "descending",
    "is", "not", "in", "than", "to", "of", "the", "a", "an",
    "show", "find", "list", "get", "give", "count", "delete", "update",
}

_WHERE_MARKERS = {"where", "with", "whose", "that", "having"}

# Words between collection and filter that should be dropped
_POST_COLLECTION_FILLERS = {
    "were", "was", "are", "is", "has", "have", "had", "been",
    "placed", "made", "created", "recorded", "logged", "generated",
    "which", "who",
}

_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(text: str, ctx: Optional[SchemaContext] = None) -> str:
    """Convert prose NL query into semi-structured text for the Lark parser."""
    if not text or not text.strip():
        return text

    s = text.strip()

    # Pass 1: strip filler phrases
    s = _strip_fillers(s)

    # Pass 2: extract intent from leading verb
    intent, s = _extract_intent(s)

    # Pass 3: extract and convert temporal expressions
    s, temporal_clause = _extract_temporal(s)

    # Pass 4: convert word-numbers to digits ("five" → "5")
    s = _convert_word_numbers(s)

    # Pass 5: join multi-word field references using schema (4/3/2-word windows)
    s = _join_field_phrases(s, ctx)

    # Pass 6: strip copulas before operators ("are greater than" → "greater than")
    s = _strip_copulas(s)

    # Pass 7: strip post-collection filler verbs ("orders were placed" → "orders")
    s = _strip_post_collection_fillers(s)

    # Pass 8: ensure "where" is present before filter expressions
    s = _ensure_where_marker(s)

    # Pass 9: assemble final string
    parts = []
    if intent:
        parts.append(intent)
    if s.strip():
        parts.append(s.strip())
    if temporal_clause:
        combined = " ".join(parts)
        if re.search(r'\b(where|with|whose|that|having)\b', combined, re.I):
            parts.append(f"and {temporal_clause}")
        else:
            parts.append(f"where {temporal_clause}")

    result = " ".join(parts)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# ---------------------------------------------------------------------------
# Pass 1: Filler removal
# ---------------------------------------------------------------------------

def _strip_fillers(text: str) -> str:
    s = text
    for phrase in sorted(_FILLER_PHRASES, key=len, reverse=True):
        escaped = re.escape(phrase).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        s = pattern.sub(" ", s, count=1)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Pass 2: Intent extraction
# ---------------------------------------------------------------------------

def _extract_intent(text: str) -> tuple[str, str]:
    words = text.split()
    if not words:
        return "", text

    first = words[0].lower().rstrip(".,!?")

    # "number of" → count
    if len(words) >= 2 and first == "number" and words[1].lower() == "of":
        return "count", " ".join(words[2:])

    # "how many" → count
    if len(words) >= 2 and first == "how" and words[1].lower() == "many":
        return "count", " ".join(words[2:])

    # Direct match
    intent = _INTENT_VERBS.get(first)
    if intent:
        return intent, " ".join(words[1:])

    # spaCy lemma fallback
    nlp = _get_nlp()
    doc = nlp(words[0])
    if doc and doc[0].pos_ == "VERB":
        lemma = doc[0].lemma_.lower()
        intent = _INTENT_VERBS.get(lemma)
        if intent:
            return intent, " ".join(words[1:])

    return "find", text


# ---------------------------------------------------------------------------
# Pass 3: Temporal extraction
# ---------------------------------------------------------------------------

def _extract_temporal(text: str) -> tuple[str, str]:
    """Extract relative time phrases → date after YYYY-MM-DD."""
    for phrase, days_back in sorted(_RELATIVE_TIMES.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(
            r'(?:in\s+(?:the\s+)?|from\s+(?:the\s+)?|during\s+(?:the\s+)?|of\s+(?:the\s+)?)?'
            + re.escape(phrase),
            re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            date_val = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

            # Check if temporal phrase is inside an existing filter context
            # e.g. "where up_event_start after last week"
            before = text[:match.start()].rstrip()
            after_text = text[match.end():].strip()

            # If "after" or "before" precedes the temporal phrase, replace inline
            inline_op = re.search(r'\b(after|before)\s*$', before, re.I)
            if inline_op:
                # "up_event_start after last week" → "up_event_start after 2026-03-02"
                remaining = before + " " + after_text
                return remaining.strip(), f"date after {date_val}"

            remaining = text[:match.start()] + text[match.end():]
            return remaining.strip(), f"date after {date_val}"

    return text, ""


# ---------------------------------------------------------------------------
# Pass 4: Word-number conversion
# ---------------------------------------------------------------------------

def _convert_word_numbers(text: str) -> str:
    words = text.split()
    return " ".join(_WORD_TO_NUM.get(w.lower(), w) for w in words)


# ---------------------------------------------------------------------------
# Pass 5: Schema-first multi-word field joining (4/3/2-word windows)
# ---------------------------------------------------------------------------

def _join_field_phrases(text: str, ctx: Optional[SchemaContext]) -> str:
    """Scan for multi-word sequences that match schema fields.
    Uses sliding windows of size 4, 3, 2 with collection_hint.
    Falls back to noun-chunk joining if no schema is available."""

    if not ctx:
        return _normalize_noun_chunks_fallback(text)

    # Extract collection hint (first non-operator word after intent, before where)
    collection_hint = _guess_collection_hint(text, ctx)

    words = text.split()
    used = [False] * len(words)

    # Mark operator/structural words so we never consume them as field parts
    for i, w in enumerate(words):
        if w.lower() in _NON_FIELD_WORDS or w in {">", ">=", "<", "<=", "=", "!="}:
            used[i] = True

    result_words = list(words)

    for window_size in (4, 3, 2):
        i = 0
        while i <= len(result_words) - window_size:
            # Skip if any word in the window is already consumed or structural
            window = result_words[i:i + window_size]
            window_lower = [w.lower() for w in window]

            if any(w in _NON_FIELD_WORDS or w in {">", ">=", "<", "<=", "=", "!="} for w in window_lower):
                i += 1
                continue

            candidate = "_".join(window_lower)
            name, score = ctx.resolve_field(candidate, collection_hint)
            if score >= 0.7:
                result_words[i:i + window_size] = [name]
                i += 1
            else:
                # Also try with trailing generic words stripped
                stripped = [w for w in window_lower if w not in
                            {"value", "values", "data", "info", "field", "column", "number", "amount"}]
                if stripped and len(stripped) < len(window_lower):
                    candidate2 = "_".join(stripped)
                    name2, score2 = ctx.resolve_field(candidate2, collection_hint)
                    if score2 >= 0.7:
                        result_words[i:i + window_size] = [name2]
                        i += 1
                        continue
                i += 1

    return " ".join(result_words)


def _guess_collection_hint(text: str, ctx: SchemaContext) -> str:
    """Extract the most likely collection from the text for field scoping."""
    words = text.split()
    for w in words:
        if w.lower() in _NON_FIELD_WORDS:
            continue
        fq, score = ctx.resolve_collection(w)
        if fq and score >= 0.7:
            return fq
    return ""


def _normalize_noun_chunks_fallback(text: str) -> str:
    """Fallback: use spaCy noun chunks when no schema is available."""
    nlp = _get_nlp()
    doc = nlp(text)

    if not doc.has_annotation("DEP"):
        return text

    replacements: list[tuple[int, int, str]] = []

    for chunk in doc.noun_chunks:
        words = [t for t in chunk if not t.is_stop and not t.is_punct]
        if len(words) < 2:
            continue
        if any(t.like_num for t in words):
            continue
        if any(not t.is_alpha for t in words):
            continue
        if any(t.text.lower() in _NON_FIELD_WORDS for t in words):
            continue

        joined = "_".join(t.text.lower() for t in words)
        replacements.append((chunk.start_char, chunk.end_char, joined))

    result = text
    for start, end, replacement in sorted(replacements, key=lambda x: -x[0]):
        result = result[:start] + replacement + result[end:]
    return result


# ---------------------------------------------------------------------------
# Pass 6: Strip copulas before operators
# ---------------------------------------------------------------------------

def _strip_copulas(text: str) -> str:
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w = words[i].lower().rstrip(".,!?")
        if w in _COPULAS:
            remaining = " ".join(words[i + 1:]).lower()
            is_before_op = False
            for op in _OPERATOR_PHRASES:
                if remaining.startswith(op):
                    is_before_op = True
                    break
            if not is_before_op and i + 1 < len(words):
                next_w = words[i + 1]
                if next_w in {">", ">=", "<", "<=", "=", "!="}:
                    is_before_op = True

            if is_before_op:
                i += 1
                continue

        result.append(words[i])
        i += 1

    return " ".join(result)


# ---------------------------------------------------------------------------
# Pass 7: Strip post-collection filler verbs
# ---------------------------------------------------------------------------

def _strip_post_collection_fillers(text: str) -> str:
    """Remove stray verbs between collection and where-clause.
    'orders were placed where...' → 'orders where...'
    Never removes comparison-critical words (operators, is, not, in).
    """
    words = text.split()
    if len(words) < 3:
        return text

    result = [words[0]]  # collection word
    hit_filter_start = False
    i = 1

    while i < len(words):
        w = words[i].lower()

        # Once we hit a where-marker or operator, everything stays
        if w in _WHERE_MARKERS or w in {">", ">=", "<", "<=", "=", "!="}:
            hit_filter_start = True
        if hit_filter_start:
            result.append(words[i])
            i += 1
            continue

        # Before filter starts: drop filler verbs, keep everything else
        if w in _POST_COLLECTION_FILLERS:
            i += 1
            continue

        result.append(words[i])
        i += 1

    return " ".join(result)


# ---------------------------------------------------------------------------
# Pass 8: Ensure where-marker before filter expressions
# ---------------------------------------------------------------------------

def _ensure_where_marker(text: str) -> str:
    """If there's a filter expression but no 'where' keyword, insert one.
    'products gross_sale_value greater than 1000' → 'products where gross_sale_value greater than 1000'
    """
    # If already has a where marker, skip
    if re.search(r'\b(where|with|whose|that|having)\b', text, re.I):
        return text

    words = text.split()
    if len(words) < 3:
        return text

    # Look for first word that could be a field_ref followed by an operator
    operators = {">", ">=", "<", "<=", "=", "!=",
                 "is", "equals", "equal", "greater", "less", "more",
                 "above", "below", "after", "before", "at", "contains",
                 "contain", "matches", "match", "like", "not", "in"}

    for i in range(1, len(words) - 1):
        word = words[i].lower()
        # If current word is NOT structural and next word IS an operator
        if word not in _NON_FIELD_WORDS and i + 1 < len(words):
            next_word = words[i + 1].lower()
            if next_word in operators or next_word in {">", ">=", "<", "<=", "=", "!="}:
                return " ".join(words[:i]) + " where " + " ".join(words[i:])

    return text
