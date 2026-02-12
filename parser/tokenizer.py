from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Set


class TokenType(Enum):
    KEYWORD     = auto()
    IDENTIFIER  = auto()
    OPERATOR    = auto()
    LITERAL     = auto()
    CONNECTOR   = auto()
    PUNCTUATION = auto()
    UNKNOWN     = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    raw: str
    position: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, pos={self.position})"


_KEYWORDS: Set[str] = {
    "find", "get", "show", "list", "retrieve", "fetch",
    "display", "select", "return", "query", "search",
    "count", "sum", "average", "total", "aggregate",
    "group", "sort", "order", "limit", "skip", "top",
    "first", "last", "all", "each", "every", "any",
    "where", "having", "between", "like", "exists",
    "update", "delete", "remove", "insert", "create",
}

_CONNECTORS: Set[str] = {
    "and", "or", "not", "but", "with", "without",
    "in", "from", "of", "by", "the", "is", "are",
    "has", "have", "to", "for", "than", "that",
    "which", "whose", "as",
}

_MULTI_WORD_OPERATORS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnot\s+equal\s+to\b",   re.I), "$ne"),
    (re.compile(r"\bnot\s+equals?\b",       re.I), "$ne"),
    (re.compile(r"\bgreater\s+than\s+or\s+equal(?:\s+to)?\b", re.I), "$gte"),
    (re.compile(r"\bless\s+than\s+or\s+equal(?:\s+to)?\b",    re.I), "$lte"),
    (re.compile(r"\bgreater\s+than\b",      re.I), "$gt"),
    (re.compile(r"\bmore\s+than\b",         re.I), "$gt"),
    (re.compile(r"\bless\s+than\b",         re.I), "$lt"),
    (re.compile(r"\bfewer\s+than\b",        re.I), "$lt"),
    (re.compile(r"\bequal\s+to\b",          re.I), "$eq"),
    (re.compile(r"\bcontains?\b",           re.I), "$regex"),
    (re.compile(r"\bmatche?s?\b",           re.I), "$regex"),
    (re.compile(r"\bstarts?\s+with\b",      re.I), "$regex"),
    (re.compile(r"\bends?\s+with\b",        re.I), "$regex"),
    (re.compile(r"\bnot\s+in\b",            re.I), "$nin"),
]

_SINGLE_WORD_OPERATORS: dict[str, str] = {
    "equals":  "$eq",
    "equal":   "$eq",
    "above":   "$gt",
    "below":   "$lt",
    "over":    "$gt",
    "under":   "$lt",
    "after":   "$gt",
    "before":  "$lt",
    "except":  "$ne",
    "null":    "$exists",
}

_PUNCTUATION: Set[str] = {"'", '"', ",", ".", "(", ")", "[", "]", "{", "}"}

_NUM_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)$")


class Tokenizer:

    def __init__(self, schema_fields: Optional[Set[str]] = None):
        self._schema_fields: Set[str] = schema_fields or set()
        self._tokens: List[Token] = []
        self._source: str = ""

    def tokenize(self, source: str) -> List[Token]:
        self._source = source
        self._tokens = []

        working = self._collapse_multi_word_operators(source)
        raw_units = self._split_into_units(working)

        for raw, pos in raw_units:
            token = self._classify(raw, pos)
            self._tokens.append(token)

        return list(self._tokens)

    @property
    def tokens(self) -> List[Token]:
        return list(self._tokens)

    def _collapse_multi_word_operators(self, text: str) -> str:
        result = text
        for pattern, mongo_op in _MULTI_WORD_OPERATORS:
            result = pattern.sub(f"__OP_{mongo_op}__", result)
        return result

    def _split_into_units(self, text: str) -> List[tuple[str, int]]:
        units: List[tuple[str, int]] = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]

            if ch.isspace():
                i += 1
                continue

            if ch in ('"', "'"):
                close = text.find(ch, i + 1)
                if close == -1:
                    close = n
                literal = text[i + 1 : close]
                units.append((literal, i))
                i = close + 1
                continue

            if ch in _PUNCTUATION:
                units.append((ch, i))
                i += 1
                continue

            j = i
            while j < n and not text[j].isspace() and text[j] not in _PUNCTUATION:
                j += 1
            word = text[i:j]
            units.append((word, i))
            i = j

        return units

    def _classify(self, raw: str, position: int) -> Token:
        if raw.startswith("__OP_") and raw.endswith("__"):
            op = raw[5:-2]
            return Token(TokenType.OPERATOR, op, raw, position)

        lower = raw.lower().strip("'\"")

        if raw in _PUNCTUATION:
            return Token(TokenType.PUNCTUATION, raw, raw, position)

        if _NUM_RE.match(raw):
            return Token(TokenType.LITERAL, raw, raw, position)

        if lower in ("true", "false", "null", "none", "nil"):
            return Token(TokenType.LITERAL, lower, raw, position)

        if lower in _SINGLE_WORD_OPERATORS:
            return Token(
                TokenType.OPERATOR,
                _SINGLE_WORD_OPERATORS[lower],
                raw,
                position,
            )

        if lower in _KEYWORDS:
            return Token(TokenType.KEYWORD, lower, raw, position)

        if lower in _CONNECTORS:
            return Token(TokenType.CONNECTOR, lower, raw, position)

        if lower in self._schema_fields or raw in self._schema_fields:
            return Token(TokenType.IDENTIFIER, raw, raw, position)

        return Token(TokenType.LITERAL, raw, raw, position)


def extract_schema_fields(schema: dict) -> Set[str]:
    fields: Set[str] = set()

    def _walk(obj: dict, depth: int = 0):
        for key, val in obj.items():
            if depth >= 2:
                fields.add(key)
            if isinstance(val, dict):
                if "object" in val and isinstance(val["object"], dict):
                    _walk(val["object"], depth + 1)
                else:
                    _walk(val, depth + 1)

    _walk(schema)
    return fields
