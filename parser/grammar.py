from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .tokenizer import Token, TokenType


class QueryIntent(Enum):
    FIND      = auto()
    AGGREGATE = auto()
    SORT      = auto()
    LIMIT     = auto()
    EXISTS    = auto()
    RANGE     = auto()
    UNKNOWN   = auto()


@dataclass
class GrammarRule:
    name: str
    intent: QueryIntent
    pattern: re.Pattern
    required_tokens: set = field(default_factory=set)
    weight: float = 0.5


_RULES: List[GrammarRule] = [
    GrammarRule(
        name="find_basic",
        intent=QueryIntent.FIND,
        pattern=re.compile(
            r"\b(?:find|get|show|list|retrieve|fetch|display|search)\b", re.I
        ),
        weight=0.6,
    ),
    GrammarRule(
        name="find_where",
        intent=QueryIntent.FIND,
        pattern=re.compile(
            r"\b(?:find|get|show)\b.*\b(?:where|with|having)\b", re.I
        ),
        required_tokens={TokenType.KEYWORD, TokenType.OPERATOR},
        weight=0.8,
    ),
    GrammarRule(
        name="find_equality",
        intent=QueryIntent.FIND,
        pattern=re.compile(
            r"\bwhere\b.*\b(?:is|equals?|=)\b", re.I
        ),
        weight=0.7,
    ),
    GrammarRule(
        name="aggregate_count",
        intent=QueryIntent.AGGREGATE,
        pattern=re.compile(r"\b(?:count|how\s+many)\b", re.I),
        weight=0.8,
    ),
    GrammarRule(
        name="aggregate_math",
        intent=QueryIntent.AGGREGATE,
        pattern=re.compile(r"\b(?:sum|average|avg|total|mean|max|min)\b", re.I),
        weight=0.8,
    ),
    GrammarRule(
        name="aggregate_group",
        intent=QueryIntent.AGGREGATE,
        pattern=re.compile(r"\bgroup\s+by\b", re.I),
        weight=0.9,
    ),
    GrammarRule(
        name="sort_explicit",
        intent=QueryIntent.SORT,
        pattern=re.compile(
            r"\b(?:sort|order)\s+(?:by|ascending|descending|asc|desc)\b", re.I
        ),
        weight=0.9,
    ),
    GrammarRule(
        name="sort_implicit",
        intent=QueryIntent.SORT,
        pattern=re.compile(
            r"\b(?:highest|lowest|newest|oldest|latest|earliest)\b", re.I
        ),
        weight=0.6,
    ),
    GrammarRule(
        name="limit_explicit",
        intent=QueryIntent.LIMIT,
        pattern=re.compile(r"\b(?:limit|top|first|last)\s+\d+\b", re.I),
        weight=0.9,
    ),
    GrammarRule(
        name="limit_implicit",
        intent=QueryIntent.LIMIT,
        pattern=re.compile(r"\b(?:only|just)\s+\d+\b", re.I),
        weight=0.5,
    ),
    GrammarRule(
        name="exists_null",
        intent=QueryIntent.EXISTS,
        pattern=re.compile(r"\b(?:is\s+null|is\s+not\s+null|exists?)\b", re.I),
        weight=0.8,
    ),
    GrammarRule(
        name="range_between",
        intent=QueryIntent.RANGE,
        pattern=re.compile(r"\bbetween\b.*\band\b", re.I),
        weight=0.9,
    ),
    GrammarRule(
        name="range_comparison",
        intent=QueryIntent.RANGE,
        pattern=re.compile(
            r"\b(?:greater|less|more|fewer|above|below|over|under)\b", re.I
        ),
        weight=0.6,
    ),
]


@dataclass
class GrammarResult:
    primary_intent: QueryIntent = QueryIntent.UNKNOWN
    confidence: float = 0.0
    matched_rules: List[str] = field(default_factory=list)
    secondary_intents: List[QueryIntent] = field(default_factory=list)
    intent_scores: Dict[QueryIntent, float] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.6


class GrammarMatcher:

    def __init__(self, rules: Optional[List[GrammarRule]] = None):
        self._rules = rules if rules is not None else list(_RULES)

    def match(self, raw_query: str, tokens: List[Token]) -> GrammarResult:
        token_types = {t.type for t in tokens}
        scores: Dict[QueryIntent, float] = {}
        matched: List[str] = []

        for rule in self._rules:
            if not self._evaluate_rule(rule, raw_query, token_types):
                continue

            matched.append(rule.name)
            current = scores.get(rule.intent, 0.0)
            scores[rule.intent] = current + rule.weight

        if not scores:
            return GrammarResult(
                primary_intent=QueryIntent.FIND,
                confidence=0.3,
                matched_rules=[],
                intent_scores={},
            )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        primary, top_score = ranked[0]

        max_possible = sum(r.weight for r in self._rules if r.intent == primary)
        confidence = min(top_score / max(max_possible, 1.0), 1.0)

        secondary = [intent for intent, _ in ranked[1:]]

        return GrammarResult(
            primary_intent=primary,
            confidence=round(confidence, 4),
            matched_rules=matched,
            secondary_intents=secondary,
            intent_scores={k: round(v, 4) for k, v in scores.items()},
        )

    @staticmethod
    def _evaluate_rule(
        rule: GrammarRule,
        raw_query: str,
        token_types: set,
    ) -> bool:
        if not rule.pattern.search(raw_query):
            return False
        if rule.required_tokens and not rule.required_tokens.issubset(token_types):
            return False
        return True
