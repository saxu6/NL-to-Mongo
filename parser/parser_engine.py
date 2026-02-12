from __future__ import annotations

import time
from typing import Any, Dict, Optional, Set

from .tokenizer import Tokenizer, extract_schema_fields
from .ast_nodes import ASTBuilder, QueryNode
from .grammar import GrammarMatcher, GrammarResult
from .query_builder import MongoQueryBuilder


class NLQueryParser:

    def __init__(self, schema: dict, *, enable_logging: bool = True):
        self._schema = schema
        self._enable_logging = enable_logging

        self._schema_fields: Set[str] = extract_schema_fields(schema)

        self._tokenizer = Tokenizer(schema_fields=self._schema_fields)
        self._ast_builder = ASTBuilder()
        self._grammar_matcher = GrammarMatcher()
        self._query_builder = MongoQueryBuilder(schema)

        self._last_tokens = None
        self._last_ast: Optional[QueryNode] = None
        self._last_grammar: Optional[GrammarResult] = None
        self._last_result: Optional[Dict[str, Any]] = None

    def parse(self, nl_query: str) -> Dict[str, Any]:
        if not nl_query or not nl_query.strip():
            raise ValueError("Cannot parse an empty query string")

        query = nl_query.strip()

        try:
            tokens = self._tokenizer.tokenize(query)
            self._last_tokens = tokens

            ast = self._ast_builder.build(tokens, raw_query=query)
            self._last_ast = ast

            grammar_result = self._grammar_matcher.match(query, tokens)
            self._last_grammar = grammar_result

            result = self._query_builder.build(ast, grammar_result, query)
            self._last_result = result

            return result

        except Exception as exc:
            raise RuntimeError(
                f"Parser pipeline failed: {exc}"
            ) from exc

    def get_parse_tree(self) -> Optional[QueryNode]:
        return self._last_ast

    def get_grammar_result(self) -> Optional[GrammarResult]:
        return self._last_grammar

    def get_last_tokens(self):
        return self._last_tokens

    @property
    def schema_fields(self) -> Set[str]:
        return set(self._schema_fields)

    @staticmethod
    def _summarise_token_types(tokens) -> dict:
        counts: dict = {}
        for tok in tokens:
            name = tok.type.name
            counts[name] = counts.get(name, 0) + 1
        return counts
