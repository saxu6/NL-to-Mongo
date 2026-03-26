"""End-to-end NL query pipeline."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .compiler import compile
from .parser import parse
from .preprocessor import preprocess
from .resolver import SchemaContext, resolve


_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "full_schema.json"
_DEFAULT_PIPELINE: Optional["QueryPipeline"] = None


class QueryPipeline:
    """Pipeline:
    1) preprocessing  (prose → semi-structured text)
    2) parsing        (text → raw AST)
    3) resolution     (raw AST + schema → resolved AST)
    4) compilation    (resolved AST → MongoDB dict)
    """

    def __init__(self, schema: dict):
        if not isinstance(schema, dict) or not schema:
            raise ValueError("QueryPipeline requires a non-empty schema dictionary")

        self._schema = schema
        self._schema_ctx = SchemaContext(schema)

    def run(self, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")

        # 1) preprocessing
        preprocessed = preprocess(text.strip(), self._schema_ctx)

        # 2) parsing + 3) AST generation
        ast = parse(preprocessed)

        # 4) semantic resolution
        resolved_ast = resolve(ast, self._schema_ctx, strict=True)

        # 5) Mongo query compilation
        return compile(resolved_ast)


@lru_cache(maxsize=1)
def _load_default_schema() -> dict:
    with _DEFAULT_SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_pipeline() -> QueryPipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = QueryPipeline(_load_default_schema())
    return _DEFAULT_PIPELINE


def parse_query(text: str, schema: Optional[dict] = None) -> dict[str, Any]:
    """Parse natural language text to a MongoDB query dictionary.

    Primary API:
        parse_query(text: str) -> dict

    `schema` is optional for backward compatibility.
    """
    if schema is None:
        return _default_pipeline().run(text)
    return QueryPipeline(schema).run(text)
