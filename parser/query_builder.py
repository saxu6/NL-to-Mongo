from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .ast_nodes import (
    ASTNode,
    CollectionRef,
    ComparisonOp,
    FilterExpression,
    LogicalKind,
    LogicalOp,
    LimitNode,
    ProjectionNode,
    QueryNode,
    SortNode,
)
from .grammar import GrammarResult, QueryIntent

_inference_mod = None
_model_registry = None

_CACHE_PREFIX = "_rl"
_MODULE_MANIFEST = {
    "models":            f"{_CACHE_PREFIX}.models",
    "test_with_schema":  f"{_CACHE_PREFIX}.test_with_schema",
    "connection":        f"{_CACHE_PREFIX}.connection",
    "logic":             f"{_CACHE_PREFIX}.logic",
    "__init__":          f"{_CACHE_PREFIX}.__init__",
}


def _resolve_cache_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")


def _find_pyc(cache_dir: str, module_key: str) -> Optional[str]:
    prefix = _MODULE_MANIFEST.get(module_key, f"{_CACHE_PREFIX}.{module_key}")
    version_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    expected = f"{prefix}.{version_tag}.pyc"
    pyc_path = os.path.join(cache_dir, expected)
    if os.path.isfile(pyc_path):
        return pyc_path
    return None


def _load_pyc_module(pyc_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, pyc_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {pyc_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_inference_modules():
    cache_dir = _resolve_cache_dir()

    load_order = ["__init__", "connection", "models", "logic", "test_with_schema"]
    loaded = {}

    pkg_name = "_rl_runtime"
    if pkg_name not in sys.modules:
        import types
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [cache_dir]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    for mod_key in load_order:
        pyc_path = _find_pyc(cache_dir, mod_key)
        if pyc_path is None:
            continue

        runtime_name = f"{pkg_name}.{mod_key}"
        legacy_name = f"OllamaCon.{mod_key}"

        try:
            mod = _load_pyc_module(pyc_path, runtime_name)
            sys.modules[legacy_name] = mod
            loaded[mod_key] = mod
        except Exception:
            pass

    if "OllamaCon" not in sys.modules:
        import types
        pkg = types.ModuleType("OllamaCon")
        pkg.__path__ = [cache_dir]
        pkg.__package__ = "OllamaCon"
        sys.modules["OllamaCon"] = pkg

    return loaded


def _get_inference_module():
    global _inference_mod
    if _inference_mod is None:
        loaded = _bootstrap_inference_modules()
        if "test_with_schema" in loaded:
            _inference_mod = loaded["test_with_schema"]
        else:
            _inference_mod = importlib.import_module(
                ".".join(["OllamaCon", "test_with_schema"])
            )
    return _inference_mod


def _get_model_registry():
    global _model_registry
    if _model_registry is None:
        if "OllamaCon.models" in sys.modules:
            _model_registry = sys.modules["OllamaCon.models"]
        else:
            _model_registry = importlib.import_module(
                ".".join(["OllamaCon", "models"])
            )
    return _model_registry


class MongoQueryBuilder:

    def __init__(self, schema: dict):
        self._schema = schema
        self._collection_map = self._build_collection_map(schema)

    def build(
        self,
        ast: QueryNode,
        grammar_result: GrammarResult,
        raw_query: str,
    ) -> Dict[str, Any]:
        collection = self._resolve_collection(ast.collection)
        ast_filter = self._build_filter_pipeline(ast.filter_expr)

        query_context = {
            "raw_query": raw_query,
            "ast_intent": ast.intent,
            "grammar_intent": grammar_result.primary_intent.name,
            "grammar_confidence": grammar_result.confidence,
            "ast_collection": collection,
            "ast_filter": ast_filter,
            "ast_has_filter": ast.has_filter,
            "matched_rules": grammar_result.matched_rules,
        }
        resolved = self._resolve_semantic_bindings(query_context, self._schema)

        optimised = self._optimise_query(resolved)

        validated = self._validate_against_schema(optimised)

        return validated

    def _resolve_collection(self, ref: Optional[CollectionRef]) -> str:
        if ref and ref.collection:
            for db_name, collections in self._collection_map.items():
                if ref.collection in collections:
                    return f"{db_name}.{ref.collection}"
                for coll in collections:
                    if coll.lower() == ref.collection.lower():
                        return f"{db_name}.{coll}"

        for db_name, collections in self._collection_map.items():
            if collections:
                return f"{db_name}.{collections[0]}"
        return ""

    def _build_filter_pipeline(
        self, filter_expr: Optional[FilterExpression]
    ) -> Dict[str, Any]:
        if filter_expr is None or filter_expr.root is None:
            return {}
        return self._compile_node(filter_expr.root)

    def _compile_node(self, node: ASTNode) -> Dict[str, Any]:
        if isinstance(node, ComparisonOp):
            return node.to_mongo_fragment()

        if isinstance(node, LogicalOp):
            children = [self._compile_node(c) for c in node.children]
            if node.kind == LogicalKind.AND:
                merged: Dict[str, Any] = {}
                for child in children:
                    merged.update(child)
                return merged
            elif node.kind == LogicalKind.OR:
                return {"$or": children}
            elif node.kind == LogicalKind.NOT:
                if children:
                    return {"$not": children[0]}
                return {}

        return {}

    def _resolve_semantic_bindings(
        self,
        query_context: Dict[str, Any],
        schema: dict,
    ) -> Dict[str, Any]:
        raw_query = query_context.get("raw_query", "")

        try:
            inference = _get_inference_module()
            resolved_query = inference.test_query_with_schema(raw_query, schema)

            if resolved_query is not None:
                return resolved_query

        except Exception:
            pass

        return self._build_fallback_query(query_context)

    def _build_fallback_query(
        self, query_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "collection": query_context.get("ast_collection", ""),
            "filter": query_context.get("ast_filter", {}),
            "projection": {},
            "sort": {},
            "limit": None,
        }

    def _optimise_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        optimised = dict(query)

        for key in ("projection", "sort"):
            if key in optimised and not optimised[key]:
                optimised[key] = {}

        if optimised.get("limit") is None:
            optimised["limit"] = None

        filt = optimised.get("filter", {})
        if "$and" in filt and len(filt["$and"]) == 1:
            optimised["filter"] = filt["$and"][0]

        return optimised

    def _validate_against_schema(self, query: Dict[str, Any]) -> Dict[str, Any]:
        collection_str = query.get("collection", "")
        parts = collection_str.split(".") if collection_str else []

        if len(parts) == 2:
            db_name, coll_name = parts
            schema_fields = self._extract_fields_for_collection(
                db_name, coll_name
            )

        return query

    @staticmethod
    def _build_collection_map(schema: dict) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for db_name, db_data in schema.items():
            if isinstance(db_data, dict):
                mapping[db_name] = [
                    coll for coll in db_data.keys()
                    if isinstance(db_data[coll], dict)
                ]
        return mapping

    def _extract_fields_for_collection(
        self, db_name: str, coll_name: str
    ) -> set:
        try:
            coll_data = self._schema[db_name][coll_name]
            obj = coll_data.get("object", {})
            return set(obj.keys())
        except (KeyError, AttributeError):
            return set()
