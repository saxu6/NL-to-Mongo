"""Semantic resolver for AST nodes.

Raw AST in, schema-resolved AST out.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

from rapidfuzz import fuzz

from .ast_nodes import ComparisonNode, QueryNode, QueryRoot, ValueNode


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _singularize(word: str) -> str:
    w = word.strip().lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _field_priority(field_name: str) -> int:
    f = field_name.lower()
    if f.endswith("_id"):
        return 5
    if f.endswith("_name"):
        return 4
    if f.endswith("_code"):
        return 3
    if f.endswith("_type"):
        return 2
    return 1


@dataclass
class FieldInfo:
    name: str
    type: str
    collections: set[str] = field(default_factory=set)


@dataclass
class CollectionInfo:
    fq_name: str
    db: str
    name: str
    fields: set[str] = field(default_factory=set)


class ResolutionError(ValueError):
    """Raised when schema resolution fails in strict mode."""


class SchemaContext:
    """Schema-derived lookup context used by the resolver."""

    def __init__(self, schema: dict):
        self.raw_schema = schema if isinstance(schema, dict) else {}
        self.fields: dict[str, FieldInfo] = {}
        self.collections: dict[str, CollectionInfo] = {}
        self.concept_map: dict[str, str] = {}
        self._build_indexes()

    def resolve_collection(self, raw_name: str) -> tuple[str, float]:
        if not raw_name:
            return "", 0.0

        name = _normalize_text(raw_name)
        singular = _singularize(name)

        if name in self.collections:
            return name, 1.0

        for fq_name, info in self.collections.items():
            if name == info.name.lower() or singular == _singularize(info.name):
                return fq_name, 1.0

        best = ""
        best_score = 0.0
        for fq_name, info in self.collections.items():
            score = fuzz.ratio(name, info.name.lower()) / 100.0
            if score > best_score:
                best_score = score
                best = fq_name

        if best and best_score >= 0.72:
            return best, round(best_score, 3)
        return "", 0.0

    def resolve_field(self, raw_field: str, collection_hint: str = "") -> tuple[str, float]:
        if not raw_field:
            return "", 0.0

        candidates = self._candidate_fields(collection_hint)
        if not candidates:
            return raw_field, 0.0

        field = raw_field.strip()
        lowered = field.lower()
        singular = _singularize(lowered)

        if field in candidates:
            return field, 1.0

        lowered_map = {f.lower(): f for f in candidates}
        if lowered in lowered_map:
            return lowered_map[lowered], 0.98

        concept_key = _normalize_text(lowered).replace(" ", "_")
        concept_match = self.concept_map.get(concept_key)
        if concept_match and concept_match in candidates:
            return concept_match, 0.93

        singular_key = _normalize_text(singular).replace(" ", "_")
        concept_match = self.concept_map.get(singular_key)
        if concept_match and concept_match in candidates:
            return concept_match, 0.9

        best = ""
        best_score = 0.0
        for cand in candidates:
            ratio = fuzz.ratio(lowered, cand.lower()) / 100.0
            partial = fuzz.partial_ratio(lowered, cand.lower()) / 100.0
            score = max(ratio, partial * 0.95)
            score += _field_priority(cand) * 0.01
            if score > best_score:
                best_score = score
                best = cand

        if best and best_score >= 0.7:
            return best, round(min(best_score, 1.0), 3)
        return raw_field, round(best_score, 3)

    def infer_collection(self, resolved_fields: Iterable[str]) -> tuple[str, float]:
        fields = [f for f in resolved_fields if f]
        if not fields:
            return "", 0.0

        scores: dict[str, int] = {fq: 0 for fq in self.collections}
        for field_name in fields:
            info = self.fields.get(field_name)
            if not info:
                continue
            for fq_name in info.collections:
                scores[fq_name] = scores.get(fq_name, 0) + 1

        best = ""
        best_hits = 0
        for fq_name, hits in scores.items():
            if hits > best_hits:
                best_hits = hits
                best = fq_name

        if not best:
            return "", 0.0
        return best, round(best_hits / max(len(fields), 1), 3)

    def field_type(self, field_name: str) -> str:
        info = self.fields.get(field_name)
        return info.type if info else "string"

    def has_collection(self, fq_name: str) -> bool:
        return fq_name in self.collections

    def collection_has_field(self, fq_name: str, field_name: str) -> bool:
        info = self.collections.get(fq_name)
        if not info:
            return False
        return field_name in info.fields

    def _candidate_fields(self, collection_hint: str) -> set[str]:
        if collection_hint and collection_hint in self.collections:
            return set(self.collections[collection_hint].fields)
        return set(self.fields.keys())

    def _build_indexes(self) -> None:
        for db_name, db_data in self.raw_schema.items():
            if not isinstance(db_data, dict):
                continue
            for coll_name, coll_data in db_data.items():
                if not isinstance(coll_data, dict):
                    continue

                fq_name = f"{db_name}.{coll_name}"
                obj = coll_data.get("object", coll_data)
                if not isinstance(obj, dict):
                    continue

                coll_info = CollectionInfo(
                    fq_name=fq_name,
                    db=db_name,
                    name=coll_name,
                    fields=set(obj.keys()),
                )
                self.collections[fq_name] = coll_info

                for field_name, field_data in obj.items():
                    field_type = "string"
                    if isinstance(field_data, dict):
                        field_type = str(field_data.get("type", "string")).lower()

                    info = self.fields.get(field_name)
                    if info is None:
                        info = FieldInfo(
                            name=field_name,
                            type=field_type,
                            collections=set(),
                        )
                        self.fields[field_name] = info
                    elif info.type == "string" and field_type != "string":
                        info.type = field_type

                    info.collections.add(fq_name)

        self.concept_map = self._build_concept_map()

    def _build_concept_map(self) -> dict[str, str]:
        concept_candidates: dict[str, list[str]] = {}

        for field_name in self.fields:
            low = field_name.lower()
            concepts = {low}

            parts = [p for p in low.replace(".", "_").split("_") if p]
            if parts:
                concepts.update(parts)
                concepts.update(_singularize(p) for p in parts)
                concepts.add("_".join(parts))
                concepts.add("".join(parts))

                if parts[-1] in {"id", "name", "code", "type"} and len(parts) >= 2:
                    base = "_".join(parts[:-1])
                    concepts.add(base)
                    concepts.add(_singularize(base))
                    concepts.add(parts[0])

            for concept in concepts:
                if not concept:
                    continue
                concept_key = _normalize_text(concept).replace(" ", "_")
                concept_candidates.setdefault(concept_key, []).append(field_name)

        resolved: dict[str, str] = {}
        for concept, choices in concept_candidates.items():
            choices = list(set(choices))
            choices.sort(key=lambda name: (_field_priority(name), -len(name)), reverse=True)
            resolved[concept] = choices[0]
        return resolved


_OPERATOR_MAP = {
    "=": "$eq",
    "==": "$eq",
    "eq": "$eq",
    "is": "$eq",
    "equals": "$eq",
    "equal": "$eq",
    "equal to": "$eq",
    "$eq": "$eq",
    "!=": "$ne",
    "<>": "$ne",
    "ne": "$ne",
    "not equal": "$ne",
    "not equals": "$ne",
    "is not": "$ne",
    "$ne": "$ne",
    ">": "$gt",
    "gt": "$gt",
    "greater than": "$gt",
    "above": "$gt",
    "after": "$gt",
    "$gt": "$gt",
    ">=": "$gte",
    "gte": "$gte",
    "at least": "$gte",
    "$gte": "$gte",
    "<": "$lt",
    "lt": "$lt",
    "less than": "$lt",
    "below": "$lt",
    "before": "$lt",
    "$lt": "$lt",
    "<=": "$lte",
    "lte": "$lte",
    "at most": "$lte",
    "$lte": "$lte",
    "contains": "$regex",
    "match": "$regex",
    "matches": "$regex",
    "like": "$regex",
    "$regex": "$regex",
    "in": "$in",
    "$in": "$in",
    "not in": "$nin",
    "$nin": "$nin",
}


def resolve(ast: QueryRoot, ctx: SchemaContext, *, strict: bool = True) -> QueryRoot:
    """Resolve AST fields/collection/operators/values against the schema."""

    if not isinstance(ast, QueryRoot):
        raise TypeError("resolve expects a QueryRoot AST")

    resolved = deepcopy(ast)
    warnings: list[str] = []
    resolved_fields: list[str] = []

    collection_score = 0.0
    if resolved.collection:
        fq_name, collection_score = ctx.resolve_collection(resolved.collection)
        if not fq_name:
            raise ResolutionError(f"Unknown collection hint '{resolved.collection}'")
        resolved.collection = fq_name

    for node in _walk(resolved):
        if not isinstance(node, ComparisonNode):
            continue

        node.operator = _normalize_operator(node.operator)

        field_name, score = ctx.resolve_field(node.field, resolved.collection)
        node.meta["confidence"] = score
        if not field_name or (score == 0.0 and strict):
            raise ResolutionError(f"Could not resolve field '{node.field}'")

        node.field = field_name
        resolved_fields.append(field_name)
        _normalize_value_node(node, ctx, strict=strict)

    if not resolved.collection:
        inferred_collection, collection_score = ctx.infer_collection(resolved_fields)
        if inferred_collection:
            resolved.collection = inferred_collection
        elif strict:
            raise ResolutionError("Could not infer collection from resolved fields")

    if resolved.collection and not ctx.has_collection(resolved.collection):
        raise ResolutionError(f"Resolved collection '{resolved.collection}' is not in schema")

    if resolved.collection:
        for node in _walk(resolved):
            if not isinstance(node, ComparisonNode):
                continue
            if not ctx.collection_has_field(resolved.collection, node.field):
                msg = (
                    f"Field '{node.field}' does not exist in collection "
                    f"'{resolved.collection}'"
                )
                if strict:
                    raise ResolutionError(msg)
                warnings.append(msg)

    if collection_score:
        resolved.meta["collection_confidence"] = collection_score
    if warnings:
        resolved.meta["warnings"] = warnings

    return resolved


def _walk(node: QueryNode) -> Iterable[QueryNode]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _normalize_operator(operator: str) -> str:
    raw = str(operator).strip()
    key = _normalize_text(raw)
    return _OPERATOR_MAP.get(key, raw)


def _normalize_value_node(node: ComparisonNode, ctx: SchemaContext, *, strict: bool) -> None:
    value_node = _ensure_value_node(node)
    raw = value_node.raw

    target_type = ctx.field_type(node.field)
    typed_value, inferred_type = _coerce_value(raw, target_type)
    if typed_value is None and raw not in {"", "null", "None", "none"} and strict:
        raise ResolutionError(
            f"Could not normalize value '{raw}' for field '{node.field}' "
            f"(expected {target_type})"
        )

    value_node.typed_value = typed_value
    value_node.inferred_type = inferred_type


def _ensure_value_node(node: ComparisonNode) -> ValueNode:
    if isinstance(node.value, ValueNode):
        value_node = node.value
    else:
        value_node = ValueNode(raw="" if node.value is None else str(node.value))
        node.value = value_node

    if value_node not in node.children:
        node.children.append(value_node)
    return value_node


def _coerce_value(raw_value: Any, target_type: str) -> tuple[Any, str]:
    if raw_value is None:
        return None, "null"

    if isinstance(raw_value, (int, float, bool)):
        if isinstance(raw_value, bool):
            return raw_value, "bool"
        if isinstance(raw_value, int):
            return raw_value, "int"
        return raw_value, "float"

    raw = str(raw_value).strip()
    if raw.lower() in {"null", "none", "nil"}:
        return None, "null"

    norm_type = (target_type or "string").lower()

    if norm_type in {"bool", "boolean"}:
        val = raw.lower()
        if val in {"true", "1", "yes", "y"}:
            return True, "bool"
        if val in {"false", "0", "no", "n"}:
            return False, "bool"
        return None, "bool"

    if norm_type in {"int", "integer", "long"}:
        try:
            return int(float(raw)), "int"
        except ValueError:
            return None, "int"

    if norm_type in {"float", "double", "decimal"}:
        try:
            return float(raw), "float"
        except ValueError:
            dt = _parse_date(raw)
            if dt is not None:
                return dt.timestamp(), "float"
            return None, "float"

    if norm_type in {"date", "datetime"}:
        dt = _parse_date(raw)
        if dt is not None:
            return dt.isoformat(), "date"
        return None, "date"

    return raw, "string"


def _parse_date(raw: str) -> Optional[datetime]:
    text = raw.strip()
    if not text:
        return None

    if len(text) == 10:
        try:
            return datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

