"""Compiler: resolved AST -> MongoDB query dictionary."""

from __future__ import annotations

from typing import Any

from .ast_nodes import ComparisonNode, LimitNode, LogicalNode, QueryNode, QueryRoot, SortNode, ValueNode


class CompileError(ValueError):
    """Raised when AST nodes cannot be compiled to Mongo query syntax."""


def compile(ast: QueryRoot) -> dict[str, Any]:
    """Compile a resolved AST into a MongoDB query dictionary."""

    if not isinstance(ast, QueryRoot):
        raise TypeError("compile expects a QueryRoot AST")

    filter_nodes: list[QueryNode] = []
    sort_nodes: list[SortNode] = []
    limit_value: int | None = None

    for child in ast.children:
        if isinstance(child, (ComparisonNode, LogicalNode)):
            filter_nodes.append(child)
        elif isinstance(child, SortNode):
            sort_nodes.append(child)
        elif isinstance(child, LimitNode):
            limit_value = child.value

    filter_dict = _compile_filter_nodes(filter_nodes)
    sort_dict = _compile_sort(sort_nodes)

    operation = _intent_to_operation(ast.intent)
    result: dict[str, Any] = {
        "collection": ast.collection,
        "operation": operation,
        "filter": filter_dict,
    }

    if sort_dict:
        result["sort"] = sort_dict
    if limit_value is not None:
        result["limit"] = limit_value

    return result


def _intent_to_operation(intent: str) -> str:
    raw = (intent or "find").strip().lower()
    if raw == "count":
        return "countDocuments"
    if raw == "delete":
        return "deleteMany"
    if raw == "update":
        return "updateMany"
    if raw == "aggregate":
        return "aggregate"
    return "find"


def _compile_filter_nodes(nodes: list[QueryNode]) -> dict[str, Any]:
    if not nodes:
        return {}

    compiled = [_compile_filter_node(node) for node in nodes]
    if len(compiled) == 1:
        return compiled[0]
    return {"$and": compiled}


def _compile_filter_node(node: QueryNode) -> dict[str, Any]:
    if isinstance(node, ComparisonNode):
        return _compile_comparison(node)

    if isinstance(node, LogicalNode):
        compiled_children = [_compile_filter_node(child) for child in node.children]
        if not compiled_children:
            return {}
        if node.operator == "$not":
            if len(compiled_children) != 1:
                raise CompileError("$not logical node must have exactly one child")
            return {"$not": compiled_children[0]}
        if node.operator not in {"$and", "$or"}:
            raise CompileError(f"Unsupported logical operator '{node.operator}'")
        return {node.operator: compiled_children}

    raise CompileError(f"Unsupported filter node type '{type(node).__name__}'")


def _compile_comparison(node: ComparisonNode) -> dict[str, Any]:
    if not node.field:
        raise CompileError("ComparisonNode.field is required")

    value = _value_from_node(node.value)
    op = (node.operator or "$eq").strip()

    if op == "$eq":
        return {node.field: value}
    if op == "$regex":
        return {node.field: {"$regex": value, "$options": "i"}}
    if op in {"$in", "$nin"} and not isinstance(value, list):
        value = [value]

    return {node.field: {op: value}}


def _compile_sort(nodes: list[SortNode]) -> dict[str, int]:
    if not nodes:
        return {}

    sort_dict: dict[str, int] = {}
    for node in nodes:
        if not node.field:
            continue
        direction = 1 if node.direction == 1 else -1
        sort_dict[node.field] = direction
    return sort_dict


def _value_from_node(value: Any) -> Any:
    if isinstance(value, ValueNode):
        if value.typed_value is not None:
            return value.typed_value
        return value.raw
    return value

