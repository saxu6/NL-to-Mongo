"""AST node definitions for the NL query parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryNode:
    """Base class for all AST nodes."""

    children: list["QueryNode"] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRoot(QueryNode):
    """Top-level query node."""

    intent: str = "find"  # find | count | aggregate | delete | update
    collection: str = ""  # unresolved collection hint from parser text
    raw_text: str = ""


@dataclass
class ComparisonNode(QueryNode):
    """Field comparison node."""

    field: str = ""
    operator: str = "$eq"  # $eq $gt $gte $lt $lte $ne $in $regex
    value: Any = None


@dataclass
class LogicalNode(QueryNode):
    """Boolean expression node."""

    operator: str = "$and"  # $and | $or | $not


@dataclass
class GroupNode(QueryNode):
    """Aggregation group node."""

    field: str = ""
    accumulator: str = "$count"  # $sum $avg $min $max $count


@dataclass
class SortNode(QueryNode):
    """Sort node."""

    field: str = ""
    direction: int = -1  # 1 = asc, -1 = desc


@dataclass
class LimitNode(QueryNode):
    """Limit node."""

    value: int = 0


@dataclass
class ProjectionNode(QueryNode):
    """Projection node."""

    fields: list[str] = field(default_factory=list)
    mode: str = "include"  # include | exclude


@dataclass
class UpdateNode(QueryNode):
    """Update operation node."""

    field: str = ""
    value: Any = None


@dataclass
class ValueNode(QueryNode):
    """Leaf value node."""

    raw: str = ""
    typed_value: Any = None
    inferred_type: str = ""  # string | float | int | date | bool

