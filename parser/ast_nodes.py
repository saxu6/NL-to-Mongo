from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional

from .tokenizer import Token, TokenType


class NodeType(Enum):
    QUERY       = auto()
    COLLECTION  = auto()
    FILTER      = auto()
    COMPARISON  = auto()
    LOGICAL     = auto()
    PROJECTION  = auto()
    SORT        = auto()
    LIMIT       = auto()


class LogicalKind(Enum):
    AND = "AND"
    OR  = "OR"
    NOT = "NOT"


@dataclass
class ASTNode:
    node_type: NodeType = NodeType.QUERY

    def accept(self, visitor):
        method = f"visit_{self.node_type.name.lower()}"
        handler = getattr(visitor, method, None)
        if handler:
            return handler(self)
        return None


@dataclass
class CollectionRef(ASTNode):
    database: Optional[str] = None
    collection: Optional[str] = None
    confidence: float = 0.0

    def __post_init__(self):
        self.node_type = NodeType.COLLECTION

    @property
    def fully_qualified(self) -> str:
        if self.database and self.collection:
            return f"{self.database}.{self.collection}"
        return self.collection or ""


@dataclass
class ComparisonOp(ASTNode):
    field_name: str = ""
    operator: str = "$eq"
    value: Any = None

    def __post_init__(self):
        self.node_type = NodeType.COMPARISON

    def to_mongo_fragment(self) -> dict:
        if self.operator == "$eq":
            return {self.field_name: self.value}
        return {self.field_name: {self.operator: self.value}}


@dataclass
class LogicalOp(ASTNode):
    kind: LogicalKind = LogicalKind.AND
    children: List[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = NodeType.LOGICAL


@dataclass
class FilterExpression(ASTNode):
    root: Optional[ASTNode] = None

    def __post_init__(self):
        self.node_type = NodeType.FILTER


@dataclass
class ProjectionNode(ASTNode):
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = NodeType.PROJECTION

    def to_mongo_projection(self) -> dict:
        proj = {}
        for f in self.include:
            proj[f] = 1
        for f in self.exclude:
            proj[f] = 0
        return proj


@dataclass
class SortNode(ASTNode):
    fields: List[tuple[str, int]] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = NodeType.SORT

    def to_mongo_sort(self) -> dict:
        return {f: d for f, d in self.fields}


@dataclass
class LimitNode(ASTNode):
    count: int = 0

    def __post_init__(self):
        self.node_type = NodeType.LIMIT


@dataclass
class QueryNode(ASTNode):
    intent: str = "find"
    collection: Optional[CollectionRef] = None
    filter_expr: Optional[FilterExpression] = None
    projection: Optional[ProjectionNode] = None
    sort: Optional[SortNode] = None
    limit: Optional[LimitNode] = None
    raw_query: str = ""

    def __post_init__(self):
        self.node_type = NodeType.QUERY

    @property
    def has_filter(self) -> bool:
        return self.filter_expr is not None and self.filter_expr.root is not None

    @property
    def child_count(self) -> int:
        return sum(
            1
            for attr in (self.collection, self.filter_expr,
                         self.projection, self.sort, self.limit)
            if attr is not None
        )


class ASTBuilder:

    _INTENT_KEYWORDS = {
        "find": "find", "get": "find", "show": "find",
        "list": "find", "retrieve": "find", "fetch": "find",
        "display": "find", "select": "find", "search": "find",
        "count": "aggregate", "sum": "aggregate",
        "average": "aggregate", "total": "aggregate",
        "group": "aggregate", "aggregate": "aggregate",
        "delete": "delete", "remove": "delete",
        "update": "update",
    }

    def __init__(self):
        self._tokens: List[Token] = []
        self._pos: int = 0

    def build(self, tokens: List[Token], raw_query: str = "") -> QueryNode:
        self._tokens = tokens
        self._pos = 0

        root = QueryNode(raw_query=raw_query)
        root.intent = self._extract_intent()
        root.collection = self._extract_collection()
        root.filter_expr = self._extract_filter()
        root.sort = self._extract_sort()
        root.limit = self._extract_limit()
        root.projection = self._extract_projection()

        return root

    def _extract_intent(self) -> str:
        for tok in self._tokens:
            if tok.type == TokenType.KEYWORD:
                mapped = self._INTENT_KEYWORDS.get(tok.value)
                if mapped:
                    return mapped
        return "find"

    def _extract_collection(self) -> CollectionRef:
        ref = CollectionRef()
        for i, tok in enumerate(self._tokens):
            if tok.type == TokenType.CONNECTOR and tok.value in ("in", "from"):
                nxt = self._peek(i + 1)
                if nxt and nxt.type == TokenType.IDENTIFIER:
                    ref.collection = nxt.value
                    ref.confidence = 0.9
                    return ref
        for tok in self._tokens:
            if tok.type == TokenType.IDENTIFIER:
                ref.collection = tok.value
                ref.confidence = 0.5
                return ref
        return ref

    def _extract_filter(self) -> Optional[FilterExpression]:
        comparisons: List[ComparisonOp] = []
        i = 0
        while i < len(self._tokens):
            tok = self._tokens[i]

            if tok.type == TokenType.IDENTIFIER:
                op_tok = self._peek(i + 1)
                val_tok = self._peek(i + 2)
                if (op_tok and op_tok.type == TokenType.OPERATOR
                        and val_tok and val_tok.type == TokenType.LITERAL):
                    comp = ComparisonOp(
                        field_name=tok.value,
                        operator=op_tok.value,
                        value=self._coerce_literal(val_tok.value),
                    )
                    comparisons.append(comp)
                    i += 3
                    continue

            if tok.type == TokenType.IDENTIFIER:
                nxt = self._peek(i + 1)
                val_tok = self._peek(i + 2)
                if (nxt and nxt.type == TokenType.CONNECTOR and nxt.value == "is"
                        and val_tok and val_tok.type == TokenType.LITERAL):
                    comp = ComparisonOp(
                        field_name=tok.value,
                        operator="$eq",
                        value=self._coerce_literal(val_tok.value),
                    )
                    comparisons.append(comp)
                    i += 3
                    continue

            i += 1

        if not comparisons:
            return None

        if len(comparisons) == 1:
            return FilterExpression(root=comparisons[0])

        logical = LogicalOp(kind=LogicalKind.AND, children=comparisons)
        return FilterExpression(root=logical)

    def _extract_sort(self) -> Optional[SortNode]:
        for i, tok in enumerate(self._tokens):
            if tok.type == TokenType.KEYWORD and tok.value in ("sort", "order"):
                nxt = self._peek(i + 1)
                field_tok = self._peek(i + 2) or self._peek(i + 1)
                if field_tok and field_tok.type == TokenType.IDENTIFIER:
                    direction = -1 if self._has_word("desc") else 1
                    return SortNode(fields=[(field_tok.value, direction)])
        return None

    def _extract_limit(self) -> Optional[LimitNode]:
        for i, tok in enumerate(self._tokens):
            if tok.type == TokenType.KEYWORD and tok.value in ("limit", "top", "first"):
                nxt = self._peek(i + 1)
                if nxt and nxt.type == TokenType.LITERAL:
                    try:
                        count = int(float(nxt.value))
                        return LimitNode(count=count)
                    except ValueError:
                        pass
        return None

    def _extract_projection(self) -> Optional[ProjectionNode]:
        fields: List[str] = []
        capturing = False
        for tok in self._tokens:
            if tok.type == TokenType.KEYWORD and tok.value in ("show", "select", "display"):
                capturing = True
                continue
            if capturing and tok.type == TokenType.IDENTIFIER:
                fields.append(tok.value)
            elif capturing and tok.type == TokenType.KEYWORD:
                break
        if fields:
            return ProjectionNode(include=fields)
        return None

    def _peek(self, index: int) -> Optional[Token]:
        if 0 <= index < len(self._tokens):
            return self._tokens[index]
        return None

    def _has_word(self, word: str) -> bool:
        return any(t.value == word for t in self._tokens)

    @staticmethod
    def _coerce_literal(value: str) -> Any:
        if value.lower() in ("null", "none", "nil"):
            return None
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
