"""Lark-based natural language parser that produces the project AST."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from lark import Lark, Transformer, v_args

from .ast_nodes import ComparisonNode, LimitNode, LogicalNode, QueryNode, QueryRoot, SortNode, ValueNode


_GRAMMAR = r"""
start: query

query: intent? collection_ref? filter_clause? sort_clause? limit_clause?

intent: find_intent
      | count_intent
      | delete_intent
      | update_intent

find_intent: "show"i | "list"i | "find"i | "get"i | "fetch"i | "display"i | "retrieve"i
count_intent: "count"i | "how"i "many"i | "number"i "of"i | "total"i
delete_intent: "delete"i | "remove"i | "drop"i
update_intent: "update"i | "modify"i | "change"i | "edit"i | "set"i

collection_ref: ("from"i | "in"i)? ("all"i | "the"i)? WORD

filter_clause: WHERE expr               -> where_filter
             | implied_comparison       -> implied_filter

?expr: or_expr
?or_expr: and_expr ("or"i and_expr)*
?and_expr: atom ("and"i atom)*
?atom: comparison
     | "(" expr ")"

comparison: field_ref operator value    -> explicit_cmp
          | implied_comparison

implied_comparison: "older"i "than"i value   -> older_than_cmp
                  | "younger"i "than"i value -> younger_than_cmp

field_ref: WORD

operator: op_eq
        | op_ne
        | op_gt
        | op_gte
        | op_lt
        | op_lte
        | op_contains
        | op_in
        | op_nin

op_eq: "=" | "is"i | "equals"i | "equal"i "to"i
op_ne: "!=" | "is"i "not"i | "does"i "not"i "equal"i | "not"i "equal"i ("to"i)?
op_gt: ">" | "greater"i "than"i | "more"i "than"i | "above"i | "after"i
op_gte: ">=" | "at"i "least"i | "greater"i "than"i "or"i "equal"i ("to"i)?
op_lt: "<" | "less"i "than"i | "below"i | "before"i
op_lte: "<=" | "at"i "most"i | "less"i "than"i "or"i "equal"i ("to"i)?
op_contains: "contains"i | "contain"i | "matches"i | "match"i | "like"i
op_in: "in"i
op_nin: "not"i "in"i

sort_clause: ("sort"i | "sorted"i | "order"i | "ordered"i) "by"i field_ref sort_dir?
sort_dir: SORT_ASC | SORT_DESC

limit_clause: ("limit"i | "top"i | "first"i) SIGNED_NUMBER

?value: QUOTED_STRING -> string_val
      | DATE          -> date_val
      | SIGNED_NUMBER -> number_val
      | WORD          -> word_val

WHERE: /(where|with|that|whose|having)\b/i
SORT_ASC: /(asc|ascending)\b/i
SORT_DESC: /(desc|descending)\b/i

DATE.2: /\d{4}-\d{2}-\d{2}(?:T[\d:.]+)?/
WORD: /[A-Za-z_][\w]*/
QUOTED_STRING: /\"[^\"]*\"/ | /'[^']*'/
SIGNED_NUMBER: /[+-]?(?:\d+\.?\d*|\.\d+)/

%import common.WS
%ignore WS
"""


@v_args(inline=True)
class _ASTBuilder(Transformer):
    """Transforms a Lark parse tree into AST dataclasses."""

    def __init__(self, raw_text: str):
        super().__init__()
        self._raw_text = raw_text

    def query(self, *items):
        root = QueryRoot(raw_text=self._raw_text)
        for item in items:
            if item is None:
                continue
            if isinstance(item, tuple):
                key, value = item
                if key == "intent":
                    root.intent = value
                elif key == "collection":
                    root.collection = value
            elif isinstance(item, QueryNode):
                root.children.append(item)
        return root

    def intent(self, intent_name):
        return ("intent", intent_name)

    def find_intent(self, *_items):
        return "find"

    def count_intent(self, *_items):
        return "count"

    def delete_intent(self, *_items):
        return "delete"

    def update_intent(self, *_items):
        return "update"

    def collection_ref(self, word):
        return ("collection", str(word))

    def where_filter(self, _where, expression):
        return expression

    def implied_filter(self, comparison):
        return comparison

    def or_expr(self, first, *rest):
        nodes = [first, *[item for item in rest if isinstance(item, QueryNode)]]
        if len(nodes) == 1:
            return nodes[0]
        return LogicalNode(operator="$or", children=list(nodes))

    def and_expr(self, first, *rest):
        nodes = [first, *[item for item in rest if isinstance(item, QueryNode)]]
        if len(nodes) == 1:
            return nodes[0]
        return LogicalNode(operator="$and", children=list(nodes))

    def explicit_cmp(self, field, operator, value):
        node = ComparisonNode(field=field, operator=operator, value=value)
        node.children.append(value)
        return node

    def older_than_cmp(self, value):
        node = ComparisonNode(field="age", operator="$gt", value=value)
        node.children.append(value)
        node.meta["inferred_from"] = "older than"
        return node

    def younger_than_cmp(self, value):
        node = ComparisonNode(field="age", operator="$lt", value=value)
        node.children.append(value)
        node.meta["inferred_from"] = "younger than"
        return node

    def field_ref(self, token):
        return str(token)

    def operator(self, op):
        return op

    def op_eq(self, *_items):
        return "$eq"

    def op_ne(self, *_items):
        return "$ne"

    def op_gt(self, *_items):
        return "$gt"

    def op_gte(self, *_items):
        return "$gte"

    def op_lt(self, *_items):
        return "$lt"

    def op_lte(self, *_items):
        return "$lte"

    def op_contains(self, *_items):
        return "$regex"

    def op_in(self, *_items):
        return "$in"

    def op_nin(self, *_items):
        return "$nin"

    def sort_clause(self, *_items):
        field = None
        direction = None
        for item in _items:
            if isinstance(item, str) and field is None:
                field = item
            elif isinstance(item, int):
                direction = item
        return SortNode(field=field or "", direction=direction if direction is not None else -1)

    def sort_dir(self, token):
        if token.type == "SORT_ASC":
            return 1
        return -1

    def limit_clause(self, *items):
        number = items[-1]
        return LimitNode(value=int(float(str(number))))

    def string_val(self, token):
        return ValueNode(raw=str(token).strip('"').strip("'"))

    def date_val(self, token):
        return ValueNode(raw=str(token))

    def number_val(self, token):
        return ValueNode(raw=str(token))

    def word_val(self, token):
        return ValueNode(raw=str(token))

    def start(self, root):
        return root


class NLParser:
    """Facade for parsing natural-language queries into AST nodes."""

    def __init__(self):
        self._parser = Lark(_GRAMMAR, parser="lalr")

    def parse(self, query: str) -> QueryRoot:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        raw = query.strip()
        tree = self._parser.parse(raw)
        return _ASTBuilder(raw).transform(tree)


_DEFAULT_PARSER = NLParser()


def parse(query: str) -> QueryRoot:
    """Parse NL query text into a QueryRoot AST."""

    return _DEFAULT_PARSER.parse(query)


def ast_to_dict(node: QueryNode) -> dict[str, Any]:
    """Convert AST dataclasses to plain dictionaries for debugging/tests."""

    if not is_dataclass(node):
        raise TypeError("ast_to_dict expects an AST dataclass node")
    return asdict(node)
