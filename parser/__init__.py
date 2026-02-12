from .parser_engine import NLQueryParser

_parser_instance = None


def parse_query(nl_query: str, schema: dict) -> dict:
    global _parser_instance

    if not nl_query or not nl_query.strip():
        raise ValueError("Query string cannot be empty")

    if _parser_instance is None or _parser_instance._schema != schema:
        _parser_instance = NLQueryParser(schema)

    return _parser_instance.parse(nl_query)
