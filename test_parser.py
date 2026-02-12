import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parser.tokenizer import Tokenizer, extract_schema_fields
from parser.ast_nodes import ASTBuilder, QueryNode
from parser.grammar import GrammarMatcher
from parser.query_builder import MongoQueryBuilder
from parser.parser_engine import NLQueryParser
from parser import parse_query

with open("full_schema.json", "r") as f:
        schema = json.load(f)
    print(f"schema loaded with {len(schema)} databases")
except Exception as e:
    print(f"schema not loaded: {e}")
    schema = None

try:
    if schema:
        fields = extract_schema_fields(schema)
        print(f"  Schema fields found: {len(fields)}")
        print(f"  Sample fields: {list(fields)[:10]}")
        tok = Tokenizer(schema_fields=fields)

        # input here 
        tokens = tok.tokenize("Find events where camera_id is 'CAM001'")
        print(f"  Tokens: {len(tokens)}")
        for t in tokens:
            print(f"    {t}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

try:
    if schema:
        builder = ASTBuilder()

        # input here
        ast = builder.build(tokens, raw_query="Find events where camera_id is 'CAM001'")
        print(f"  AST intent: {ast.intent}")
        print(f"  AST children: {ast.child_count}")
        print(f"  Has filter: {ast.has_filter}")
        if ast.collection:
            print(f"  Collection: {ast.collection.fully_qualified}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

try:
    if schema:
        # input here
        result = parse_query("Find events where camera_id is 'CAM001'", schema)
        print(f"  Result: {json.dumps(result, indent=2)}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()
