"""Quick smoke test for the parser module."""
import sys
import os
import json

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Step 1: Import tokenizer ===")
try:
    from parser.tokenizer import Tokenizer, extract_schema_fields
    print("  OK: tokenizer imported")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 2: Import ast_nodes ===")
try:
    from parser.ast_nodes import ASTBuilder, QueryNode
    print("  OK: ast_nodes imported")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 3: Import grammar ===")
try:
    from parser.grammar import GrammarMatcher
    print("  OK: grammar imported")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 4: Import query_builder ===")
try:
    from parser.query_builder import MongoQueryBuilder
    print("  OK: query_builder imported")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 5: Import parser_engine ===")
try:
    from parser.parser_engine import NLQueryParser
    print("  OK: parser_engine imported")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 6: Import parse_query from parser ===")
try:
    from parser import parse_query
    print("  OK: parse_query imported")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 7: Load schema ===")
try:
    with open("full_schema.json", "r") as f:
        schema = json.load(f)
    print(f"  OK: schema loaded with {len(schema)} databases")
except Exception as e:
    print(f"  FAIL: {e}")
    schema = None

print("\n=== Step 8: Test tokenizer standalone ===")
try:
    if schema:
        fields = extract_schema_fields(schema)
        print(f"  Schema fields found: {len(fields)}")
        print(f"  Sample fields: {list(fields)[:10]}")
        tok = Tokenizer(schema_fields=fields)
        tokens = tok.tokenize("Find events where camera_id is 'CAM001'")
        print(f"  Tokens: {len(tokens)}")
        for t in tokens:
            print(f"    {t}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 9: Test AST builder standalone ===")
try:
    if schema:
        builder = ASTBuilder()
        ast = builder.build(tokens, raw_query="Find events where camera_id is 'CAM001'")
        print(f"  AST intent: {ast.intent}")
        print(f"  AST children: {ast.child_count}")
        print(f"  Has filter: {ast.has_filter}")
        if ast.collection:
            print(f"  Collection: {ast.collection.fully_qualified}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 10: Test full parse_query (requires inference backend) ===")
try:
    if schema:
        import logging
        # Only show parser logs, suppress noisy HTTP internals
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        print("  (This step calls the inference backend — may take 30-60s...)")
        result = parse_query("Find events where camera_id is 'CAM001'", schema)
        print(f"  Result: {json.dumps(result, indent=2)}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== DONE ===")
