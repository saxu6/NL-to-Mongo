"""
Ollama Verification Script
Goal: Confirm Ollama is properly configured and capable of handling our requirements
"""

import json
import sys
import subprocess
from typing import Dict, Any, Optional

try:
    import ollama
except ImportError:
    print("ERROR: ollama package not installed. Install it with: pip install ollama")
    sys.exit(1)


def check_ollama_service() -> bool:
    """M1.1: Check if Ollama service is running"""
    print("\n" + "="*60)
    print("M1.1: Checking Ollama Installation and Service Status")
    print("="*60)
    
    try:
        # Try to list models - this will fail if Ollama service is not running
        models = ollama.list()
        print("✓ Ollama service is running")
        return True
    except Exception as e:
        print(f"✗ Ollama service is not running or not accessible")
        print(f"  Error: {str(e)}")
        print("\n  Troubleshooting:")
        print("  1. Make sure Ollama is installed: https://ollama.ai/download")
        print("  2. Start Ollama service: ollama serve")
        print("  3. Or run Ollama desktop application")
        return False


def list_ollama_models() -> list:
    """M1.2: List present Ollama models"""
    print("\n" + "="*60)
    print("M1.2: Listing Available Ollama Models")
    print("="*60)
    
    try:
        models = ollama.list()
        model_list = models.get('models', [])
        
        if not model_list:
            print("⚠ No models found. You may need to pull a model first.")
            print("  Example: ollama pull llama2")
            return []
        
        print(f"✓ Found {len(model_list)} model(s):")
        for i, model in enumerate(model_list, 1):
            model_name = model.get('name', 'Unknown')
            model_size = model.get('size', 0)
            size_gb = model_size / (1024**3) if model_size else 0
            print(f"  {i}. {model_name} ({size_gb:.2f} GB)")
        
        return [model.get('name') for model in model_list]
    except Exception as e:
        print(f"✗ Failed to list models: {str(e)}")
        return []


def test_basic_prompt(model_name: str) -> bool:
    """Test basic prompt response"""
    print("\n" + "="*60)
    print("M1.2: Testing Basic Prompt Response")
    print("="*60)
    
    try:
        test_prompt = "What is MongoDB?"
        print(f"Prompt: {test_prompt}")
        print("Generating response...")
        
        response = ollama.generate(model=model_name, prompt=test_prompt)
        
        if response and 'response' in response:
            answer = response['response']
            print(f"✓ Response received ({len(answer)} characters)")
            print(f"\nResponse preview:\n{answer[:200]}...")
            return True
        else:
            print("✗ No response received")
            return False
    except Exception as e:
        print(f"✗ Failed to generate response: {str(e)}")
        return False


def test_json_output(model_name: str) -> bool:
    """M1.3: Test JSON format output capability"""
    print("\n" + "="*60)
    print("M1.3: Testing JSON Format Output Capability")
    print("="*60)
    
    try:
        json_prompt = """Convert the following natural language query to JSON format:
"Find all users where age is greater than 25"

Respond ONLY with valid JSON in this format:
{
  "collection": "users",
  "filter": {"age": {"$gt": 25}}
}"""
        
        print("Prompt: Convert NL query to JSON")
        print("Generating JSON response...")
        
        response = ollama.generate(
            model=model_name,
            prompt=json_prompt,
            options={
                'temperature': 0.1,  # Lower temperature for more consistent JSON
            }
        )
        
        if response and 'response' in response:
            answer = response['response'].strip()
            print(f"✓ Response received")
            print(f"\nRaw response:\n{answer}\n")
            
            # Try to extract JSON from response
            json_str = extract_json_from_response(answer)
            
            if json_str:
                try:
                    parsed_json = json.loads(json_str)
                    print("✓ Valid JSON parsed successfully:")
                    print(json.dumps(parsed_json, indent=2))
                    return True
                except json.JSONDecodeError as e:
                    print(f"✗ Response contains text but JSON parsing failed: {str(e)}")
                    return False
            else:
                print("⚠ Could not extract JSON from response")
                return False
        else:
            print("✗ No response received")
            return False
    except Exception as e:
        print(f"✗ Failed to test JSON output: {str(e)}")
        return False


def extract_json_from_response(text: str) -> Optional[str]:
    """Extract JSON from response text (may contain markdown code blocks)"""
    # Remove markdown code blocks if present
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    
    # Try to find JSON object boundaries
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx+1]
    
    return text.strip()


def test_mongodb_query_conversion(model_name: str) -> bool:
    """M1.3: Test simple NL to MongoDB query conversion"""
    print("\n" + "="*60)
    print("M1.3: Testing NL to MongoDB Query Conversion")
    print("="*60)
    
    test_queries = [
        "Find all users where age is greater than 25",
        "Get products with price less than 100",
        "Show me all orders from the last 7 days"
    ]
    
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"NL Query: {query}")
        
        try:
            prompt = f"""Convert the following natural language query to a MongoDB query in JSON format:

"{query}"

Respond with ONLY valid JSON in this format:
{{
  "collection": "collection_name",
  "filter": {{}},
  "projection": {{}},
  "sort": {{}},
  "limit": null
}}

Use MongoDB operators like $gt, $lt, $gte, $lte, $in, $regex, etc. in the filter field."""
            
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                }
            )
            
            if response and 'response' in response:
                answer = response['response'].strip()
                json_str = extract_json_from_response(answer)
                
                if json_str:
                    try:
                        parsed_json = json.loads(json_str)
                        print("✓ Valid MongoDB query JSON:")
                        print(json.dumps(parsed_json, indent=2))
                        success_count += 1
                    except json.JSONDecodeError as e:
                        print(f"✗ JSON parsing failed: {str(e)}")
                        print(f"Response: {answer[:200]}...")
                else:
                    print(f"✗ Could not extract JSON")
                    print(f"Response: {answer[:200]}...")
            else:
                print("✗ No response received")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    print(f"\n--- Summary ---")
    print(f"Successfully converted: {success_count}/{len(test_queries)} queries")
    return success_count == len(test_queries)


def main():
    """Main verification function"""
    print("\n" + "="*60)
    print("OLLAMA VERIFICATION SCRIPT")
    print("="*60)
    
    results = {
        'service_running': False,
        'models_available': False,
        'basic_prompt': False,
        'json_output': False,
        'mongodb_conversion': False
    }
    
    # M1.1: Check Ollama service
    if not check_ollama_service():
        print("\n" + "="*60)
        print("VERIFICATION FAILED: Ollama service is not running")
        print("="*60)
        return results
    
    results['service_running'] = True
    
    # M1.2: List models
    models = list_ollama_models()
    if not models:
        print("\n" + "="*60)
        print("VERIFICATION INCOMPLETE: No models available")
        print("Please pull a model first: ollama pull llama2")
        print("="*60)
        return results
    
    results['models_available'] = True
    
    # Use the first available model
    model_name = models[0]
    print(f"\nUsing model: {model_name}")
    
    # Test basic prompt
    results['basic_prompt'] = test_basic_prompt(model_name)
    
    # Test JSON output
    results['json_output'] = test_json_output(model_name)
    
    # Test MongoDB query conversion
    results['mongodb_conversion'] = test_mongodb_query_conversion(model_name)
    
    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"M1.1 - Ollama Service: {'✓ PASS' if results['service_running'] else '✗ FAIL'}")
    print(f"M1.2 - Models Available: {'✓ PASS' if results['models_available'] else '✗ FAIL'}")
    print(f"M1.2 - Basic Prompt: {'✓ PASS' if results['basic_prompt'] else '✗ FAIL'}")
    print(f"M1.3 - JSON Output: {'✓ PASS' if results['json_output'] else '✗ FAIL'}")
    print(f"M1.3 - MongoDB Conversion: {'✓ PASS' if results['mongodb_conversion'] else '✗ FAIL'}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
    else:
        print("⚠ SOME VERIFICATIONS FAILED - Review output above")
    print("="*60)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0 if all(results.values()) else 1)
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

