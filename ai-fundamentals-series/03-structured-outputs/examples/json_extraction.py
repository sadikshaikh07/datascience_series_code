"""
Structured JSON Output Examples
Blog 3: Structured Outputs & Function Calling - Section 2

Demonstrates how to get consistent JSON output from LLMs using schemas,
validation, and best practices for reliable structured data generation.

Use cases: API responses, data extraction, form processing, configuration generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError, Field


class StructuredOutputGenerator:
    """
    Generates structured JSON outputs from LLMs with validation.
    Demonstrates various approaches to ensure consistent formatting.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.technique_name = "Structured JSON Output"
    
    def basic_json_extraction(self, text: str, fields: List[str]) -> LLMResponse:
        """Basic JSON extraction with field specification."""
        fields_str = ", ".join(fields)
        
        prompt = f"""Extract the following information from the text and return it as valid JSON:

Fields to extract: {fields_str}

Text: "{text}"

Return only valid JSON with the specified fields:"""
        
        return self.provider.generate(prompt)
    
    def schema_guided_extraction(self, text: str, schema: Dict[str, Any]) -> LLMResponse:
        """JSON extraction guided by a detailed schema."""
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""Extract information from the text and format it according to this exact JSON schema:

Schema:
{schema_str}

Text: "{text}"

Return valid JSON that matches the schema exactly. Include all required fields:"""
        
        return self.provider.generate(prompt)
    
    def few_shot_json_generation(self, examples: List[tuple], new_input: str) -> LLMResponse:
        """Use examples to show the desired JSON format."""
        prompt = "Convert text to JSON following these examples:\n\n"
        
        for text_input, json_output in examples:
            prompt += f"Input: {text_input}\nOutput: {json_output}\n\n"
        
        prompt += f"Input: {new_input}\nOutput:"
        
        return self.provider.generate(prompt)
    
    def constrained_json_generation(self, task: str, constraints: Dict[str, Any]) -> LLMResponse:
        """Generate JSON with specific constraints and validation rules."""
        constraints_str = json.dumps(constraints, indent=2)
        
        prompt = f"""Complete this task and return the result as JSON following these constraints:

Task: {task}

Constraints:
{constraints_str}

Generate valid JSON that satisfies all constraints:"""
        
        return self.provider.generate(prompt)
    
    def multi_level_json_extraction(self, text: str, nested_schema: Dict[str, Any]) -> LLMResponse:
        """Extract complex nested JSON structures."""
        schema_str = json.dumps(nested_schema, indent=2)
        
        prompt = f"""Analyze the text and extract information into this nested JSON structure:

Text: "{text}"

Required JSON structure:
{schema_str}

Return complete JSON with all nested levels populated:"""
        
        return self.provider.generate(prompt)
    
    def json_array_generation(self, items_description: str, item_schema: Dict[str, Any], 
                             count: Optional[int] = None) -> LLMResponse:
        """Generate JSON arrays of structured objects."""
        schema_str = json.dumps(item_schema, indent=2)
        count_str = f" (exactly {count} items)" if count else ""
        
        prompt = f"""Generate a JSON array{count_str} based on this description:

Description: {items_description}

Each array item should follow this schema:
{schema_str}

Return a valid JSON array:"""
        
        return self.provider.generate(prompt)


class JSONValidator:
    """Validates and cleans JSON output from LLMs."""
    
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response that may contain extra text."""
        # Try to find JSON in the response
        start_markers = ['{', '[']
        end_markers = ['}', ']']
        
        for start_marker, end_marker in zip(start_markers, end_markers):
            start_idx = response.find(start_marker)
            if start_idx != -1:
                # Find the matching closing bracket/brace
                bracket_count = 0
                for i, char in enumerate(response[start_idx:], start_idx):
                    if char == start_marker:
                        bracket_count += 1
                    elif char == end_marker:
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_str = response[start_idx:i+1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                continue
        return None
    
    @staticmethod
    def validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate JSON data against a simple schema."""
        errors = []
        
        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in data:
                expected_type = field_schema.get('type')
                actual_value = data[field]
                
                if expected_type == 'string' and not isinstance(actual_value, str):
                    errors.append(f"Field '{field}' should be string, got {type(actual_value).__name__}")
                elif expected_type == 'integer' and not isinstance(actual_value, int):
                    errors.append(f"Field '{field}' should be integer, got {type(actual_value).__name__}")
                elif expected_type == 'array' and not isinstance(actual_value, list):
                    errors.append(f"Field '{field}' should be array, got {type(actual_value).__name__}")
                elif expected_type == 'object' and not isinstance(actual_value, dict):
                    errors.append(f"Field '{field}' should be object, got {type(actual_value).__name__}")
        
        return len(errors) == 0, errors


# Pydantic models for type-safe JSON handling
class PersonModel(BaseModel):
    """Example Pydantic model for person data."""
    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., ge=0, le=150, description="Age in years")
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$', description="Valid email address")
    occupation: Optional[str] = Field(None, description="Job title or occupation")
    skills: List[str] = Field(default_factory=list, description="List of skills")


class ProductModel(BaseModel):
    """Example Pydantic model for product data."""
    name: str = Field(..., description="Product name")
    price: float = Field(..., gt=0, description="Price in USD")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(..., description="Whether item is in stock")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Product specifications")


def demo_structured_outputs():
    """Comprehensive demonstration of structured JSON output techniques."""
    print("=== Structured JSON Output Techniques Demo ===\n")
    
    try:
        generator = StructuredOutputGenerator()
        validator = JSONValidator()
        
        print(f"🤖 Using provider: {generator.provider.provider_name}")
        print(f"📝 Technique: {generator.technique_name}")
        print("-" * 60 + "\n")
        
        # 1. Basic JSON Extraction
        print("1. Basic JSON Extraction")
        print("Task: Extract structured information from text")
        
        text = "John Smith is a 35-year-old software engineer at Google. He can be reached at john.smith@google.com."
        fields = ["name", "age", "occupation", "company", "email"]
        
        response = generator.basic_json_extraction(text, fields)
        print(f"Input: {text}")
        print(f"Fields: {fields}")
        print(f"Raw response: {response.content}")
        
        # Try to extract and validate JSON
        extracted_json = validator.extract_json_from_response(response.content)
        if extracted_json:
            print(f"Extracted JSON: {json.dumps(extracted_json, indent=2)}")
        else:
            print("❌ Could not extract valid JSON from response")
        print()
        
        # 2. Schema-Guided Extraction
        print("2. Schema-Guided JSON Extraction")
        print("Task: Extract data following a specific schema")
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "contact": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "phone": {"type": "string"}
                    }
                },
                "employment": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "position": {"type": "string"},
                        "years_experience": {"type": "integer"}
                    }
                }
            },
            "required": ["name", "age", "contact", "employment"]
        }
        
        text2 = "Sarah Johnson, 28, works as a data scientist at Microsoft for 3 years. Contact: sarah.j@microsoft.com, phone: 555-123-4567."
        response = generator.schema_guided_extraction(text2, schema)
        
        print(f"Input: {text2}")
        print("Schema: Complex nested object")
        print(f"Response: {response.content}")
        
        extracted_json = validator.extract_json_from_response(response.content)
        if extracted_json:
            is_valid, errors = validator.validate_against_schema(extracted_json, schema)
            print(f"✅ Valid schema: {is_valid}")
            if errors:
                for error in errors:
                    print(f"   ❌ {error}")
        print()
        
        # 3. Few-Shot JSON Generation
        print("3. Few-Shot JSON Generation")
        print("Task: Learn JSON format from examples")
        
        examples = [
            ("Apple iPhone 14, $999, Electronics, In Stock", 
             '{"name": "Apple iPhone 14", "price": 999.0, "category": "Electronics", "in_stock": true}'),
            ("Nike Air Max, $120, Footwear, Out of Stock",
             '{"name": "Nike Air Max", "price": 120.0, "category": "Footwear", "in_stock": false}')
        ]
        
        new_input = "Sony PlayStation 5, $499, Gaming, Available"
        response = generator.few_shot_json_generation(examples, new_input)
        
        print("Examples:")
        for inp, out in examples:
            print(f"  Input: {inp}")
            print(f"  Output: {out}")
        
        print(f"\nNew input: {new_input}")
        print(f"Generated JSON: {response.content}")
        print()
        
        # 4. Constrained JSON Generation
        print("4. Constrained JSON Generation")
        print("Task: Generate JSON with specific business rules")
        
        task = "Create a user profile for a premium subscription service"
        constraints = {
            "user_tier": {"enum": ["basic", "premium", "enterprise"]},
            "subscription_months": {"minimum": 1, "maximum": 12},
            "features": {"minItems": 3, "maxItems": 10},
            "billing_amount": {"minimum": 10.0, "maximum": 1000.0}
        }
        
        response = generator.constrained_json_generation(task, constraints)
        print(f"Task: {task}")
        print(f"Constraints: {len(constraints)} rules defined")
        print(f"Generated profile: {response.content}")
        print()
        
        # 5. Multi-Level JSON Extraction
        print("5. Multi-Level JSON Extraction")
        print("Task: Extract complex nested structures")
        
        nested_schema = {
            "company": {
                "name": "string",
                "location": "string",
                "employees": {
                    "count": "integer",
                    "departments": ["string"]
                }
            },
            "financial": {
                "revenue": "number",
                "expenses": {
                    "operational": "number",
                    "marketing": "number"
                }
            }
        }
        
        complex_text = """TechCorp Inc. is located in San Francisco with 250 employees across Engineering, Sales, and Marketing departments. 
        Last year they generated $5.2M in revenue, with operational expenses of $2.1M and marketing expenses of $800K."""
        
        response = generator.multi_level_json_extraction(complex_text, nested_schema)
        print(f"Complex text: {complex_text[:100]}...")
        print("Schema: Multi-level nested structure")
        print(f"Extracted data: {response.content}")
        print()
        
        # 6. JSON Array Generation
        print("6. JSON Array Generation")
        print("Task: Generate arrays of structured objects")
        
        item_schema = {
            "id": "integer",
            "task": "string", 
            "priority": "string",
            "estimated_hours": "number",
            "assigned_to": "string"
        }
        
        description = "Create a project task list for developing a mobile app"
        response = generator.json_array_generation(description, item_schema, count=5)
        
        print(f"Description: {description}")
        print(f"Item schema: {len(item_schema)} fields per item")
        print(f"Generated array: {response.content}")
        print()
        
        # 7. Pydantic Model Validation
        print("7. Type-Safe JSON with Pydantic")
        print("Task: Validate JSON against Python models")
        
        person_text = "Alice Cooper, age 29, software architect at Apple, skills: Python, React, AWS"
        
        # Generate JSON for person
        person_prompt = f"""Extract person information from this text and format as JSON:
        
Text: {person_text}

Required fields: name, age, email (generate if missing), occupation, skills (as array)

JSON:"""
        
        response = generator.provider.generate(person_prompt)
        print(f"Input: {person_text}")
        print(f"Generated JSON: {response.content}")
        
        # Try to validate with Pydantic
        try:
            extracted_json = validator.extract_json_from_response(response.content)
            if extracted_json:
                # Add email if missing (Pydantic requires it)
                if 'email' not in extracted_json:
                    extracted_json['email'] = 'alice.cooper@apple.com'
                
                person = PersonModel(**extracted_json)
                print(f"✅ Pydantic validation successful!")
                print(f"   Name: {person.name}")
                print(f"   Age: {person.age}")
                print(f"   Email: {person.email}")
                print(f"   Skills: {person.skills}")
        except ValidationError as e:
            print(f"❌ Pydantic validation failed: {e}")
        print()
        
        print("=" * 70)
        print("🎯 Structured JSON Output Key Insights:")
        print("   ✅ Schemas improve consistency and reliability")
        print("   ✅ Few-shot examples teach complex formats")
        print("   ✅ Validation catches errors early")
        print("   ✅ Pydantic provides type safety")
        print("   ⚠️  Always validate LLM-generated JSON")
        print("   ⚠️  Provide clear examples for complex structures")
        print("   💡 Use constraints to enforce business rules")
        print("   💡 Extract JSON from responses with extra text")
        
    except Exception as e:
        print(f"❌ Error during structured output demo: {e}")
        print("💡 Make sure your LLM provider is properly configured")


def compare_json_approaches():
    """Compare different approaches to getting structured JSON."""
    print("\n=== JSON Generation Approach Comparison ===\n")
    
    generator = StructuredOutputGenerator()
    task_input = "Create a product listing for: Gaming laptop, $1299, high-performance for gaming"
    
    approaches = [
        ("Basic prompt", f"Convert to JSON: {task_input}"),
        ("Schema-guided", f"Convert to JSON following this schema - name:string, price:number, category:string, features:array:\n{task_input}"),
        ("Example-driven", f"Convert to JSON like this example:\n'Phone, $699, Electronics' -> {{'name':'Phone','price':699,'category':'Electronics'}}\n\nNow convert: {task_input}")
    ]
    
    print("Task: Generate JSON for product information")
    print(f"Input: {task_input}\n")
    
    for approach_name, prompt in approaches:
        print(f"{approach_name}:")
        try:
            response = generator.provider.generate(prompt)
            print(f"   Result: {response.content.strip()}")
        except Exception as e:
            print(f"   Error: {e}")
        print()
    
    print("📊 Comparison Results:")
    print("   Basic: Fast but inconsistent format")
    print("   Schema-guided: Reliable structure, clear requirements")
    print("   Example-driven: Best for complex formats with patterns")
    print("   Winner: Schema-guided for production systems")


if __name__ == "__main__":
    demo_structured_outputs()
    compare_json_approaches()