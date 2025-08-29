"""
Advanced Structured Outputs Implementation
Blog 3: Structured Outputs & Function Calling - Section 2 Advanced

Demonstrates TWO approaches to structured outputs as specified in the todo:
1. JSON Schema in prompts (traditional approach)  
2. Controlled Generation/Constrained Decoding (native provider features)

This shows users what happens under the hood vs using built-in provider features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
from shared.llm_providers.openai_provider import OpenAIProvider
from typing import Dict, List, Any, Optional, Union, Type
import json
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError, Field
from enum import Enum
import time


# Pydantic Models for Controlled Generation
class TaskPriority(str, Enum):
    """Enum for task priorities."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    URGENT = "urgent"


class PersonInfo(BaseModel):
    """Person information model."""
    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., ge=0, le=150, description="Age in years") 
    email: str = Field(..., description="Valid email address")
    occupation: Optional[str] = Field(None, description="Job title or occupation")
    skills: List[str] = Field(default_factory=list, description="List of professional skills")
    experience_years: Optional[int] = Field(None, ge=0, description="Years of professional experience")


class ProductInfo(BaseModel):
    """Product information model."""
    name: str = Field(..., description="Product name")
    price: float = Field(..., gt=0, description="Price in USD")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(..., description="Whether item is currently in stock")
    description: Optional[str] = Field(None, description="Product description")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Technical specifications")


class TaskInfo(BaseModel):
    """Task information model."""
    id: int = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")
    priority: TaskPriority = Field(..., description="Task priority level")
    estimated_hours: float = Field(..., gt=0, description="Estimated time to complete")
    assigned_to: Optional[str] = Field(None, description="Person assigned to the task")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")


class CompanyInfo(BaseModel):
    """Company information model."""
    name: str = Field(..., description="Company name")
    industry: str = Field(..., description="Primary industry")
    location: str = Field(..., description="Headquarters location")
    employee_count: int = Field(..., ge=1, description="Number of employees")
    founded_year: int = Field(..., ge=1800, le=2024, description="Year company was founded")
    revenue: Optional[float] = Field(None, ge=0, description="Annual revenue in millions USD")
    is_public: bool = Field(..., description="Whether company is publicly traded")


class AnalysisResult(BaseModel):
    """Analysis result model."""
    summary: str = Field(..., description="Summary of the analysis")
    key_findings: List[str] = Field(..., description="List of key findings")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the analysis (0-1)")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional analysis metadata")


class TraditionalStructuredOutputGenerator:
    """
    Approach 1: Traditional JSON Schema in Prompts
    This shows what happens "under the hood" when we manually specify schemas in prompts.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.approach_name = "Traditional Schema-in-Prompt"
    
    def extract_with_schema_prompt(self, text: str, target_model: Type[BaseModel]) -> dict:
        """Extract data using JSON schema specified in the prompt."""
        # Get the JSON schema from the Pydantic model
        schema = target_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""Extract information from the following text and return it as JSON that exactly matches this schema:

SCHEMA:
{schema_str}

TEXT TO ANALYZE:
{text}

INSTRUCTIONS:
- Return ONLY valid JSON that matches the schema exactly
- Include all required fields
- Use appropriate data types (string, integer, boolean, etc.)
- If information is missing, use null for optional fields
- For arrays, provide empty arrays if no items found

JSON OUTPUT:"""
        
        return self._execute_with_retry(prompt, target_model)
    
    def _execute_with_retry(self, prompt: str, target_model: Type[BaseModel], max_retries: int = 3) -> dict:
        """Execute prompt with validation and retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.provider.generate(prompt)
                
                # Extract JSON from response
                json_str = self._extract_json_string(response.content)
                if not json_str:
                    raise ValueError("No valid JSON found in response")
                
                # Parse JSON
                data = json.loads(json_str)
                
                # Validate against Pydantic model
                validated_model = target_model(**data)
                
                return {
                    "success": True,
                    "data": validated_model.model_dump(),
                    "raw_response": response.content,
                    "approach": self.approach_name,
                    "attempts": attempt + 1
                }
                
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "raw_response": response.content if 'response' in locals() else None,
                        "approach": self.approach_name,
                        "attempts": attempt + 1
                    }
                
                # Add more specific instructions for retry
                prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {str(e)}\nPlease ensure the JSON is valid and matches the schema exactly."
    
    def _extract_json_string(self, text: str) -> Optional[str]:
        """Extract JSON string from potentially verbose response."""
        # Look for JSON object or array
        start_chars = ['{', '[']
        end_chars = ['}', ']']
        
        for start_char, end_char in zip(start_chars, end_chars):
            start_idx = text.find(start_char)
            if start_idx != -1:
                bracket_count = 0
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == start_char:
                        bracket_count += 1
                    elif char == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            return text[start_idx:i+1]
        
        return None


class ControlledGenerationManager:
    """
    Approach 2: Controlled Generation/Constrained Decoding
    This uses built-in LLM provider features for structured output generation.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.approach_name = "Controlled Generation (Native)"
    
    def extract_with_controlled_generation(self, text: str, target_model: Type[BaseModel]) -> dict:
        """Extract data using provider's native structured output features."""
        try:
            # Check if provider supports structured outputs
            if hasattr(self.provider, 'generate_structured'):
                return self._use_native_structured_output(text, target_model)
            else:
                return self._simulate_controlled_generation(text, target_model)
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Controlled generation failed: {str(e)}",
                "approach": self.approach_name
            }
    
    def _use_native_structured_output(self, text: str, target_model: Type[BaseModel]) -> dict:
        """Use provider's native structured output API."""
        prompt = f"""Analyze the following text and extract structured information:

{text}

Please provide the information in the requested format."""
        
        try:
            # This would use OpenAI's structured outputs or similar features
            if isinstance(self.provider, OpenAIProvider):
                return self._openai_structured_output(prompt, target_model)
            else:
                return self._generic_structured_output(prompt, target_model)
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Native structured output failed: {str(e)}",
                "approach": self.approach_name
            }
    
    def _openai_structured_output(self, prompt: str, target_model: Type[BaseModel]) -> dict:
        """Use OpenAI's native structured output features (response_format parameter)."""
        try:
            # Check if provider supports native structured outputs
            if hasattr(self.provider, 'generate_structured'):
                # Use the native structured output method
                response = self.provider.generate_structured(prompt, target_model)
                
                # Parse the JSON response
                data = json.loads(response.content)
                validated_model = target_model(**data)
                
                return {
                    "success": True,
                    "data": validated_model.model_dump(),
                    "raw_response": response.content,
                    "approach": f"{self.approach_name} (OpenAI Native)",
                    "controlled_generation": True,
                    "usage": response.usage
                }
            else:
                # Fallback to enhanced prompt-based approach
                schema = target_model.model_json_schema()
                
                enhanced_prompt = f"""{prompt}

IMPORTANT: Your response must be valid JSON that exactly matches this structure:
{json.dumps(schema, indent=2)}

Ensure all required fields are present and data types are correct."""
                
                response = self.provider.generate(enhanced_prompt)
                
                json_str = self._extract_json_string(response.content)
                if not json_str:
                    raise ValueError("No valid JSON in response")
                
                data = json.loads(json_str)
                validated_model = target_model(**data)
                
                return {
                    "success": True,
                    "data": validated_model.model_dump(),
                    "raw_response": response.content,
                    "approach": f"{self.approach_name} (OpenAI Fallback)",
                    "controlled_generation": False,
                    "usage": response.usage
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "approach": f"{self.approach_name} (OpenAI)",
                "controlled_generation": True
            }
    
    def _generic_structured_output(self, prompt: str, target_model: Type[BaseModel]) -> dict:
        """Generic structured output for non-OpenAI providers."""
        # For other providers, we can implement similar constrained decoding
        # This would typically involve:
        # 1. JSON schema-guided generation
        # 2. Token-level constraints
        # 3. Grammar-based parsing
        
        schema = target_model.model_json_schema()
        
        enhanced_prompt = f"""{prompt}

Generate a JSON response that strictly follows this schema:
{json.dumps(schema, indent=2)}

Rules:
1. Return ONLY valid JSON, no additional text
2. All required fields must be present
3. Use correct data types for each field
4. Follow any constraints (min/max values, patterns, etc.)

JSON:"""
        
        response = self.provider.generate(enhanced_prompt)
        
        try:
            json_str = self._extract_json_string(response.content)
            if not json_str:
                raise ValueError("No valid JSON found")
            
            data = json.loads(json_str)
            validated_model = target_model(**data)
            
            return {
                "success": True,
                "data": validated_model.model_dump(),
                "raw_response": response.content,
                "approach": f"{self.approach_name} (Generic)",
                "controlled_generation": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "approach": f"{self.approach_name} (Generic)",
                "controlled_generation": True
            }
    
    def _simulate_controlled_generation(self, text: str, target_model: Type[BaseModel]) -> dict:
        """Simulate controlled generation for providers without native support."""
        schema = target_model.model_json_schema()
        
        # Create a highly structured prompt that mimics controlled generation
        prompt = f"""TEXT TO ANALYZE:
{text}

TASK: Extract structured information and format as JSON.

SCHEMA CONSTRAINTS:
{json.dumps(schema, indent=2)}

GENERATION RULES:
- Output must be valid JSON only
- Match schema exactly
- Use proper data types
- Include all required fields
- No additional text or explanations

Begin JSON generation now:"""
        
        try:
            response = self.provider.generate(prompt)
            json_str = self._extract_json_string(response.content)
            
            if not json_str:
                raise ValueError("No valid JSON in response")
            
            data = json.loads(json_str)
            validated_model = target_model(**data)
            
            return {
                "success": True,
                "data": validated_model.model_dump(),
                "raw_response": response.content,
                "approach": f"{self.approach_name} (Simulated)",
                "controlled_generation": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "approach": f"{self.approach_name} (Simulated)",
                "controlled_generation": False
            }
    
    def _extract_json_string(self, text: str) -> Optional[str]:
        """Extract JSON from response text."""
        start_chars = ['{', '[']
        end_chars = ['}', ']']
        
        for start_char, end_char in zip(start_chars, end_chars):
            start_idx = text.find(start_char)
            if start_idx != -1:
                bracket_count = 0
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == start_char:
                        bracket_count += 1
                    elif char == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            return text[start_idx:i+1]
        
        return None


def demo_advanced_structured_outputs():
    """Comprehensive demo comparing both approaches to structured outputs."""
    print("=== Advanced Structured Outputs: Two Approaches Demo ===\n")
    
    # Initialize both approaches
    traditional = TraditionalStructuredOutputGenerator()
    controlled = ControlledGenerationManager()
    
    print(f"🔧 Approach 1: {traditional.approach_name}")
    print(f"🤖 Approach 2: {controlled.approach_name}")
    print("-" * 70 + "\n")
    
    # Test cases with different complexity levels
    test_cases = [
        {
            "name": "Simple Person Extraction",
            "text": "John Smith, 35 years old, software engineer at Google with 10 years experience. Email: john@google.com. Skills: Python, JavaScript, AWS.",
            "model": PersonInfo,
            "description": "Basic structured data extraction"
        },
        {
            "name": "Product Information",
            "text": "MacBook Pro 16-inch laptop, price $2499, Electronics category, currently in stock. High-performance laptop with M2 Pro chip, 16GB RAM, 512GB SSD.",
            "model": ProductInfo,
            "description": "E-commerce product data extraction"
        },
        {
            "name": "Task Management",
            "text": "Task #1: Implement user authentication system. High priority task assigned to Sarah Johnson. Should take about 20 hours to complete. Due date is 2024-03-15.",
            "model": TaskInfo,
            "description": "Project management data extraction"
        },
        {
            "name": "Company Analysis",
            "text": "TechCorp Inc. is a software company founded in 2015, headquartered in San Francisco. They have 250 employees and operate in the cloud computing industry. The company is privately held with annual revenue of $50 million.",
            "model": CompanyInfo,
            "description": "Complex business data extraction"
        }
    ]
    
    results_comparison = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        print(f"   Input: {test_case['text'][:80]}...")
        print(f"   Target Model: {test_case['model'].__name__}")
        print()
        
        # Test Approach 1: Traditional Schema-in-Prompt
        print("   📋 Approach 1: Traditional Schema-in-Prompt")
        start_time = time.time()
        result1 = traditional.extract_with_schema_prompt(test_case['text'], test_case['model'])
        time1 = time.time() - start_time
        
        if result1['success']:
            print("      ✅ Success")
            print(f"      ⏱️  Time: {time1:.2f}s")
            print(f"      🔄 Attempts: {result1['attempts']}")
            print("      📋 Extracted Data:")
            for key, value in result1['data'].items():
                print(f"         {key}: {value}")
        else:
            print("      ❌ Failed")
            print(f"      🚨 Error: {result1['error']}")
            print(f"      🔄 Attempts: {result1['attempts']}")
        print()
        
        # Test Approach 2: Controlled Generation
        print("   🤖 Approach 2: Controlled Generation/Constrained Decoding")
        start_time = time.time()
        result2 = controlled.extract_with_controlled_generation(test_case['text'], test_case['model'])
        time2 = time.time() - start_time
        
        if result2['success']:
            print("      ✅ Success")
            print(f"      ⏱️  Time: {time2:.2f}s")
            print(f"      🎯 Native Support: {result2.get('controlled_generation', False)}")
            print("      📋 Extracted Data:")
            for key, value in result2['data'].items():
                print(f"         {key}: {value}")
        else:
            print("      ❌ Failed")
            print(f"      🚨 Error: {result2['error']}")
        print()
        
        # Store comparison results
        results_comparison.append({
            'test_case': test_case['name'],
            'approach1_success': result1['success'],
            'approach1_time': time1,
            'approach2_success': result2['success'],
            'approach2_time': time2,
            'approach1_attempts': result1.get('attempts', 0),
            'controlled_generation': result2.get('controlled_generation', False)
        })
        
        print("-" * 50 + "\n")
    
    # Results Summary
    print("=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)
    
    approach1_successes = sum(1 for r in results_comparison if r['approach1_success'])
    approach2_successes = sum(1 for r in results_comparison if r['approach2_success'])
    avg_time1 = sum(r['approach1_time'] for r in results_comparison) / len(results_comparison)
    avg_time2 = sum(r['approach2_time'] for r in results_comparison) / len(results_comparison)
    avg_attempts1 = sum(r['approach1_attempts'] for r in results_comparison) / len(results_comparison)
    
    print(f"Test Cases: {len(test_cases)}")
    print()
    print("📋 Approach 1 (Traditional Schema-in-Prompt):")
    print(f"   ✅ Success Rate: {approach1_successes}/{len(test_cases)} ({approach1_successes/len(test_cases)*100:.1f}%)")
    print(f"   ⏱️  Average Time: {avg_time1:.2f}s")
    print(f"   🔄 Average Attempts: {avg_attempts1:.1f}")
    print()
    print("🤖 Approach 2 (Controlled Generation):")
    print(f"   ✅ Success Rate: {approach2_successes}/{len(test_cases)} ({approach2_successes/len(test_cases)*100:.1f}%)")
    print(f"   ⏱️  Average Time: {avg_time2:.2f}s")
    print(f"   🎯 Native Support Available: {any(r['controlled_generation'] for r in results_comparison)}")
    print()
    print("🔍 KEY DIFFERENCES:")
    print("   Traditional Approach:")
    print("     ➕ Works with any LLM provider")
    print("     ➕ Full control over prompt structure")
    print("     ➕ Can implement custom retry logic")
    print("     ➖ Requires manual JSON extraction")
    print("     ➖ May need multiple attempts for complex schemas")
    print("     ➖ More prone to parsing errors")
    print()
    print("   Controlled Generation Approach:")
    print("     ➕ Built-in validation and constraints")
    print("     ➕ More reliable output format")
    print("     ➕ Better performance with native support")
    print("     ➕ Less post-processing needed")
    print("     ➖ Limited to providers with native support")
    print("     ➖ Less flexibility in error handling")
    print("     ➖ May not work with all model types")
    print()
    print("🏆 RECOMMENDATION:")
    print("   Use Controlled Generation when available for:")
    print("   - Production systems requiring reliability")
    print("   - Complex schemas with strict validation")
    print("   - High-volume processing")
    print()
    print("   Use Traditional Schema-in-Prompt for:")
    print("   - Provider compatibility across different LLMs")
    print("   - Custom error handling requirements")
    print("   - Educational purposes to understand the process")


def demo_pydantic_integration():
    """Demonstrate advanced Pydantic integration patterns."""
    print("\n=== Advanced Pydantic Integration Demo ===\n")
    
    controlled = ControlledGenerationManager()
    
    # Complex analysis task
    analysis_text = """
    Our Q3 sales data shows a 25% increase over Q2, with particularly strong performance in the mobile app segment.
    Customer satisfaction scores averaged 4.2/5.0, up from 3.8 last quarter. Key findings include:
    1. Mobile users spend 40% more time in the app
    2. Subscription conversion rate improved by 15%
    3. Customer support tickets decreased by 20%
    
    Recommendations: Invest more in mobile development, expand subscription features, 
    and maintain current customer support quality standards.
    """
    
    print("📊 Complex Analysis Extraction")
    print(f"Input: {analysis_text[:100]}...")
    print()
    
    result = controlled.extract_with_controlled_generation(analysis_text, AnalysisResult)
    
    if result['success']:
        print("✅ Analysis Extraction Successful!")
        analysis = AnalysisResult(**result['data'])
        
        print(f"📋 Summary: {analysis.summary}")
        print(f"🎯 Confidence: {analysis.confidence_score:.2f}")
        print("🔍 Key Findings:")
        for finding in analysis.key_findings:
            print(f"   • {finding}")
        print("💡 Recommendations:")
        for rec in analysis.recommendations:
            print(f"   • {rec}")
        
        if analysis.metadata:
            print("📈 Metadata:")
            for key, value in analysis.metadata.items():
                print(f"   {key}: {value}")
    else:
        print(f"❌ Analysis failed: {result['error']}")
    
    print("\n" + "="*50)
    print("🎯 Pydantic Integration Benefits:")
    print("   ✅ Type safety and validation")
    print("   ✅ Automatic documentation generation")  
    print("   ✅ IDE support with autocomplete")
    print("   ✅ Serialization/deserialization")
    print("   ✅ Custom validation rules")
    print("   💡 Perfect for API development")
    print("   💡 Integrates well with FastAPI, Django")


if __name__ == "__main__":
    demo_advanced_structured_outputs()
    demo_pydantic_integration()