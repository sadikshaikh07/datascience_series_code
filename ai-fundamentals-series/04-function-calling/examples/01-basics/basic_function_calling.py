"""
Basic Function Calling Examples
Blog 3: Structured Outputs & Function Calling - Learning Fundamentals

Demonstrates the CORE CONCEPTS of function calling:
- How to register tools with AI
- Basic tool execution patterns
- Understanding function calling workflow
- Simple error handling

This file focuses on LEARNING the fundamentals. For production patterns and
approach comparisons, see advanced_comparison.py.
"""

import sys
import os
# Add the ai-fundamentals-series directory to Python path for shared imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
# Import from the shared tools in the same examples directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.tools import SharedTools, execute_tool, get_tool_descriptions
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool call request from the AI."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    call_id: Optional[str]
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BasicFunctionCallingAgent:
    """
    EDUCATIONAL function calling agent demonstrating core concepts.
    
    This agent shows the fundamental concepts:
    1. How to register tools
    2. How AI decides to use tools  
    3. Basic execution workflow
    4. Simple error handling
    
    For advanced patterns and approach comparisons, see advanced_comparison.py
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.available_tools = ["calculate", "get_current_time", "get_weather"]
        
        print("🤖 Basic Function Calling Agent initialized!")
        print(f"📋 Available tools: {self.available_tools}")
    
    def register_custom_tool(self, name: str, description: str):
        """
        EDUCATIONAL: Show how tool registration works conceptually.
        
        In real implementation, you would:
        1. Define the function
        2. Register it with parameters schema  
        3. Add to available tools list
        """
        if name not in self.available_tools:
            self.available_tools.append(name)
            print(f"✅ Tool '{name}' registered: {description}")
        else:
            print(f"⚠️  Tool '{name}' already exists")
    
    def list_available_tools(self) -> Dict[str, str]:
        """Show all available tools with descriptions."""
        descriptions = get_tool_descriptions()
        available_descriptions = {
            tool: descriptions.get(tool, "Description not available")
            for tool in self.available_tools
        }
        return available_descriptions
    
    def chat_with_tools(self, user_message: str) -> str:
        """
        EDUCATIONAL: Basic function calling implementation.
        
        This shows the core workflow:
        1. Create prompt with tool descriptions
        2. Ask AI to respond or use tools
        3. Parse AI response for function calls
        4. Execute functions
        5. Return results
        
        This is a simplified version to understand the concept.
        """
        
        # Step 1: Build prompt with available tools
        tools_info = self.list_available_tools()
        tools_prompt = "Available tools:\\n"
        for tool_name, description in tools_info.items():
            tools_prompt += f"- {tool_name}: {description}\\n"
        
        # Step 2: Create instruction prompt
        system_prompt = f"""You are a helpful AI assistant with access to tools.

{tools_prompt}

INSTRUCTIONS:
- If the user's question requires using a tool, format your response as:
  TOOL_CALL: tool_name(parameter1=value1, parameter2=value2)
- If no tool is needed, respond normally
- Only use tools that are available
- Be helpful and provide context with your responses

USER QUESTION: {user_message}"""
        
        # Step 3: Get AI response
        print("🤔 AI is thinking about whether to use tools...")
        response = self.provider.generate(system_prompt)
        
        # Step 4: Check if AI wants to use a tool
        if "TOOL_CALL:" in response.content:
            return self._execute_tool_call(response.content, user_message)
        else:
            # No tool needed, return direct response
            print("💬 AI responded directly (no tools needed)")
            return response.content
    
    def _execute_tool_call(self, ai_response: str, original_question: str) -> str:
        """
        EDUCATIONAL: Extract and execute tool calls from AI response.
        
        This shows how to:
        1. Parse AI response for function calls
        2. Extract function name and parameters
        3. Execute the function
        4. Handle errors
        5. Return results to user
        """
        
        try:
            # Step 1: Extract tool call (simplified parsing)
            tool_call_line = [line for line in ai_response.split('\\n') if 'TOOL_CALL:' in line][0]
            tool_call_part = tool_call_line.split('TOOL_CALL:')[1].strip()
            
            # Step 2: Parse function name (simplified)
            if '(' in tool_call_part:
                tool_name = tool_call_part.split('(')[0].strip()
                params_part = tool_call_part.split('(')[1].split(')')[0]
                
                # Step 3: Parse parameters (very simplified)
                params = {}
                if params_part.strip():
                    # Basic parameter parsing for educational purposes
                    for param in params_part.split(','):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            params[key] = value
                
                print(f"🔧 Tool call detected: {tool_name} with params: {params}")
                
                # Step 4: Execute the tool
                if tool_name in self.available_tools:
                    result = execute_tool(tool_name, **params)
                    print(f"⚡ Tool execution result: {result}")
                    
                    # Step 5: Generate final response
                    final_prompt = f"""Based on the tool result, provide a helpful response to the user.

Original question: {original_question}
Tool used: {tool_name}
Tool result: {result}

Provide a natural, helpful response incorporating this information:"""
                    
                    final_response = self.provider.generate(final_prompt)
                    return final_response.content
                else:
                    return f"Error: Tool '{tool_name}' not available"
            
        except Exception as e:
            print(f"❌ Tool execution error: {e}")
            return f"I encountered an error while trying to use tools: {str(e)}"
        
        return "I couldn't parse the tool call properly. Please try again."


def demo_basic_function_calling():
    """Demonstrate basic function calling concepts step by step."""
    print("=== Basic Function Calling Demo ===\\n")
    
    try:
        # Step 1: Create agent
        print("Step 1: Initialize Basic Function Calling Agent")
        agent = BasicFunctionCallingAgent()
        print()
        
        # Step 2: Show available tools
        print("Step 2: List Available Tools")
        tools = agent.list_available_tools()
        for name, desc in tools.items():
            print(f"   📋 {name}: {desc}")
        print()
        
        # Step 3: Register a custom tool (educational)
        print("Step 3: Register Custom Tool (Educational)")
        agent.register_custom_tool("example_tool", "This is just an example for learning")
        print()
        
        # Step 4: Test function calling with different types of queries
        test_queries = [
            "What's 25 * 34?",
            "What time is it?", 
            "Hello, how are you?",  # No tool needed
            "What's the weather like in Paris?"
        ]
        
        print("Step 4: Test Function Calling")
        for i, query in enumerate(test_queries, 1):
            print(f"\\n🔍 Test Query {i}: '{query}'")
            print("-" * 50)
            
            response = agent.chat_with_tools(query)
            print(f"🤖 Final Response: {response}")
            
            if i < len(test_queries):  # Don't add separator after last query
                print()
        
        print("\\n" + "=" * 70)
        print("🎯 Basic Function Calling Key Concepts Demonstrated:")
        print("   ✅ Tool registration and discovery")
        print("   ✅ AI decision making for tool usage")  
        print("   ✅ Function call parsing and execution")
        print("   ✅ Error handling and recovery")
        print("   ✅ Response generation with tool results")
        print("\\n💡 Next Step: Try advanced_comparison.py to see different approaches!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("💡 Make sure your LLM provider is properly configured")


def compare_function_calling_approaches():
    """
    EDUCATIONAL: Show the difference between basic and advanced approaches.
    
    This is for learning - shows what approaches exist without implementing them all.
    """
    print("\\n=== Function Calling Approaches Comparison ===\\n")
    
    approaches = {
        "Basic (This File)": {
            "description": "Educational prompt-based function calling",
            "pros": [
                "Easy to understand",
                "Works with any LLM",
                "Full control over process",
                "Good for learning"
            ],
            "cons": [
                "Manual parsing required", 
                "Error-prone text processing",
                "Limited scalability"
            ]
        },
        "Advanced Manual (advanced_comparison.py)": {
            "description": "Production-ready manual function calling",
            "pros": [
                "Robust error handling",
                "Works with any LLM",
                "Flexible and customizable",
                "Production ready"
            ],
            "cons": [
                "More complex implementation",
                "Still requires manual parsing"
            ]
        },
        "Native OpenAI (advanced_comparison.py)": {
            "description": "OpenAI's built-in function calling",
            "pros": [
                "Reliable and fast",
                "No manual parsing",
                "Built-in validation", 
                "Optimized performance"
            ],
            "cons": [
                "OpenAI-specific",
                "Less control over process",
                "Requires compatible models"
            ]
        }
    }
    
    print("📊 Function Calling Implementation Approaches:\\n")
    
    for approach_name, details in approaches.items():
        print(f"🔧 {approach_name}:")
        print(f"   Description: {details['description']}")
        print("   Pros:")
        for pro in details['pros']:
            print(f"     ✅ {pro}")
        print("   Cons:")  
        for con in details['cons']:
            print(f"     ❌ {con}")
        print()
    
    print("🎓 Learning Recommendation:")
    print("   1. Start here (basic_tools.py) - Learn core concepts")
    print("   2. Move to advanced_comparison.py - See production approaches") 
    print("   3. Choose the right approach for your use case")


if __name__ == "__main__":
    demo_basic_function_calling()
    compare_function_calling_approaches()