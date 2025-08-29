"""
Function Calling Approach Comparison
Blog 3: Structured Outputs & Function Calling - Production Approaches

Demonstrates and COMPARES two approaches to function calling:
1. Manual function calling via instruction prompts (shows what happens under the hood)
2. Native OpenAI function calling features (tools parameter)

This helps users understand the differences, trade-offs, and when to use each approach.
For learning fundamentals, start with basic_tools.py first.
"""

import sys
import os
# Add the ai-fundamentals-series directory to Python path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
from shared.llm_providers.openai_provider import get_openai_provider
# Import from the shared tools in the same examples directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.tools import SharedTools, OPENAI_TOOLS, TOOL_REGISTRY, execute_tool
from typing import Dict, List, Any, Optional
import json
import time


class ManualFunctionCallingAgent:
    """
    Approach 1: Manual function calling via instruction prompts.
    
    This shows what happens "under the hood" when using function calling:
    1. Tell AI about available functions via prompt
    2. Instruct AI on how to format function call requests
    3. Parse AI response for function calls
    4. Execute functions and return results to AI
    5. Get final AI response
    
    This approach works with ANY LLM provider.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        # Use shared tools instead of duplicating implementations
        self.available_tools = ["calculate", "get_current_time", "get_weather"]
        self.approach_name = "Manual (Instruction-based)"
    
    def get_tools_description(self) -> str:
        """Generate tool descriptions for the AI prompt."""
        descriptions = []
        
        tool_info = {
            "calculate": "Perform mathematical calculations. Usage: calculate(expression='2 + 3 * 4')",
            "get_current_time": "Get current date and time. Usage: get_current_time(format='human')",
            "get_weather": "Get weather for a location. Usage: get_weather(location='Tokyo', units='celsius')"
        }
        
        for tool_name in self.available_tools:
            if tool_name in tool_info:
                descriptions.append(f"- {tool_info[tool_name]}")
        
        return "\n".join(descriptions)
    
    def chat_with_manual_function_calling(self, user_message: str) -> str:
        """
        Manual function calling implementation showing the full process.
        
        This demonstrates the step-by-step process:
        1. Create prompt with tool descriptions
        2. Get AI response with potential function calls
        3. Parse and execute function calls
        4. Return results to AI for final response
        """
        
        # Step 1: Create prompt with tool descriptions
        tools_description = self.get_tools_description()
        
        system_prompt = f"""You are a helpful AI assistant with access to these tools:

{tools_description}

IMPORTANT INSTRUCTIONS:
- If you need to use a tool, format your response as: FUNCTION_CALL: function_name(parameter=value)
- You can call multiple functions by listing them on separate lines
- If no tools are needed, respond normally
- Always provide context and explanation with your responses

USER QUESTION: {user_message}"""
        
        print(f"📋 Available tools: {', '.join(self.available_tools)}")
        print("🔄 Step 1: AI analyzing question and deciding on tool usage...")
        
        # Step 2: Get initial AI response
        response = self.provider.generate(system_prompt)
        print(f"🤖 AI Response: {response.content[:100]}...")
        
        # Step 3: Check if AI wants to call functions
        if "FUNCTION_CALL:" in response.content:
            return self._process_function_calls(response.content, user_message)
        else:
            print("✅ No function calls needed - returning direct response")
            return response.content
    
    def _process_function_calls(self, ai_response: str, original_question: str) -> str:
        """Process and execute function calls from AI response."""
        
        print("🔧 Step 2: Function calls detected - parsing and executing...")
        
        function_results = []
        
        # Parse function calls from AI response
        lines = ai_response.split('\n')
        for line in lines:
            if "FUNCTION_CALL:" in line:
                try:
                    # Extract function call
                    func_call = line.split("FUNCTION_CALL:")[1].strip()
                    
                    # Parse function name and parameters (simplified parsing)
                    if '(' in func_call and ')' in func_call:
                        func_name = func_call.split('(')[0].strip()
                        params_str = func_call.split('(')[1].split(')')[0]
                        
                        # Parse parameters
                        params = {}
                        if params_str.strip():
                            for param in params_str.split(','):
                                if '=' in param:
                                    key, value = param.split('=', 1)
                                    key = key.strip()
                                    value = value.strip().strip("'\"")
                                    params[key] = value
                        
                        print(f"   Executing: {func_name}({params})")
                        
                        # Execute the function using shared tools
                        if func_name in self.available_tools:
                            result = execute_tool(func_name, **params)
                            function_results.append(f"{func_name} result: {result}")
                            print(f"   ✅ Result: {result}")
                        else:
                            error_msg = f"Function {func_name} not available"
                            function_results.append(error_msg)
                            print(f"   ❌ {error_msg}")
                            
                except Exception as e:
                    error_msg = f"Error executing function call: {e}"
                    function_results.append(error_msg)
                    print(f"   ❌ {error_msg}")
        
        # Step 4: Return results to AI for final response
        if function_results:
            print("🔄 Step 3: Sending function results back to AI for final response...")
            
            results_summary = "\n".join(function_results)
            final_prompt = f"""Based on the function execution results, provide a helpful final response to the user.

Original question: {original_question}

Function execution results:
{results_summary}

Provide a natural, conversational response that incorporates these results:"""
            
            final_response = self.provider.generate(final_prompt)
            print("✅ Final response generated")
            return final_response.content
        
        return "I attempted to call functions but couldn't get any results."


class NativeFunctionCallingAgent:
    """
    Approach 2: Native OpenAI function calling.
    
    This uses OpenAI's built-in function calling features:
    1. Define functions in OpenAI tools format
    2. OpenAI handles function call detection and parsing
    3. We execute functions and send results back
    4. OpenAI provides the final response
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        # Use shared tool definitions
        self.tools = OPENAI_TOOLS[:3]  # First 3 tools: calculate, get_current_time, get_weather
        self.approach_name = "Native OpenAI (Tools API)"
    
    def chat_with_native_function_calling(self, user_message: str) -> str:
        """
        Native OpenAI function calling implementation.
        
        This uses OpenAI's built-in function calling:
        1. Send tools definitions to OpenAI
        2. OpenAI decides when to call functions
        3. Execute functions locally
        4. Send results back to OpenAI
        5. Get final response
        """
        
        # Check if provider supports native function calling
        if not hasattr(self.provider, 'generate_with_tools'):
            print("⚠️  Provider doesn't support native function calling")
            return "This provider doesn't support native function calling. Try the manual approach."
        
        print(f"📋 Available tools: {len(self.tools)} native OpenAI tools")
        print("🔄 Step 1: OpenAI analyzing question and deciding on tool usage...")
        
        try:
            # Step 1: Call OpenAI with tools available
            response = self.provider.generate_with_tools(
                prompt=user_message,
                tools=self.tools
            )
            
            # Step 2: Check if OpenAI wants to call functions
            tool_calls = response.metadata.get('tool_calls', [])
            
            if tool_calls:
                print(f"🔧 Step 2: {len(tool_calls)} function call(s) detected by OpenAI")
                
                # Step 3: Execute each function call
                function_messages = []
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    arguments_str = tool_call['function']['arguments']
                    
                    try:
                        arguments = json.loads(arguments_str)
                        print(f"   Executing: {function_name}({arguments})")
                        
                        # Execute using shared tools
                        result = execute_tool(function_name, **arguments)
                        print(f"   ✅ Result: {result}")
                        
                        function_messages.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": str(result)
                        })
                        
                    except json.JSONDecodeError as e:
                        error_msg = f"Error parsing arguments: {e}"
                        print(f"   ❌ {error_msg}")
                        function_messages.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": error_msg
                        })
                
                # Step 4: Send function results back to OpenAI
                print("🔄 Step 3: Sending function results back to OpenAI for final response...")
                
                messages = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response.content or "", "tool_calls": tool_calls},
                    *function_messages
                ]
                
                # Get final response from OpenAI
                final_response = self.provider.client.chat.completions.create(
                    model=self.provider.config.model,
                    messages=messages,
                    tools=self.tools
                )
                
                print("✅ Final response generated by OpenAI")
                return final_response.choices[0].message.content
                
            else:
                # No function calls needed
                print("✅ No function calls needed - OpenAI responded directly")
                return response.content
                
        except Exception as e:
            return f"Error with native function calling: {str(e)}"


def demo_approach_comparison():
    """Demonstrate both approaches side-by-side for comparison."""
    print("=== Function Calling Approaches Comparison Demo ===\n")
    
    print("This demo compares TWO approaches to function calling:")
    print("1. Manual approach (works with any LLM)")
    print("2. Native OpenAI approach (uses built-in function calling)")
    print("-" * 70)
    
    # Initialize both agents
    manual_agent = ManualFunctionCallingAgent()
    native_agent = NativeFunctionCallingAgent()
    
    # Test queries
    test_queries = [
        "What's 25 * 34 + 567?",
        "What time is it right now?",
        "What's the weather like in Tokyo?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"🔍 Test Query {i}: '{query}'")
        print('='*70)
        
        # Test Manual Approach
        print(f"\n🛠️  APPROACH 1: {manual_agent.approach_name}")
        print("-" * 50)
        start_time = time.time()
        
        try:
            manual_result = manual_agent.chat_with_manual_function_calling(query)
            manual_time = time.time() - start_time
            print(f"📝 Result: {manual_result}")
            print(f"⏱️  Time: {manual_time:.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test Native Approach  
        print(f"\n🤖 APPROACH 2: {native_agent.approach_name}")
        print("-" * 50)
        start_time = time.time()
        
        try:
            native_result = native_agent.chat_with_native_function_calling(query)
            native_time = time.time() - start_time
            print(f"📝 Result: {native_result}")
            print(f"⏱️  Time: {native_time:.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    
    comparison_points = [
        {
            "aspect": "Compatibility",
            "manual": "✅ Works with any LLM provider",
            "native": "❌ OpenAI only (or compatible APIs)"
        },
        {
            "aspect": "Implementation Complexity", 
            "manual": "⚠️  More complex (manual parsing)",
            "native": "✅ Simpler (built-in handling)"
        },
        {
            "aspect": "Reliability",
            "manual": "⚠️  Text parsing can be error-prone",
            "native": "✅ Robust built-in validation"
        },
        {
            "aspect": "Performance",
            "manual": "⚠️  Multiple API calls needed",
            "native": "✅ Optimized single workflow"
        },
        {
            "aspect": "Control & Flexibility",
            "manual": "✅ Full control over process",
            "native": "❌ Limited to provider's implementation"
        },
        {
            "aspect": "Error Handling",
            "manual": "⚠️  Manual error handling required",
            "native": "✅ Built-in error handling"
        }
    ]
    
    print(f"{'Aspect':<25} {'Manual Approach':<35} {'Native Approach'}")
    print("-" * 80)
    
    for point in comparison_points:
        print(f"{point['aspect']:<25} {point['manual']:<35} {point['native']}")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("   Manual Approach - Choose when:")
    print("     • Using non-OpenAI providers")
    print("     • Need maximum control and customization")
    print("     • Building educational/learning projects")
    
    print("   Native Approach - Choose when:")
    print("     • Using OpenAI or compatible APIs")
    print("     • Need production reliability and performance")
    print("     • Want simpler implementation")
    
    print("\n💡 For learning function calling basics, start with basic_tools.py!")


if __name__ == "__main__":
    demo_approach_comparison()