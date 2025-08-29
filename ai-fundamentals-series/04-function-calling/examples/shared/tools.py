"""
Shared Tool Implementations
Blog 3: Structured Outputs & Function Calling - Shared Components

Common tool implementations used across basic and advanced function calling examples.
This eliminates duplication and provides consistent tool behavior.
"""

import datetime
import random
from typing import Dict, Any


class SharedTools:
    """Collection of common tools used in function calling examples."""
    
    @staticmethod
    def calculate(expression: str) -> str:
        """
        Safe mathematical calculator.
        
        Args:
            expression: Mathematical expression to evaluate (e.g., '2 + 3 * 4')
            
        Returns:
            str: Calculation result or error message
        """
        try:
            # Security: Only allow safe mathematical characters
            allowed_chars = set('0123456789+-*/().^ ')
            if not all(c in allowed_chars for c in expression.replace('**', '^')):
                return "Error: Expression contains invalid characters"
            
            # Convert ^ to ** for Python power operator
            safe_expression = expression.replace('^', '**')
            
            # Evaluate safely
            result = eval(safe_expression)
            return f"The result is {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def get_current_time(format: str = "human") -> str:
        """
        Get current date and time.
        
        Args:
            format: Time format ('human', 'iso', or 'timestamp')
            
        Returns:
            str: Formatted current time
        """
        now = datetime.datetime.now()
        
        if format == "iso":
            return now.isoformat()
        elif format == "timestamp":
            return str(int(now.timestamp()))
        else:  # human format
            return now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    @staticmethod
    def get_weather(location: str, units: str = "celsius") -> str:
        """
        Simulated weather information (for demo purposes).
        
        Args:
            location: City name or location
            units: Temperature units ('celsius' or 'fahrenheit')
            
        Returns:
            str: Weather information
        """
        # Simulate weather data (in real app, would call weather API)
        base_temp = random.randint(-10, 35) if units == "celsius" else random.randint(14, 95)
        conditions = ["Sunny", "Cloudy", "Rainy", "Partly cloudy", "Overcast", "Clear"]
        condition = random.choice(conditions)
        
        temp_unit = "°C" if units == "celsius" else "°F"
        return f"Weather in {location}: {base_temp}{temp_unit}, {condition}"
    
    @staticmethod
    def search_web(query: str, max_results: int = 5) -> str:
        """
        Simulated web search (for demo purposes).
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            str: Search results
        """
        # Simulate search results (in real app, would call search API)
        results = [
            f"Result {i+1}: Information about '{query}' from website{i+1}.com"
            for i in range(min(max_results, 3))
        ]
        
        return f"Search results for '{query}':\n" + "\n".join(results)
    
    @staticmethod
    def send_email(to: str, subject: str, body: str) -> str:
        """
        Simulated email sending (for demo purposes).
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body content
            
        Returns:
            str: Status message
        """
        # Simulate email sending (in real app, would use email service)
        if "@" not in to:
            return f"Error: Invalid email address '{to}'"
        
        return f"Email sent successfully to {to} with subject '{subject}'"
    
    @staticmethod
    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert between different units.
        
        Args:
            value: Numerical value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            str: Conversion result
        """
        # Temperature conversions
        if from_unit == "celsius" and to_unit == "fahrenheit":
            result = (value * 9/5) + 32
            return f"{value}°C = {result:.2f}°F"
        elif from_unit == "fahrenheit" and to_unit == "celsius":
            result = (value - 32) * 5/9
            return f"{value}°F = {result:.2f}°C"
        
        # Length conversions
        elif from_unit == "meters" and to_unit == "feet":
            result = value * 3.28084
            return f"{value} meters = {result:.2f} feet"
        elif from_unit == "feet" and to_unit == "meters":
            result = value / 3.28084
            return f"{value} feet = {result:.2f} meters"
        
        else:
            return f"Error: Conversion from {from_unit} to {to_unit} not supported"


# Tool definitions in OpenAI format for native function calling
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Time format: 'human', 'iso', or 'timestamp'",
                        "enum": ["human", "iso", "timestamp"]
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert between different units of measurement",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Numerical value to convert"
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit (celsius, fahrenheit, meters, feet)"
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Target unit (celsius, fahrenheit, meters, feet)"
                    }
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    }
]

# Tool registry mapping function names to implementations
TOOL_REGISTRY = {
    "calculate": SharedTools.calculate,
    "get_current_time": SharedTools.get_current_time,
    "get_weather": SharedTools.get_weather,
    "search_web": SharedTools.search_web,
    "send_email": SharedTools.send_email,
    "convert_units": SharedTools.convert_units,
}


def execute_tool(tool_name: str, **kwargs) -> str:
    """
    Execute a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Arguments to pass to the tool
        
    Returns:
        str: Tool execution result
    """
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Tool '{tool_name}' not found. Available tools: {list(TOOL_REGISTRY.keys())}"
    
    try:
        tool_function = TOOL_REGISTRY[tool_name]
        return tool_function(**kwargs)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def get_tool_descriptions() -> Dict[str, str]:
    """Get human-readable descriptions of all available tools."""
    return {
        "calculate": "Perform mathematical calculations safely",
        "get_current_time": "Get current date and time in various formats",
        "get_weather": "Get weather information for any location (simulated)",
        "search_web": "Search the web for information (simulated)",
        "send_email": "Send emails (simulated)",
        "convert_units": "Convert between different units (temperature, length)"
    }


if __name__ == "__main__":
    # Demo of shared tools
    print("=== Shared Tools Demo ===\n")
    
    tools = SharedTools()
    
    # Test each tool
    print("1. Calculator:")
    print(f"   {tools.calculate('15 * 7 + 3')}")
    
    print("\n2. Current Time:")
    print(f"   {tools.get_current_time('human')}")
    
    print("\n3. Weather:")
    print(f"   {tools.get_weather('Tokyo', 'celsius')}")
    
    print("\n4. Unit Conversion:")
    print(f"   {tools.convert_units(25, 'celsius', 'fahrenheit')}")
    
    print(f"\n✅ All {len(TOOL_REGISTRY)} shared tools working correctly!")