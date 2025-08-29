"""
OpenAI Provider Implementation
Blog 2: Prompt Engineering Fundamentals

Simplified OpenAI implementation with support for:
- Official OpenAI API
- OpenAI-compatible APIs (Ollama, LocalAI, etc.) via base_url parameter
- Native function calling and structured outputs

NOTE: We focus on OpenAI for simplicity. Other providers like Anthropic, Google, etc.
can be added using their specific documentation while following this same structure.
A bonus blog will cover multiple providers and advanced configurations.

One-liner usage: 
provider = get_openai_provider("gpt-4")
response = provider.generate("Your prompt here")
"""

import os
import time
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import asdict
import json

from .base_llm import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from openai import OpenAI
    from pydantic import BaseModel
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    BaseModel = None


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider with support for official API and compatible endpoints.
    
    Features:
    - Standard chat completions
    - Native function calling (tools)
    - Structured outputs (response_format)
    - Configurable base_url for compatibility with other providers
    """
    
    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        super().__init__(config)
        
        # Initialize OpenAI client with configurable base_url
        client_params = {
            "api_key": os.getenv('OPENAI_API_KEY', 'sk-placeholder'),
            "timeout": config.timeout
        }
        
        # Add base_url if specified (for Ollama, LocalAI, etc.)
        if config.base_url:
            client_params["base_url"] = config.base_url
            
        self.client = OpenAI(**client_params)
        self.provider_name = "openai"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using OpenAI chat completions API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                **(self.config.extra_params or {})
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Build usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "response_time": elapsed_time
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is properly configured."""
        if not OPENAI_AVAILABLE:
            return False
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'sk-placeholder':
            return False
        
        try:
            # Test with a simple request
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], 
                           system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate response with function calling capabilities.
        
        This is the NATIVE OpenAI function calling approach using the tools parameter.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,  # Native OpenAI function calling
                tool_choice="auto",  # Let the model decide when to use tools
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                **(self.config.extra_params or {})
            )
            
            # Extract the response
            choice = response.choices[0]
            message = choice.message
            
            # Build usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            
            # Handle tool calls
            metadata = {
                "finish_reason": choice.finish_reason,
                "response_id": response.id,
                "tool_calls": []
            }
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    metadata["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return LLMResponse(
                content=message.content or "",
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                metadata=metadata
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI function calling failed: {str(e)}")
    
    def generate_structured(self, prompt: str, response_format: Union[Dict[str, Any], Type[BaseModel]], 
                           system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate structured output using OpenAI's native response_format parameter.
        
        This is the NATIVE OpenAI structured output approach.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Handle response format
        if hasattr(response_format, 'model_json_schema'):
            # Pydantic model
            format_param = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                    "strict": True
                }
            }
        else:
            # JSON schema dict
            format_param = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                response_format=format_param,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                **(self.config.extra_params or {})
            )
            
            content = response.choices[0].message.content
            
            # Build usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                    "structured_output": True
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI structured output failed: {str(e)}")
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count using tiktoken if available, otherwise rough estimation.
        """
        try:
            import tiktoken
            
            # Get encoding for the model
            if "gpt-4" in self.config.model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.config.model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
            
        except ImportError:
            # Fallback to rough estimation if tiktoken not available
            return len(text) // 4
    
    def list_available_models(self) -> list:
        """
        Get list of available OpenAI models.
        
        Returns:
            list: List of model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if 'gpt' in model.id]
        except Exception:
            return ['gpt-4', 'gpt-3.5-turbo']  # Default models
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost for API call (approximate rates as of 2024).
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            float: Estimated cost in USD
        """
        # Approximate pricing (check OpenAI pricing page for current rates)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
        }
        
        model_family = 'gpt-4' if 'gpt-4' in self.config.model else 'gpt-3.5-turbo'
        rates = pricing.get(model_family, pricing['gpt-3.5-turbo'])
        
        input_cost = (prompt_tokens / 1000) * rates['input']
        output_cost = (completion_tokens / 1000) * rates['output']
        
        return input_cost + output_cost


def get_openai_provider(model: str = "gpt-4", base_url: Optional[str] = None) -> OpenAIProvider:
    """
    Convenience function to create an OpenAI provider.
    
    Args:
        model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        base_url: Custom base URL for OpenAI-compatible APIs (Ollama, LocalAI, etc.)
        
    Returns:
        Configured OpenAI provider
        
    Examples:
        # Official OpenAI
        provider = get_openai_provider("gpt-4")
        
        # Ollama (OpenAI-compatible)
        provider = get_openai_provider("llama3.1", "http://localhost:11434/v1")
        
        # LocalAI
        provider = get_openai_provider("gpt-3.5-turbo", "http://localhost:8080/v1")
    """
    config = LLMConfig(
        model=model,
        max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1000')),
        temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
        top_p=float(os.getenv('OPENAI_TOP_P', '1.0')),
        base_url=base_url
    )
    
    return OpenAIProvider(config)


def demo_openai_provider():
    """Demonstrate OpenAI provider capabilities."""
    print("=== OpenAI Provider Demo ===\n")
    
    try:
        # Create provider with custom config
        config = LLMConfig(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            max_tokens=150,
            temperature=0.7
        )
        
        provider = OpenAIProvider(config)
        
        print(f"✅ OpenAI Provider initialized with model: {config.model}")
        print(f"📊 Provider available: {provider.is_available()}")
        
        # Test basic generation
        print("\n1. Basic Generation Test:")
        response = provider.generate("Explain what an API is in one sentence.")
        
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        
        # Test with system prompt
        print("\n2. System Prompt Test:")
        system_prompt = "You are a helpful assistant that explains things simply."
        response = provider.generate(
            "What is machine learning?", 
            system_prompt=system_prompt
        )
        
        print(f"Response: {response.content}")
        
        # Test token counting
        print("\n3. Token Counting Test:")
        test_text = "This is a test sentence for token counting."
        token_count = provider.get_token_count(test_text)
        print(f"Text: '{test_text}'")
        print(f"Estimated tokens: {token_count}")
        
        # Show available models
        print("\n4. Available Models:")
        models = provider.list_available_models()
        print(f"Available GPT models: {models[:5]}...")  # Show first 5
        
        print("\n✅ OpenAI Provider demo completed successfully!")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Make sure to set your OPENAI_API_KEY in the .env file")
    except Exception as e:
        print(f"❌ Demo Error: {e}")


if __name__ == "__main__":
    demo_openai_provider()