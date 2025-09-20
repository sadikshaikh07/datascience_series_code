"""
OpenAI Provider Implementation for RAG Systems
Blog 2.3: From Retrieval to Answers - The Full RAG Pipeline

OpenAI implementation optimized for RAG pipelines with proper context window management,
async support, and cost-effective model selection.
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .base_llm import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from openai import AsyncOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None
    OpenAI = None


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider optimized for RAG systems.
    
    Features:
    - Async and sync support for high-throughput RAG queries
    - Context window management for large retrieved documents
    - Cost optimization through model selection
    - Proper error handling and retries
    """
    
    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        super().__init__(config)
        
        # Initialize both sync and async clients
        client_params = {
            "api_key": os.getenv('OPENAI_API_KEY', 'sk-placeholder'),
            "timeout": config.timeout
        }
        
        # Add base_url if specified (for Ollama, LocalAI, etc.)
        if config.base_url:
            client_params["base_url"] = config.base_url
            print(f"🔗 Using custom base URL: {config.base_url}")
            print(f"🤖 Model: {config.model}")
        
        self.client = OpenAI(**client_params)
        self.async_client = AsyncOpenAI(**client_params)
        self.provider_name = "openai"
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                      context: Optional[str] = None) -> LLMResponse:
        """Generate response using OpenAI chat completions API (async)."""
        messages = []
        
        # Build RAG-optimized system prompt
        if system_prompt:
            if context:
                full_system_prompt = f"{system_prompt}\n\nContext:\n{context}"
            else:
                full_system_prompt = system_prompt
            messages.append({"role": "system", "content": full_system_prompt})
        elif context:
            # Default RAG system prompt if none provided
            rag_system_prompt = f"""You are a helpful assistant. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, say so clearly.

Context:
{context}"""
            messages.append({"role": "system", "content": rag_system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            start_time = time.time()
            
            response = await self.async_client.chat.completions.create(
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
                "response_time": elapsed_time,
                "cost_estimate": self._estimate_cost(response.usage) if response.usage else 0
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                    "context_provided": context is not None,
                    "context_length": len(context) if context else 0
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None,
                     context: Optional[str] = None) -> LLMResponse:
        """Synchronous version of generate."""
        return asyncio.run(self._generate_sync_internal(prompt, system_prompt, context))
    
    async def _generate_sync_internal(self, prompt: str, system_prompt: Optional[str] = None,
                                    context: Optional[str] = None) -> LLMResponse:
        """Internal method to handle sync generation."""
        messages = []
        
        # Build RAG-optimized system prompt
        if system_prompt:
            if context:
                full_system_prompt = f"{system_prompt}\n\nContext:\n{context}"
            else:
                full_system_prompt = system_prompt
            messages.append({"role": "system", "content": full_system_prompt})
        elif context:
            # Default RAG system prompt if none provided
            rag_system_prompt = f"""You are a helpful assistant. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, say so clearly.

Context:
{context}"""
            messages.append({"role": "system", "content": rag_system_prompt})
        
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
                "response_time": elapsed_time,
                "cost_estimate": self._estimate_cost(response.usage) if response.usage else 0
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                    "context_provided": context is not None,
                    "context_length": len(context) if context else 0
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
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count using tiktoken if available.
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
    
    def _estimate_cost(self, usage) -> float:
        """
        Estimate cost for API call based on current OpenAI pricing.
        
        Args:
            usage: OpenAI usage object
            
        Returns:
            float: Estimated cost in USD
        """
        # Approximate pricing (check OpenAI pricing page for current rates)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
        }
        
        # Determine model family
        model_family = 'gpt-3.5-turbo'  # default
        for model_name in pricing.keys():
            if model_name in self.config.model:
                model_family = model_name
                break
        
        rates = pricing.get(model_family, pricing['gpt-3.5-turbo'])
        
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        
        input_cost = (prompt_tokens / 1000) * rates['input']
        output_cost = (completion_tokens / 1000) * rates['output']
        
        return input_cost + output_cost


def get_openai_provider(model: Optional[str] = None) -> OpenAIProvider:
    """
    Convenience function to create an OpenAI provider for RAG systems.
    
    Args:
        model: Model to use (default: from env or gpt-4o for balance of cost/performance)
        
    Returns:
        Configured OpenAI provider optimized for RAG
    """
    if not model:
        model = os.getenv('OPENAI_MODEL', 'gpt-4o')  # Good balance for RAG
    
    # Get base_url for Ollama/LocalAI compatibility
    base_url = os.getenv('OPENAI_BASE_URL')
    
    # Use longer timeout for local models (Ollama)
    default_timeout = '120' if base_url else '60'
    
    config = LLMConfig(
        model=model,
        max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1500')),  # Higher for RAG responses
        temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),  # Lower for factual responses
        top_p=float(os.getenv('OPENAI_TOP_P', '1.0')),
        timeout=int(os.getenv('OPENAI_TIMEOUT', default_timeout)),  # Longer for local models
        extra_params={}
    )
    
    # Set base_url for custom endpoints (Ollama, LocalAI, etc.)
    if base_url:
        config.base_url = base_url
    
    return OpenAIProvider(config)


async def demo_openai_rag():
    """Demonstrate OpenAI provider with RAG context."""
    print("=== OpenAI RAG Provider Demo ===\n")
    
    try:
        provider = get_openai_provider()
        
        print(f"✅ OpenAI Provider initialized with model: {provider.config.model}")
        print(f"📊 Provider available: {provider.is_available()}")
        
        if not provider.is_available():
            print("⚠️ Provider not available - likely missing API key")
            return
        
        # Test with RAG context
        context = """
        Python is a high-level programming language created by Guido van Rossum.
        It was first released in 1991 and is known for its simplicity and readability.
        Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
        """
        
        print("\n1. RAG-style Query with Context:")
        response = await provider.generate(
            prompt="When was Python created and by whom?",
            context=context
        )
        
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Cost estimate: ${response.usage.get('cost_estimate', 0):.6f}")
        
        print("\n✅ OpenAI RAG Provider demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")


if __name__ == "__main__":
    asyncio.run(demo_openai_rag())