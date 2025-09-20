"""
Anthropic Provider Implementation for RAG Systems
Blog 2.3: From Retrieval to Answers - The Full RAG Pipeline

Anthropic Claude implementation optimized for RAG pipelines with excellent
context window utilization and strong reasoning capabilities.
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any

from .base_llm import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from anthropic import AsyncAnthropic, Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None
    Anthropic = None


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider optimized for RAG systems.
    
    Features:
    - Excellent long context handling (200k tokens)
    - Strong reasoning capabilities for complex RAG queries
    - Async and sync support
    - Cost-effective for large context windows
    """
    
    def __init__(self, config: LLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        super().__init__(config)
        
        # Initialize both sync and async clients
        client_params = {
            "api_key": os.getenv('ANTHROPIC_API_KEY'),
            "timeout": config.timeout
        }
        
        self.client = Anthropic(**client_params)
        self.async_client = AsyncAnthropic(**client_params)
        self.provider_name = "anthropic"
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                      context: Optional[str] = None) -> LLMResponse:
        """Generate response using Anthropic Claude API (async)."""
        
        # Build RAG-optimized system prompt
        if system_prompt:
            if context:
                full_system_prompt = f"{system_prompt}\n\nContext:\n{context}"
            else:
                full_system_prompt = system_prompt
        elif context:
            # Default RAG system prompt if none provided
            full_system_prompt = f"""You are a helpful assistant. Use the provided context to answer questions accurately and comprehensively. If the context doesn't contain relevant information, clearly state that and provide what general knowledge you can while being honest about the limitations.

Context:
{context}"""
        else:
            full_system_prompt = "You are a helpful assistant."
        
        try:
            start_time = time.time()
            
            response = await self.async_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                system=full_system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **(self.config.extra_params or {})
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract response content
            content = response.content[0].text if response.content else ""
            
            # Build usage information
            usage = {
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
                "response_time": elapsed_time,
                "cost_estimate": self._estimate_cost(response.usage) if response.usage else 0
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "stop_reason": response.stop_reason,
                    "response_id": response.id,
                    "context_provided": context is not None,
                    "context_length": len(context) if context else 0
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")
    
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None,
                     context: Optional[str] = None) -> LLMResponse:
        """Synchronous version of generate."""
        # Build RAG-optimized system prompt
        if system_prompt:
            if context:
                full_system_prompt = f"{system_prompt}\n\nContext:\n{context}"
            else:
                full_system_prompt = system_prompt
        elif context:
            # Default RAG system prompt if none provided
            full_system_prompt = f"""You are a helpful assistant. Use the provided context to answer questions accurately and comprehensively. If the context doesn't contain relevant information, clearly state that and provide what general knowledge you can while being honest about the limitations.

Context:
{context}"""
        else:
            full_system_prompt = "You are a helpful assistant."
        
        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                system=full_system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **(self.config.extra_params or {})
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract response content
            content = response.content[0].text if response.content else ""
            
            # Build usage information
            usage = {
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
                "response_time": elapsed_time,
                "cost_estimate": self._estimate_cost(response.usage) if response.usage else 0
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "stop_reason": response.stop_reason,
                    "response_id": response.id,
                    "context_provided": context is not None,
                    "context_length": len(context) if context else 0
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Anthropic provider is properly configured."""
        if not ANTHROPIC_AVAILABLE:
            return False
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return False
        
        try:
            # Test with a simple request
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Test"}]
            )
            return True
        except Exception:
            return False
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for Anthropic models.
        Claude uses a similar tokenization to GPT models.
        """
        # Anthropic's tokenization is similar to OpenAI's
        # Use slightly more conservative estimate
        return int(len(text) / 3.5)
    
    def _estimate_cost(self, usage) -> float:
        """
        Estimate cost for Anthropic API call based on current pricing.
        
        Args:
            usage: Anthropic usage object
            
        Returns:
            float: Estimated cost in USD
        """
        # Approximate pricing (check Anthropic pricing page for current rates)
        pricing = {
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},  # per 1K tokens
            'claude-3-5-sonnet-20240620': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
            'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075}
        }
        
        # Determine model pricing
        rates = pricing.get(self.config.model, pricing['claude-3-5-sonnet-20241022'])
        
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        
        input_cost = (input_tokens / 1000) * rates['input']
        output_cost = (output_tokens / 1000) * rates['output']
        
        return input_cost + output_cost


def get_anthropic_provider(model: Optional[str] = None) -> AnthropicProvider:
    """
    Convenience function to create an Anthropic provider for RAG systems.
    
    Args:
        model: Model to use (default: claude-3-5-sonnet for best performance)
        
    Returns:
        Configured Anthropic provider optimized for RAG
    """
    if not model:
        model = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')
    
    config = LLMConfig(
        model=model,
        max_tokens=int(os.getenv('ANTHROPIC_MAX_TOKENS', '2000')),  # Higher for detailed RAG responses
        temperature=float(os.getenv('ANTHROPIC_TEMPERATURE', '0.1')),  # Lower for factual responses
        top_p=float(os.getenv('ANTHROPIC_TOP_P', '1.0')),
        timeout=int(os.getenv('ANTHROPIC_TIMEOUT', '60'))  # Longer for RAG
    )
    
    return AnthropicProvider(config)


async def demo_anthropic_rag():
    """Demonstrate Anthropic provider with RAG context."""
    print("=== Anthropic RAG Provider Demo ===\n")
    
    try:
        provider = get_anthropic_provider()
        
        print(f"✅ Anthropic Provider initialized with model: {provider.config.model}")
        print(f"📊 Provider available: {provider.is_available()}")
        
        if not provider.is_available():
            print("⚠️ Provider not available - likely missing API key")
            return
        
        # Test with RAG context
        context = """
        Retrieval-Augmented Generation (RAG) is a technique that combines pre-trained language models with external knowledge retrieval. 
        It works by first retrieving relevant documents from a knowledge base, then using these documents as context for generating responses.
        RAG helps address limitations like knowledge cutoffs and hallucination in large language models.
        The technique was introduced by Facebook AI Research in 2020 and has become widely adopted for building knowledge-intensive applications.
        """
        
        print("\n1. RAG-style Query with Context:")
        response = await provider.generate(
            prompt="What is RAG and why is it useful?",
            context=context
        )
        
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Cost estimate: ${response.usage.get('cost_estimate', 0):.6f}")
        
        print("\n✅ Anthropic RAG Provider demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")


if __name__ == "__main__":
    asyncio.run(demo_anthropic_rag())