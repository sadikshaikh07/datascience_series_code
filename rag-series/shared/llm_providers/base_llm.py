"""
Base LLM Provider Interface for RAG Systems
Blog 2.3: From Retrieval to Answers - The Full RAG Pipeline

Unified interface for different LLM providers optimized for RAG pipelines.
Supports both OpenAI and Anthropic with proper context window management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in shared folder first, then parent directories
env_paths = [
    os.path.join(os.path.dirname(__file__), '..', '.env'),  # shared/.env
    os.path.join(os.path.dirname(__file__), '..', '..', '.env'),  # rag-series/.env
    '.env'  # current directory
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
        break
else:
    load_dotenv()  # fallback to default behavior


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers in RAG systems."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class LLMConfig:
    """
    Configuration settings for LLM providers optimized for RAG systems.
    
    RAG-specific considerations:
    - model: Should support sufficient context window for retrieved documents
    - max_tokens: Balance between response quality and cost
    - temperature: Lower for factual RAG responses, higher for creative tasks
    - context_window: Maximum tokens for the model (for RAG context management)
    """
    model: str
    max_tokens: int = 1000
    temperature: float = 0.1  # Lower temperature for factual RAG responses
    top_p: float = 1.0
    timeout: int = 60  # Longer timeout for RAG queries with context
    context_window: Optional[int] = None  # Will be set based on model
    extra_params: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers optimized for RAG systems.
    
    This ensures all providers implement the same interface with RAG-specific
    optimizations like context window management and async support.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        
        # Set context window if not specified
        if not config.context_window:
            self.config.context_window = self._get_default_context_window()
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                      context: Optional[str] = None) -> LLMResponse:
        """
        Generate response from the LLM with optional RAG context.
        
        Args:
            prompt: The user prompt/query
            system_prompt: Optional system prompt for RAG instructions
            context: Retrieved context from RAG system
            
        Returns:
            LLMResponse: Standardized response object
        """
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None,
                     context: Optional[str] = None) -> LLMResponse:
        """Synchronous version of generate for compatibility."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass
    
    def _get_default_context_window(self) -> int:
        """Get default context window for the model."""
        model_context_windows = {
            # OpenAI models
            'gpt-3.5-turbo': 4096,
            'gpt-4': 8192,
            'gpt-4o': 128000,
            'gpt-4-turbo': 128000,
            # Anthropic models
            'claude-3-haiku-20240307': 200000,
            'claude-3-5-sonnet-20241022': 200000,
            'claude-3-5-sonnet-20240620': 200000,
        }
        
        # Try exact match first
        if self.config.model in model_context_windows:
            return model_context_windows[self.config.model]
        
        # Try partial matches
        for model_name, context_window in model_context_windows.items():
            if model_name in self.config.model:
                return context_window
        
        # Default fallback
        return 4096
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for text (provider-specific implementation).
        Default implementation uses rough character-based estimation.
        """
        return len(text) // 4
    
    def calculate_available_context(self, prompt: str, system_prompt: Optional[str] = None,
                                  response_buffer: int = 500) -> int:
        """
        Calculate how many tokens are available for RAG context.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            response_buffer: Tokens to reserve for response
            
        Returns:
            int: Available tokens for context
        """
        prompt_tokens = self.get_token_count(prompt)
        system_tokens = self.get_token_count(system_prompt or "")
        reserved_tokens = self.config.max_tokens + response_buffer
        
        total_used = prompt_tokens + system_tokens + reserved_tokens
        available = self.config.context_window - total_used
        
        return max(0, available)
    
    def validate_config(self) -> bool:
        """Validate that the provider configuration is correct."""
        required_attrs = ['model', 'max_tokens', 'temperature']
        return all(hasattr(self.config, attr) for attr in required_attrs)


def get_default_provider() -> BaseLLMProvider:
    """
    Get the default LLM provider for RAG systems based on environment configuration.
    
    Priority order:
    1. OpenAI/Ollama (if OPENAI_BASE_URL is set for local models, or OPENAI_API_KEY for cloud)
    2. Anthropic (if ANTHROPIC_API_KEY is set)
    3. Simulation mode (for testing without API keys)
    
    Returns:
        BaseLLMProvider: Configured provider
        
    Environment variables:
        OPENAI_API_KEY: OpenAI API key (for cloud)
        OPENAI_BASE_URL: Custom endpoint (for Ollama/LocalAI)
        ANTHROPIC_API_KEY: Anthropic API key
        DEFAULT_LLM_PROVIDER: Force specific provider (openai/anthropic)
    """
    preferred_provider = os.getenv('DEFAULT_LLM_PROVIDER', '').lower()
    
    # Check for API keys and provider availability
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_openai_compatible = bool(os.getenv('OPENAI_BASE_URL'))  # Ollama, LocalAI, etc.
    has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))
    
    # For Ollama/LocalAI, we don't need a real API key
    openai_available = has_openai or has_openai_compatible
    
    try:
        if preferred_provider == 'openai' and openai_available:
            from .openai_provider import get_openai_provider
            provider = get_openai_provider()
            print(f"🤖 Using OpenAI-compatible provider (Ollama/LocalAI)")
            return provider
        elif preferred_provider == 'anthropic' and has_anthropic:
            from .anthropic_provider import get_anthropic_provider
            return get_anthropic_provider()
        elif openai_available:
            from .openai_provider import get_openai_provider
            provider = get_openai_provider()
            if has_openai_compatible:
                print(f"🤖 Auto-detected Ollama/LocalAI configuration")
            return provider
        elif has_anthropic:
            from .anthropic_provider import get_anthropic_provider
            return get_anthropic_provider()
        else:
            # Fallback to simulation mode
            print(f"⚠️ No API keys or local endpoints found, using simulation mode")
            from .simulation_provider import get_simulation_provider
            return get_simulation_provider()
            
    except Exception as e:
        print(f"Warning: Failed to initialize preferred provider: {e}")
        # Fallback to simulation mode
        from .simulation_provider import get_simulation_provider
        return get_simulation_provider()


async def test_provider(provider: BaseLLMProvider) -> bool:
    """
    Test if a provider is working correctly.
    
    Args:
        provider: The provider to test
        
    Returns:
        bool: True if provider is working
    """
    try:
        response = await provider.generate("Say 'Hello, World!' in a friendly way.")
        return len(response.content) > 0
    except Exception as e:
        print(f"Provider test failed: {e}")
        return False


def test_provider_sync(provider: BaseLLMProvider) -> bool:
    """Synchronous version of test_provider."""
    try:
        response = provider.generate_sync("Say 'Hello, World!' in a friendly way.")
        return len(response.content) > 0
    except Exception as e:
        print(f"Provider test failed: {e}")
        return False