"""
OpenAI LLM Provider Interface
Blog 2: Prompt Engineering Fundamentals

Uses OpenAI API with configurable base_url to support:
- OpenAI (official API)
- OpenAI-compatible APIs (Ollama, LocalAI, etc.)
- Other providers can be added following similar patterns

NOTE: We focus on OpenAI for simplicity. Other providers like Anthropic, Google, etc.
can be added using their specific documentation while following this same structure.
The bonus blog covers OpenAI and its configurations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# WHAT: python-dotenv loads API keys and config from .env file into environment variables
# WHY:  Keeps sensitive credentials secure and separate from code
# HOW:  Create .env file from .env.example and add your actual API keys
load_dotenv()


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class LLMConfig:
    """
    Configuration settings for OpenAI and compatible APIs.
    
    Hyperparameter explanations:
    - model: Which AI model to use (gpt-4, gpt-3.5-turbo, etc.)
    - max_tokens: Maximum response length (higher = longer responses, more cost)
    - temperature: Creativity/randomness (0.0=deterministic, 1.0=creative, 2.0=very random)
    - top_p: Nucleus sampling (0.1=focused, 1.0=diverse vocabulary)
    - base_url: API endpoint (allows using Ollama, LocalAI, etc.)
    """
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7  # Balanced creativity for educational examples
    top_p: float = 1.0        # Full vocabulary diversity
    timeout: int = 30         # 30 seconds should be sufficient for most tasks
    base_url: Optional[str] = None  # None = OpenAI official, or custom URL for compatibility
    extra_params: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This ensures all providers implement the same interface,
    making it easy to switch between OpenAI, VertexAI, Ollama, etc.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate response from the LLM.
        
        Args:
            prompt: The user prompt/query
            system_prompt: Optional system prompt for context
            
        Returns:
            LLMResponse: Standardized response object
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for text (provider-specific implementation).
        Default implementation uses rough character-based estimation.
        """
        # Rough estimation: ~4 characters per token for most models
        return len(text) // 4
    
    def validate_config(self) -> bool:
        """Validate that the provider configuration is correct."""
        required_attrs = ['model', 'max_tokens', 'temperature']
        return all(hasattr(self.config, attr) for attr in required_attrs)


def get_openai_config() -> LLMConfig:
    """Get default OpenAI configuration from environment variables."""
    return LLMConfig(
        model=os.getenv('OPENAI_MODEL', 'gpt-4'),
        max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1000')),
        temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
        top_p=float(os.getenv('OPENAI_TOP_P', '1.0')),
        base_url=os.getenv('OPENAI_BASE_URL')  # For Ollama, LocalAI, etc.
    )


def get_default_provider() -> BaseLLMProvider:
    """
    Get the default OpenAI provider based on environment configuration.
    
    Uses OpenAI by default, with support for OpenAI-compatible APIs via OPENAI_BASE_URL.
    
    One-liner usage: 
        provider = get_default_provider()
        response = provider.generate("Your prompt here")
    
    Returns:
        BaseLLMProvider: Configured OpenAI provider
        
    Environment variables:
        OPENAI_API_KEY: Your OpenAI API key (required)
        OPENAI_MODEL: Model to use (default: gpt-4)
        OPENAI_BASE_URL: Custom endpoint for Ollama/LocalAI (optional)
        OPENAI_MAX_TOKENS: Max response length (default: 1000)
        OPENAI_TEMPERATURE: Creativity level 0-2 (default: 0.7)
    """
    # Import here to avoid circular imports
    from .openai_provider import get_openai_provider
    
    model = os.getenv('OPENAI_MODEL', 'gpt-4')
    base_url = os.getenv('OPENAI_BASE_URL')
    
    try:
        print (model, base_url)
        return get_openai_provider(model, base_url)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI provider: {e}. Please check your OPENAI_API_KEY and configuration.")


# Utility function for quick testing
def test_provider(provider: BaseLLMProvider) -> bool:
    """
    Test if a provider is working correctly.
    
    Args:
        provider: The provider to test
        
    Returns:
        bool: True if provider is working
    """
    try:
        response = provider.generate("Say 'Hello, World!' in a friendly way.")
        return len(response.content) > 0
    except Exception as e:
        print(f"Provider test failed: {e}")
        return False