"""
LLM Providers Package
Blog 2: Prompt Engineering Fundamentals

This package provides a unified interface for different LLM providers,
making it easy to switch between OpenAI, Anthropic, Ollama, and others.

Usage:
    from llm_providers import get_default_provider
    
    # Quick start with default provider
    provider = get_default_provider()
    response = provider.generate("Your prompt here")
    
    # Or create specific provider directly
    from llm_providers import OpenAIProvider
    provider = OpenAIProvider()
    response = provider.generate("Your prompt here")
"""

from .base_llm import (
    BaseLLMProvider,
    LLMResponse,
    LLMConfig,
    get_default_provider,
    test_provider
)

from .openai_provider import OpenAIProvider

__all__ = [
    'BaseLLMProvider',
    'LLMResponse', 
    'LLMConfig',
    'get_default_provider',
    'test_provider',
    'OpenAIProvider',
]


def get_available_providers() -> dict:
    """
    Get information about available providers.
    
    Returns:
        dict: Provider availability and status
    """
    providers = {
        'openai': {
            'available': True,
            'description': 'OpenAI GPT models (GPT-4, GPT-3.5-turbo)',
            'requires': 'openai package and API key'
        }
    }
    
    return providers
