"""
LLM Providers Package for RAG Series
Blog 2.3: From Retrieval to Answers - The Full RAG Pipeline

This package provides a unified interface for different LLM providers used in RAG systems,
making it easy to switch between OpenAI, Anthropic, and others while maintaining compatibility
with the RAG pipeline components.

Usage:
    from shared.llm_providers import get_default_provider
    
    # Quick start with default provider
    provider = get_default_provider()
    response = await provider.generate("Your prompt here")
    
    # Or create specific provider directly
    from shared.llm_providers import OpenAIProvider, AnthropicProvider
    provider = OpenAIProvider()
    response = await provider.generate("Your prompt here")
"""

from .base_llm import (
    BaseLLMProvider,
    LLMResponse,
    LLMConfig,
    get_default_provider,
    test_provider
)

from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

__all__ = [
    'BaseLLMProvider',
    'LLMResponse', 
    'LLMConfig',
    'get_default_provider',
    'test_provider',
    'OpenAIProvider',
    'AnthropicProvider',
]


def get_available_providers() -> dict:
    """
    Get information about available providers for RAG systems.
    
    Returns:
        dict: Provider availability and status
    """
    providers = {
        'openai': {
            'available': True,
            'description': 'OpenAI GPT models (GPT-4, GPT-3.5-turbo, GPT-4o)',
            'requires': 'openai package and OPENAI_API_KEY',
            'context_windows': {
                'gpt-3.5-turbo': 4096,
                'gpt-4': 8192,
                'gpt-4o': 128000
            }
        },
        'anthropic': {
            'available': True,
            'description': 'Anthropic Claude models (Claude-3.5-Sonnet, Claude-3-Haiku)',
            'requires': 'anthropic package and ANTHROPIC_API_KEY',
            'context_windows': {
                'claude-3-haiku-20240307': 200000,
                'claude-3-5-sonnet-20241022': 200000
            }
        }
    }
    
    return providers