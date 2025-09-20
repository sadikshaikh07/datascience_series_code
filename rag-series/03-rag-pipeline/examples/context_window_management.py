"""
Context Window Management Strategies for RAG Systems

This module demonstrates various strategies for managing context windows
when the retrieved content exceeds the model's token limits.

Strategies covered:
1. Truncation - Simple cut-off at token limit
2. Compression - LLM-based content summarization
3. Hierarchical - Structured content organization
4. Sliding Window - Sequential processing with overlap

Model Context Windows:
- GPT-3.5-turbo: 4,096 tokens
- GPT-4: 8,192 tokens (some variants up to 32k)
- GPT-4o: 128,000 tokens
- Claude 3.5 Sonnet: 200,000 tokens
"""

import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    score: float
    source: str
    chunk_id: str

class ContextWindowManager:
    """Manages context window constraints for different models"""
    
    MODEL_LIMITS = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4o": 128000,
        "claude-3-5-sonnet": 200000
    }
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", reserve_tokens: int = 1000):
        """
        Initialize context window manager
        
        Args:
            model_name: Name of the model to manage context for
            reserve_tokens: Tokens to reserve for system prompt and response
        """
        self.model_name = model_name
        self.max_tokens = self.MODEL_LIMITS.get(model_name, 4096)
        self.reserve_tokens = reserve_tokens
        self.available_tokens = self.max_tokens - reserve_tokens
        
        # Initialize tokenizer for GPT models
        if "gpt" in model_name.lower():
            self.encoding = tiktoken.encoding_for_model(model_name)
        else:
            # Fallback for other models
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncation_strategy(self, documents: List[Document], query: str) -> str:
        """
        Strategy 1: Truncation - Simple cut-off at token limit
        
        Pros: Fast, simple
        Cons: May cut important information, no intelligent selection
        """
        context_parts = []
        current_tokens = 0
        query_tokens = self.count_tokens(query)
        
        print(f"🔪 Truncation Strategy")
        print(f"Available tokens: {self.available_tokens}")
        print(f"Query tokens: {query_tokens}")
        
        for doc in documents:
            doc_tokens = self.count_tokens(doc.content)
            
            if current_tokens + doc_tokens + query_tokens <= self.available_tokens:
                context_parts.append(f"Source: {doc.source}\n{doc.content}")
                current_tokens += doc_tokens
                print(f"✅ Added document (tokens: {doc_tokens}, total: {current_tokens})")
            else:
                print(f"❌ Truncated at document limit (would exceed by {current_tokens + doc_tokens - self.available_tokens} tokens)")
                break
        
        return "\n\n---\n\n".join(context_parts)
    
    def compression_strategy(self, documents: List[Document], query: str) -> str:
        """
        Strategy 2: Compression - Use LLM to summarize content
        
        Pros: Preserves key information, intelligent summarization
        Cons: Slower, costs additional API calls, may lose nuance
        """
        print(f"🗜️ Compression Strategy")
        
        full_context = "\n\n---\n\n".join([
            f"Source: {doc.source}\n{doc.content}" for doc in documents
        ])
        
        full_tokens = self.count_tokens(full_context)
        print(f"Original context tokens: {full_tokens}")
        
        if full_tokens <= self.available_tokens:
            print("✅ No compression needed")
            return full_context
        
        # Calculate target compression ratio
        target_tokens = int(self.available_tokens * 0.8)  # Leave some buffer
        compression_ratio = target_tokens / full_tokens
        
        print(f"Target tokens: {target_tokens} (compression ratio: {compression_ratio:.2f})")
        
        # Simulate compression (in real implementation, use LLM API)
        compressed_content = self._simulate_compression(full_context, compression_ratio)
        
        compressed_tokens = self.count_tokens(compressed_content)
        print(f"Compressed to {compressed_tokens} tokens")
        
        return compressed_content
    
    def hierarchical_strategy(self, documents: List[Document], query: str) -> str:
        """
        Strategy 3: Hierarchical - Organize content by relevance/structure
        
        Pros: Maintains logical structure, prioritizes relevant content
        Cons: Requires content understanding, complex implementation
        """
        print(f"🏗️ Hierarchical Strategy")
        
        # Group documents by relevance score
        high_relevance = [doc for doc in documents if doc.score > 0.8]
        medium_relevance = [doc for doc in documents if 0.5 <= doc.score <= 0.8]
        low_relevance = [doc for doc in documents if doc.score < 0.5]
        
        print(f"High relevance: {len(high_relevance)} docs")
        print(f"Medium relevance: {len(medium_relevance)} docs")
        print(f"Low relevance: {len(low_relevance)} docs")
        
        context_parts = []
        current_tokens = 0
        query_tokens = self.count_tokens(query)
        
        # Process in order of relevance
        for priority, docs in [("HIGH", high_relevance), ("MEDIUM", medium_relevance), ("LOW", low_relevance)]:
            if not docs:
                continue
                
            section_header = f"\n=== {priority} RELEVANCE SOURCES ===\n"
            section_tokens = self.count_tokens(section_header)
            
            if current_tokens + section_tokens + query_tokens <= self.available_tokens:
                context_parts.append(section_header)
                current_tokens += section_tokens
                
                for doc in docs:
                    doc_content = f"Source: {doc.source} (Score: {doc.score:.2f})\n{doc.content}"
                    doc_tokens = self.count_tokens(doc_content)
                    
                    if current_tokens + doc_tokens + query_tokens <= self.available_tokens:
                        context_parts.append(doc_content)
                        current_tokens += doc_tokens
                        print(f"✅ Added {priority.lower()} relevance doc (tokens: {doc_tokens})")
                    else:
                        print(f"❌ Stopped at {priority.lower()} relevance section due to token limit")
                        break
            else:
                print(f"❌ Cannot fit {priority.lower()} relevance section")
                break
        
        return "\n\n".join(context_parts)
    
    def sliding_window_strategy(self, documents: List[Document], query: str, window_overlap: float = 0.2) -> List[str]:
        """
        Strategy 4: Sliding Window - Process in overlapping chunks
        
        Pros: Handles large content, maintains context continuity
        Cons: Multiple API calls, potential for inconsistent answers
        """
        print(f"🪟 Sliding Window Strategy")
        
        full_context = "\n\n---\n\n".join([
            f"Source: {doc.source}\n{doc.content}" for doc in documents
        ])
        
        full_tokens = self.count_tokens(full_context)
        query_tokens = self.count_tokens(query)
        
        if full_tokens + query_tokens <= self.available_tokens:
            print("✅ Single window sufficient")
            return [full_context]
        
        # Calculate window size and overlap
        window_size = self.available_tokens - query_tokens
        overlap_size = int(window_size * window_overlap)
        stride = window_size - overlap_size
        
        print(f"Window size: {window_size} tokens")
        print(f"Overlap size: {overlap_size} tokens")
        print(f"Stride: {stride} tokens")
        
        # Split context into tokens for precise windowing
        tokens = self.encoding.encode(full_context)
        windows = []
        
        start = 0
        window_num = 1
        
        while start < len(tokens):
            end = min(start + window_size, len(tokens))
            window_tokens = tokens[start:end]
            window_text = self.encoding.decode(window_tokens)
            
            windows.append(window_text)
            print(f"📖 Window {window_num}: tokens {start}-{end} ({len(window_tokens)} tokens)")
            
            if end >= len(tokens):
                break
                
            start += stride
            window_num += 1
        
        return windows
    
    def _simulate_compression(self, text: str, ratio: float) -> str:
        """
        Simulate text compression (placeholder for actual LLM compression)
        In production, this would use an LLM API call
        """
        sentences = re.split(r'[.!?]+', text)
        target_sentences = max(1, int(len(sentences) * ratio))
        
        # Simple compression: keep first portion of sentences
        compressed_sentences = sentences[:target_sentences]
        compressed_text = '. '.join(compressed_sentences)
        
        # Add compression indicator
        if len(sentences) > target_sentences:
            compressed_text += f"\n\n[Content compressed: {len(sentences)} → {target_sentences} sentences]"
        
        return compressed_text

def demo_context_window_management():
    """Demonstrate all context window management strategies"""
    
    # Sample documents with varying relevance scores
    documents = [
        Document(
            content="Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace.",
            score=0.95,
            source="python_basics.txt",
            chunk_id="chunk_1"
        ),
        Document(
            content="Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
            score=0.85,
            source="ml_intro.txt", 
            chunk_id="chunk_2"
        ),
        Document(
            content="Data structures are a way of organizing and storing data so that they can be accessed and worked with efficiently. They define the relationship between the data, and the operations that can be performed on the data. Common data structures include arrays, linked lists, stacks, queues, trees, and graphs.",
            score=0.75,
            source="data_structures.txt",
            chunk_id="chunk_3"
        ),
        Document(
            content="Version control systems are tools that help manage changes to source code over time. Git is the most popular distributed version control system. It tracks changes in any set of files, usually used for coordinating work among programmers collaboratively developing source code during software development.",
            score=0.45,
            source="git_basics.txt",
            chunk_id="chunk_4"
        ),
        Document(
            content="Web development involves creating websites and web applications. It includes front-end development (what users see and interact with) and back-end development (server-side logic). Popular technologies include HTML, CSS, JavaScript, and various frameworks like React, Angular, and Vue.js.",
            score=0.35,
            source="web_dev.txt",
            chunk_id="chunk_5"
        )
    ]
    
    query = "What is Python and how is it used in machine learning?"
    
    print("=" * 80)
    print("CONTEXT WINDOW MANAGEMENT STRATEGIES DEMO")
    print("=" * 80)
    
    # Test with limited context window (simulating gpt-3.5-turbo)
    manager = ContextWindowManager("gpt-3.5-turbo", reserve_tokens=500)
    
    print(f"\nModel: {manager.model_name}")
    print(f"Max tokens: {manager.max_tokens}")
    print(f"Available for context: {manager.available_tokens}")
    print(f"Query: {query}\n")
    
    # Strategy 1: Truncation
    print("\n" + "="*50)
    truncated_context = manager.truncation_strategy(documents, query)
    print(f"Result length: {len(truncated_context)} chars")
    
    # Strategy 2: Compression
    print("\n" + "="*50)
    compressed_context = manager.compression_strategy(documents, query)
    print(f"Result length: {len(compressed_context)} chars")
    
    # Strategy 3: Hierarchical
    print("\n" + "="*50)
    hierarchical_context = manager.hierarchical_strategy(documents, query)
    print(f"Result length: {len(hierarchical_context)} chars")
    
    # Strategy 4: Sliding Window
    print("\n" + "="*50)
    windows = manager.sliding_window_strategy(documents, query)
    print(f"Number of windows: {len(windows)}")
    for i, window in enumerate(windows, 1):
        print(f"Window {i} length: {len(window)} chars")
    
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    print("📊 Truncation: Fast, simple, may lose important info")
    print("📊 Compression: Preserves key info, requires extra API calls")
    print("📊 Hierarchical: Prioritizes relevant content, maintains structure")
    print("📊 Sliding Window: Handles large content, multiple processing steps")

if __name__ == "__main__":
    demo_context_window_management()