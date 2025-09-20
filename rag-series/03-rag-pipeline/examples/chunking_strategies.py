"""
Advanced Chunking Strategies for RAG Systems

This module demonstrates various text chunking strategies for optimal
retrieval performance in RAG systems.

Strategies covered:
1. Fixed-Size Chunking - Equal token/character splits
2. Sliding Window with Overlap - Overlapping chunks for context preservation  
3. Semantic Splitting - Content-aware boundaries
4. Hybrid Chunking - Combination of semantic and fixed approaches
5. Recursive Splitting - Hierarchical document breakdown
6. Multi-Granularity - Multiple chunk sizes for different use cases

Each strategy has different trade-offs in terms of semantic coherence,
retrieval accuracy, and computational complexity.
"""

import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# Handle optional dependencies gracefully
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy not available. Semantic chunking will use fallback methods.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not available. Semantic chunking will be limited.")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Some advanced features will be limited.")

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    start_idx: int
    end_idx: int
    chunk_id: str
    metadata: Dict[str, Any]
    token_count: int
    overlap_with: List[str] = None  # IDs of overlapping chunks

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[Chunk]:
        """Split text into chunks"""
        pass

class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with configurable size and overlap"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, unit: str = "tokens"):
        """
        Initialize fixed-size chunker
        
        Args:
            chunk_size: Size of each chunk
            overlap: Overlap between consecutive chunks
            unit: 'tokens' or 'characters'
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.unit = unit
        
        if unit == "tokens":
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, **kwargs) -> List[Chunk]:
        """Create fixed-size chunks"""
        print(f"🔧 Fixed-Size Chunking (size: {self.chunk_size} {self.unit}, overlap: {self.overlap})")
        
        chunks = []
        
        if self.unit == "tokens":
            tokens = self.encoding.encode(text)
            total_tokens = len(tokens)
            
            start = 0
            chunk_id = 0
            
            while start < total_tokens:
                end = min(start + self.chunk_size, total_tokens)
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                # Find character positions for metadata
                char_start = len(self.encoding.decode(tokens[:start]))
                char_end = len(self.encoding.decode(tokens[:end]))
                
                chunk = Chunk(
                    content=chunk_text,
                    start_idx=char_start,
                    end_idx=char_end,
                    chunk_id=f"fixed_{chunk_id}",
                    metadata={
                        "strategy": "fixed_size",
                        "token_start": start,
                        "token_end": end,
                        "overlap_tokens": self.overlap if chunk_id > 0 else 0
                    },
                    token_count=len(chunk_tokens)
                )
                
                chunks.append(chunk)
                
                if end >= total_tokens:
                    break
                    
                start = end - self.overlap
                chunk_id += 1
        
        else:  # characters
            text_length = len(text)
            start = 0
            chunk_id = 0
            
            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk_text = text[start:end]
                
                chunk = Chunk(
                    content=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    chunk_id=f"fixed_{chunk_id}",
                    metadata={
                        "strategy": "fixed_size",
                        "overlap_chars": self.overlap if chunk_id > 0 else 0
                    },
                    token_count=len(self.encoding.encode(chunk_text)) if hasattr(self, 'encoding') else 0
                )
                
                chunks.append(chunk)
                
                if end >= text_length:
                    break
                    
                start = end - self.overlap
                chunk_id += 1
        
        print(f"Created {len(chunks)} chunks")
        return chunks

class SlidingWindowChunker(ChunkingStrategy):
    """Sliding window chunking with intelligent overlap"""
    
    def __init__(self, window_size: int = 512, stride: int = 256):
        """
        Initialize sliding window chunker
        
        Args:
            window_size: Size of each window in tokens
            stride: Step size between windows
        """
        self.window_size = window_size
        self.stride = stride
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, **kwargs) -> List[Chunk]:
        """Create sliding window chunks with overlap tracking"""
        print(f"🪟 Sliding Window Chunking (window: {self.window_size}, stride: {self.stride})")
        
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < total_tokens:
            end = min(start + self.window_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Calculate overlap with previous chunk
            overlap_tokens = max(0, self.window_size - self.stride) if chunk_id > 0 else 0
            overlap_with = [f"sliding_{chunk_id-1}"] if chunk_id > 0 else []
            
            # Find character positions
            char_start = len(self.encoding.decode(tokens[:start]))
            char_end = len(self.encoding.decode(tokens[:end]))
            
            chunk = Chunk(
                content=chunk_text,
                start_idx=char_start,
                end_idx=char_end,
                chunk_id=f"sliding_{chunk_id}",
                metadata={
                    "strategy": "sliding_window",
                    "window_size": self.window_size,
                    "stride": self.stride,
                    "overlap_tokens": overlap_tokens,
                    "token_start": start,
                    "token_end": end
                },
                token_count=len(chunk_tokens),
                overlap_with=overlap_with
            )
            
            chunks.append(chunk)
            
            if end >= total_tokens:
                break
                
            start += self.stride
            chunk_id += 1
        
        print(f"Created {len(chunks)} overlapping windows")
        return chunks

class SemanticChunker(ChunkingStrategy):
    """Semantic chunking using sentence boundaries and embeddings"""
    
    def __init__(self, max_chunk_size: int = 512, similarity_threshold: float = 0.7):
        """
        Initialize semantic chunker
        
        Args:
            max_chunk_size: Maximum chunk size in tokens
            similarity_threshold: Threshold for semantic similarity grouping
        """
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Load spaCy model for sentence segmentation
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️ spaCy model 'en_core_web_sm' not found. Using basic sentence splitting.")
                self.nlp = None
        
        # Load sentence transformer for embeddings
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer model: {e}")
                self.sentence_model = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to regex-based splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, **kwargs) -> List[Chunk]:
        """Create semantically coherent chunks"""
        print(f"🧠 Semantic Chunking (max_size: {self.max_chunk_size}, threshold: {self.similarity_threshold})")
        
        sentences = self._split_into_sentences(text)
        print(f"Split into {len(sentences)} sentences")
        
        if len(sentences) <= 1:
            # Single sentence or very short text
            return [Chunk(
                content=text,
                start_idx=0,
                end_idx=len(text),
                chunk_id="semantic_0",
                metadata={"strategy": "semantic", "sentence_count": len(sentences)},
                token_count=len(self.encoding.encode(text))
            )]
        
        # Get sentence embeddings
        if not self.sentence_model:
            print("⚠️ Sentence transformer not available, falling back to simple chunking")
            # Fallback to simple sentence-based chunking without similarity
            return self._simple_sentence_chunking(sentences, text)
        
        embeddings = self.sentence_model.encode(sentences)
        
        # Group sentences into semantically similar chunks
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.encoding.encode(sentence))
            
            # Check if adding this sentence would exceed token limit
            if current_chunk_tokens + sentence_tokens > self.max_chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk = self._create_semantic_chunk(current_chunk_sentences, text, chunk_id)
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
                chunk_id += 1
            
            elif not current_chunk_sentences:
                # First sentence in chunk
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
            
            else:
                # Check semantic similarity with current chunk
                current_embedding = np.mean([embeddings[j] for j in range(i - len(current_chunk_sentences), i)], axis=0)
                new_embedding = embeddings[i]
                
                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity([current_embedding], [new_embedding])[0][0]
                else:
                    # Fallback cosine similarity calculation
                    dot_product = np.dot(current_embedding, new_embedding)
                    norms = np.linalg.norm(current_embedding) * np.linalg.norm(new_embedding)
                    similarity = dot_product / norms if norms > 0 else 0
                
                if similarity >= self.similarity_threshold:
                    # Add to current chunk
                    current_chunk_sentences.append(sentence)
                    current_chunk_tokens += sentence_tokens
                else:
                    # Create chunk and start new one
                    chunk = self._create_semantic_chunk(current_chunk_sentences, text, chunk_id)
                    chunks.append(chunk)
                    
                    current_chunk_sentences = [sentence]
                    current_chunk_tokens = sentence_tokens
                    chunk_id += 1
        
        # Create final chunk if sentences remain
        if current_chunk_sentences:
            chunk = self._create_semantic_chunk(current_chunk_sentences, text, chunk_id)
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _create_semantic_chunk(self, sentences: List[str], original_text: str, chunk_id: int) -> Chunk:
        """Create a chunk from a list of sentences"""
        chunk_text = " ".join(sentences)
        
        # Find start and end positions in original text
        start_idx = original_text.find(sentences[0])
        end_idx = start_idx + len(chunk_text)
        
        return Chunk(
            content=chunk_text,
            start_idx=start_idx,
            end_idx=end_idx,
            chunk_id=f"semantic_{chunk_id}",
            metadata={
                "strategy": "semantic",
                "sentence_count": len(sentences),
                "avg_sentence_length": len(chunk_text) / len(sentences)
            },
            token_count=len(self.encoding.encode(chunk_text))
        )
    
    def _simple_sentence_chunking(self, sentences: List[str], original_text: str) -> List[Chunk]:
        """Fallback method for sentence-based chunking without embeddings"""
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            # Check if adding this sentence would exceed token limit
            if current_chunk_tokens + sentence_tokens > self.max_chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk = self._create_semantic_chunk(current_chunk_sentences, original_text, chunk_id)
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
                chunk_id += 1
            else:
                # Add to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
        
        # Create final chunk if sentences remain
        if current_chunk_sentences:
            chunk = self._create_semantic_chunk(current_chunk_sentences, original_text, chunk_id)
            chunks.append(chunk)
        
        return chunks

class HybridChunker(ChunkingStrategy):
    """Hybrid chunking combining semantic and fixed-size approaches"""
    
    def __init__(self, target_size: int = 512, max_deviation: int = 128):
        """
        Initialize hybrid chunker
        
        Args:
            target_size: Target chunk size in tokens
            max_deviation: Maximum allowed deviation from target size
        """
        self.target_size = target_size
        self.max_deviation = max_deviation
        self.semantic_chunker = SemanticChunker(max_chunk_size=target_size + max_deviation)
        self.fixed_chunker = FixedSizeChunker(chunk_size=target_size)
    
    def chunk_text(self, text: str, **kwargs) -> List[Chunk]:
        """Create hybrid chunks using semantic boundaries with size constraints"""
        print(f"🔄 Hybrid Chunking (target: {self.target_size}±{self.max_deviation} tokens)")
        
        # Start with semantic chunking
        semantic_chunks = self.semantic_chunker.chunk_text(text)
        
        refined_chunks = []
        chunk_id = 0
        
        for semantic_chunk in semantic_chunks:
            chunk_size = semantic_chunk.token_count
            
            if chunk_size <= self.target_size + self.max_deviation:
                # Chunk is within acceptable range
                refined_chunk = Chunk(
                    content=semantic_chunk.content,
                    start_idx=semantic_chunk.start_idx,
                    end_idx=semantic_chunk.end_idx,
                    chunk_id=f"hybrid_{chunk_id}",
                    metadata={
                        "strategy": "hybrid",
                        "refinement": "kept_semantic",
                        "original_size": chunk_size,
                        "target_size": self.target_size
                    },
                    token_count=chunk_size
                )
                refined_chunks.append(refined_chunk)
                chunk_id += 1
            
            else:
                # Chunk is too large, apply fixed-size splitting
                sub_chunks = self.fixed_chunker.chunk_text(semantic_chunk.content)
                
                for sub_chunk in sub_chunks:
                    refined_chunk = Chunk(
                        content=sub_chunk.content,
                        start_idx=semantic_chunk.start_idx + sub_chunk.start_idx,
                        end_idx=semantic_chunk.start_idx + sub_chunk.end_idx,
                        chunk_id=f"hybrid_{chunk_id}",
                        metadata={
                            "strategy": "hybrid",
                            "refinement": "split_oversized",
                            "original_size": chunk_size,
                            "target_size": self.target_size,
                            "parent_chunk": semantic_chunk.chunk_id
                        },
                        token_count=sub_chunk.token_count
                    )
                    refined_chunks.append(refined_chunk)
                    chunk_id += 1
        
        print(f"Refined {len(semantic_chunks)} semantic chunks into {len(refined_chunks)} hybrid chunks")
        return refined_chunks

class RecursiveChunker(ChunkingStrategy):
    """Recursive chunking with hierarchical document structure"""
    
    def __init__(self, chunk_sizes: List[int] = [1024, 512, 256]):
        """
        Initialize recursive chunker
        
        Args:
            chunk_sizes: List of chunk sizes to try in order (largest to smallest)
        """
        self.chunk_sizes = sorted(chunk_sizes, reverse=True)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, **kwargs) -> List[Chunk]:
        """Create hierarchical chunks recursively"""
        print(f"🌳 Recursive Chunking (sizes: {self.chunk_sizes})")
        
        chunks = []
        self._recursive_split(text, 0, len(text), 0, chunks, 0)
        
        print(f"Created {len(chunks)} recursive chunks")
        return chunks
    
    def _recursive_split(self, text: str, start: int, end: int, level: int, chunks: List[Chunk], chunk_id_counter: int) -> int:
        """Recursively split text into chunks"""
        segment = text[start:end]
        segment_tokens = len(self.encoding.encode(segment))
        
        # Check if current segment fits in any chunk size
        target_size = self.chunk_sizes[min(level, len(self.chunk_sizes) - 1)]
        
        if segment_tokens <= target_size or level >= len(self.chunk_sizes):
            # Create leaf chunk
            chunk = Chunk(
                content=segment,
                start_idx=start,
                end_idx=end,
                chunk_id=f"recursive_{chunk_id_counter}",
                metadata={
                    "strategy": "recursive",
                    "level": level,
                    "target_size": target_size,
                    "is_leaf": True
                },
                token_count=segment_tokens
            )
            chunks.append(chunk)
            return chunk_id_counter + 1
        
        else:
            # Split at natural boundaries (paragraphs, sentences)
            split_points = self._find_split_points(segment, target_size)
            
            if not split_points:
                # Force split if no natural boundaries found
                mid_point = len(segment) // 2
                split_points = [mid_point]
            
            current_start = start
            for split_point in split_points:
                absolute_split = start + split_point
                chunk_id_counter = self._recursive_split(
                    text, current_start, absolute_split, level + 1, chunks, chunk_id_counter
                )
                current_start = absolute_split
            
            # Handle remaining text
            if current_start < end:
                chunk_id_counter = self._recursive_split(
                    text, current_start, end, level + 1, chunks, chunk_id_counter
                )
            
            return chunk_id_counter
    
    def _find_split_points(self, text: str, target_size: int) -> List[int]:
        """Find natural split points in text"""
        # Look for paragraph breaks first
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            current_pos = 0
            split_points = []
            
            for para in paragraphs[:-1]:  # Exclude last paragraph
                current_pos += len(para) + 2  # +2 for \n\n
                split_points.append(current_pos)
            
            return split_points
        
        # Fall back to sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            current_pos = 0
            split_points = []
            
            for sentence in sentences[:-1]:  # Exclude last sentence
                current_pos += len(sentence) + 1  # +1 for punctuation
                split_points.append(current_pos)
            
            return split_points
        
        return []

class MultiGranularityChunker(ChunkingStrategy):
    """Multi-granularity chunking for different retrieval scenarios"""
    
    def __init__(self, granularities: Dict[str, int] = None):
        """
        Initialize multi-granularity chunker
        
        Args:
            granularities: Dict mapping granularity names to chunk sizes
        """
        self.granularities = granularities or {
            "fine": 256,      # Precise retrieval
            "medium": 512,    # Balanced approach  
            "coarse": 1024    # Broad context
        }
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, **kwargs) -> List[Chunk]:
        """Create chunks at multiple granularities"""
        print(f"📊 Multi-Granularity Chunking (levels: {list(self.granularities.keys())})")
        
        all_chunks = []
        
        for granularity_name, chunk_size in self.granularities.items():
            print(f"  Creating {granularity_name} chunks (size: {chunk_size})")
            
            chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_size//4)
            granularity_chunks = chunker.chunk_text(text)
            
            # Update metadata and IDs for granularity level
            for i, chunk in enumerate(granularity_chunks):
                chunk.chunk_id = f"{granularity_name}_{i}"
                chunk.metadata.update({
                    "granularity": granularity_name,
                    "granularity_size": chunk_size,
                    "strategy": "multi_granularity"
                })
                all_chunks.append(chunk)
        
        print(f"Created {len(all_chunks)} chunks across {len(self.granularities)} granularities")
        return all_chunks

def demo_chunking_strategies():
    """Demonstrate all chunking strategies"""
    
    # Sample text for chunking
    sample_text = """
    Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. In particular, it focuses on how to program computers to process and analyze large amounts of natural language data.

    The field of NLP has evolved significantly over the past few decades. Early approaches were rule-based, relying on hand-crafted grammars and lexicons. These systems were limited in their ability to handle the complexity and ambiguity of natural language.

    The introduction of statistical methods in the 1990s marked a significant shift in NLP. These approaches used probabilistic models trained on large corpora of text to make predictions about language. Machine learning algorithms became central to NLP tasks such as part-of-speech tagging, parsing, and machine translation.

    Deep learning has revolutionized NLP in recent years. Neural network models, particularly recurrent neural networks (RNNs) and transformers, have achieved state-of-the-art performance on many NLP tasks. The development of large language models like GPT and BERT has further advanced the field.

    Modern NLP applications include machine translation, sentiment analysis, question answering, text summarization, and chatbots. These systems are now capable of understanding context, handling ambiguity, and generating human-like text.

    Looking forward, NLP continues to evolve with advances in multimodal learning, few-shot learning, and more efficient model architectures. The field remains active with ongoing research in understanding language semantics, improving model interpretability, and addressing ethical considerations in AI systems.
    """
    
    print("=" * 80)
    print("CHUNKING STRATEGIES DEMONSTRATION")
    print("=" * 80)
    print(f"Sample text length: {len(sample_text)} characters")
    
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = len(encoding.encode(sample_text))
    print(f"Total tokens: {total_tokens}")
    
    # Initialize chunkers
    chunkers = {
        "Fixed-Size": FixedSizeChunker(chunk_size=200, overlap=20),
        "Sliding Window": SlidingWindowChunker(window_size=200, stride=150),
        "Semantic": SemanticChunker(max_chunk_size=250),
        "Hybrid": HybridChunker(target_size=200, max_deviation=50),
        "Recursive": RecursiveChunker(chunk_sizes=[300, 200, 100]),
        "Multi-Granularity": MultiGranularityChunker()
    }
    
    results = {}
    
    for strategy_name, chunker in chunkers.items():
        print(f"\n{'-'*50}")
        print(f"Testing {strategy_name} Strategy")
        print(f"{'-'*50}")
        
        try:
            chunks = chunker.chunk_text(sample_text)
            results[strategy_name] = chunks
            
            print(f"✅ Created {len(chunks)} chunks")
            print(f"📊 Token distribution: {[chunk.token_count for chunk in chunks]}")
            
            # Show first chunk as example
            if chunks:
                first_chunk = chunks[0]
                print(f"📝 First chunk preview: {first_chunk.content[:100]}...")
                print(f"🏷️ Metadata: {first_chunk.metadata}")
        
        except Exception as e:
            print(f"❌ Error with {strategy_name}: {e}")
    
    # Comparison analysis
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    for strategy_name, chunks in results.items():
        if chunks:
            token_counts = [chunk.token_count for chunk in chunks]
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            
            print(f"{strategy_name:15} | Chunks: {len(chunks):2} | Avg: {avg_tokens:5.1f} | Range: {min_tokens}-{max_tokens}")
    
    print(f"\n{'='*80}")
    print("CHUNKING STRATEGY RECOMMENDATIONS")
    print(f"{'='*80}")
    print("🎯 Fixed-Size: Fast, predictable, good for uniform processing")
    print("🎯 Sliding Window: Good overlap, handles context boundaries")
    print("🎯 Semantic: Preserves meaning, variable sizes, computationally intensive")
    print("🎯 Hybrid: Balance of semantic coherence and size control")
    print("🎯 Recursive: Hierarchical structure, good for complex documents")
    print("🎯 Multi-Granularity: Supports different retrieval scenarios")

if __name__ == "__main__":
    demo_chunking_strategies()