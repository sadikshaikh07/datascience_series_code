"""
Search Variants for RAG Systems

This module demonstrates different search approaches for retrieving
relevant documents in RAG systems.

Search variants covered:
1. Keyword Search (BM25, TF-IDF) - Traditional IR methods
2. Semantic/Vector Search - Dense embeddings with cosine similarity
3. Hybrid Search - Combining keyword and semantic approaches

Each approach has different strengths:
- Keyword: Good for exact term matches, fast, interpretable
- Semantic: Captures meaning and context, handles synonyms
- Hybrid: Best of both worlds, highest accuracy
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
from collections import Counter, defaultdict
import math
from abc import ABC, abstractmethod

# Vector search dependencies
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Keyword search dependencies  
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

@dataclass
class SearchResult:
    """Represents a search result with score and metadata"""
    doc_id: str
    content: str
    score: float
    search_method: str
    metadata: Dict[str, Any] = None

@dataclass
class Document:
    """Represents a document in the search index"""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = None

class SearchEngine(ABC):
    """Abstract base class for search engines"""
    
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for search"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for relevant documents"""
        pass

class KeywordSearchEngine(SearchEngine):
    """Keyword-based search using BM25 and TF-IDF"""
    
    def __init__(self, algorithm: str = "bm25"):
        """
        Initialize keyword search engine
        
        Args:
            algorithm: "bm25" or "tfidf"
        """
        self.algorithm = algorithm
        self.documents = []
        self.doc_lookup = {}
        
        if algorithm == "bm25":
            self.searcher = None  # Will be initialized after indexing
        elif algorithm == "tfidf":
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2),  # Unigrams and bigrams
                max_features=10000
            )
            self.doc_vectors = None
        else:
            raise ValueError("Algorithm must be 'bm25' or 'tfidf'")
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for keyword search"""
        print(f"🔍 Indexing {len(documents)} documents for {self.algorithm.upper()} search")
        
        self.documents = documents
        self.doc_lookup = {doc.doc_id: doc for doc in documents}
        
        # Preprocess all documents
        processed_docs = [self._preprocess_text(doc.content) for doc in documents]
        
        if self.algorithm == "bm25":
            # Tokenize for BM25
            tokenized_docs = [doc.split() for doc in processed_docs]
            self.searcher = BM25Okapi(tokenized_docs)
            
        elif self.algorithm == "tfidf":
            # Fit TF-IDF vectorizer
            self.doc_vectors = self.vectorizer.fit_transform(processed_docs)
        
        print(f"✅ Indexing complete")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for documents using keyword matching"""
        if not self.documents:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        query_processed = self._preprocess_text(query)
        
        if self.algorithm == "bm25":
            query_tokens = query_processed.split()
            scores = self.searcher.get_scores(query_tokens)
            
        elif self.algorithm == "tfidf":
            query_vector = self.vectorizer.transform([query_processed])
            scores = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc = self.documents[idx]
                result = SearchResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(scores[idx]),
                    search_method=f"keyword_{self.algorithm}",
                    metadata={
                        "algorithm": self.algorithm,
                        "query_terms": query_processed.split(),
                        "doc_index": idx
                    }
                )
                results.append(result)
        
        return results

class SemanticSearchEngine(SearchEngine):
    """Semantic search using dense vector embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_type: str = "faiss"):
        """
        Initialize semantic search engine
        
        Args:
            model_name: Sentence transformer model name
            index_type: "faiss" or "simple" (in-memory numpy)
        """
        self.model_name = model_name
        self.index_type = index_type
        
        print(f"🧠 Loading semantic model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        self.documents = []
        self.doc_embeddings = None
        self.faiss_index = None
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents by computing embeddings"""
        print(f"🔍 Computing embeddings for {len(documents)} documents")
        
        self.documents = documents
        doc_texts = [doc.content for doc in documents]
        
        # Compute embeddings
        self.doc_embeddings = self.encoder.encode(
            doc_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        if self.index_type == "faiss":
            # Build FAISS index for fast similarity search
            embedding_dim = self.doc_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine sim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.doc_embeddings)
            self.faiss_index.add(self.doc_embeddings.astype('float32'))
            
            print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
        else:
            print(f"✅ Simple index ready with {len(self.doc_embeddings)} embeddings")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for semantically similar documents"""
        if self.doc_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        if self.index_type == "faiss":
            # FAISS search
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            # Convert to lists
            scores = scores[0].tolist()
            indices = indices[0].tolist()
            
        else:
            # Simple cosine similarity search
            similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            scores = similarities[top_indices].tolist()
            indices = top_indices.tolist()
        
        # Create results
        results = []
        for score, idx in zip(scores, indices):
            if idx < len(self.documents) and score > 0:
                doc = self.documents[idx]
                result = SearchResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(score),
                    search_method="semantic_vector",
                    metadata={
                        "model": self.model_name,
                        "index_type": self.index_type,
                        "embedding_dim": self.doc_embeddings.shape[1],
                        "doc_index": idx
                    }
                )
                results.append(result)
        
        return results

class HybridSearchEngine(SearchEngine):
    """Hybrid search combining keyword and semantic approaches"""
    
    def __init__(self, 
                 keyword_weight: float = 0.4,
                 semantic_weight: float = 0.6,
                 keyword_algorithm: str = "bm25",
                 semantic_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize hybrid search engine
        
        Args:
            keyword_weight: Weight for keyword search scores
            semantic_weight: Weight for semantic search scores  
            keyword_algorithm: Algorithm for keyword search
            semantic_model: Model for semantic search
        """
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        
        # Ensure weights sum to 1
        total_weight = keyword_weight + semantic_weight
        self.keyword_weight /= total_weight
        self.semantic_weight /= total_weight
        
        print(f"🔄 Initializing hybrid search (keyword: {self.keyword_weight:.2f}, semantic: {self.semantic_weight:.2f})")
        
        # Initialize component search engines
        self.keyword_engine = KeywordSearchEngine(keyword_algorithm)
        self.semantic_engine = SemanticSearchEngine(semantic_model)
        
        self.documents = []
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents in both keyword and semantic engines"""
        print(f"🔍 Indexing {len(documents)} documents for hybrid search")
        
        self.documents = documents
        
        # Index in both engines
        self.keyword_engine.index_documents(documents)
        self.semantic_engine.index_documents(documents)
        
        print("✅ Hybrid indexing complete")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform hybrid search combining keyword and semantic results"""
        if not self.documents:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        # Get results from both engines
        # Use larger top_k for component searches to ensure good coverage
        component_k = min(top_k * 2, len(self.documents))
        
        keyword_results = self.keyword_engine.search(query, component_k)
        semantic_results = self.semantic_engine.search(query, component_k)
        
        # Normalize scores to [0, 1] range
        keyword_scores = self._normalize_scores([r.score for r in keyword_results])
        semantic_scores = self._normalize_scores([r.score for r in semantic_results])
        
        # Create score lookup by document ID
        keyword_score_map = {r.doc_id: score for r, score in zip(keyword_results, keyword_scores)}
        semantic_score_map = {r.doc_id: score for r, score in zip(semantic_results, semantic_scores)}
        
        # Combine scores for all unique documents
        all_doc_ids = set(keyword_score_map.keys()) | set(semantic_score_map.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            keyword_score = keyword_score_map.get(doc_id, 0.0)
            semantic_score = semantic_score_map.get(doc_id, 0.0)
            
            combined_score = (
                self.keyword_weight * keyword_score +
                self.semantic_weight * semantic_score
            )
            combined_scores[doc_id] = combined_score
        
        # Sort by combined score and take top-k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create hybrid results
        doc_lookup = {doc.doc_id: doc for doc in self.documents}
        results = []
        
        for doc_id, combined_score in sorted_docs:
            if combined_score > 0:
                doc = doc_lookup[doc_id]
                result = SearchResult(
                    doc_id=doc_id,
                    content=doc.content,
                    score=combined_score,
                    search_method="hybrid",
                    metadata={
                        "keyword_score": keyword_score_map.get(doc_id, 0.0),
                        "semantic_score": semantic_score_map.get(doc_id, 0.0),
                        "keyword_weight": self.keyword_weight,
                        "semantic_weight": self.semantic_weight,
                        "combined_score": combined_score
                    }
                )
                results.append(result)
        
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max normalization"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)  # All scores are equal
        
        return [(score - min_score) / (max_score - min_score) for score in scores]

def create_sample_documents() -> List[Document]:
    """Create sample documents for testing search engines"""
    
    sample_docs = [
        {
            "doc_id": "nlp_intro",
            "content": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language.",
            "metadata": {"category": "AI", "difficulty": "beginner"}
        },
        {
            "doc_id": "machine_learning",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical techniques to give computers the ability to learn from data.",
            "metadata": {"category": "AI", "difficulty": "intermediate"}
        },
        {
            "doc_id": "deep_learning",
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision and natural language processing.",
            "metadata": {"category": "AI", "difficulty": "advanced"}
        },
        {
            "doc_id": "transformers",
            "content": "Transformers are a type of neural network architecture that has become dominant in NLP tasks. They use self-attention mechanisms to process sequences of data and have enabled breakthroughs in language models like GPT and BERT.",
            "metadata": {"category": "AI", "difficulty": "advanced"}
        },
        {
            "doc_id": "python_programming",
            "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and artificial intelligence applications due to its extensive libraries and frameworks.",
            "metadata": {"category": "Programming", "difficulty": "beginner"}
        },
        {
            "doc_id": "data_structures",
            "content": "Data structures are ways of organizing and storing data in computers so that it can be accessed and modified efficiently. Common data structures include arrays, linked lists, stacks, queues, trees, and graphs.",
            "metadata": {"category": "Programming", "difficulty": "intermediate"}
        },
        {
            "doc_id": "algorithms",
            "content": "Algorithms are step-by-step procedures for solving problems or performing computations. They are fundamental to computer science and include sorting algorithms, searching algorithms, and graph algorithms.",
            "metadata": {"category": "Programming", "difficulty": "intermediate"}
        },
        {
            "doc_id": "databases",
            "content": "Databases are organized collections of structured information stored electronically in computer systems. They are managed by database management systems (DBMS) and use SQL for querying and manipulation.",
            "metadata": {"category": "Data", "difficulty": "intermediate"}
        },
        {
            "doc_id": "web_development",
            "content": "Web development involves creating websites and web applications. It includes front-end development (user interface) and back-end development (server-side logic). Popular technologies include HTML, CSS, JavaScript, and various frameworks.",
            "metadata": {"category": "Programming", "difficulty": "beginner"}
        },
        {
            "doc_id": "cybersecurity",
            "content": "Cybersecurity involves protecting computer systems, networks, and data from digital attacks. It includes practices like encryption, access control, network security, and incident response to prevent unauthorized access and data breaches.",
            "metadata": {"category": "Security", "difficulty": "advanced"}
        }
    ]
    
    return [Document(doc_id=doc["doc_id"], content=doc["content"], metadata=doc["metadata"]) 
            for doc in sample_docs]

def demo_search_variants():
    """Demonstrate different search variants"""
    
    print("=" * 80)
    print("SEARCH VARIANTS DEMONSTRATION")
    print("=" * 80)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"📚 Created {len(documents)} sample documents")
    
    # Test queries
    test_queries = [
        "What is artificial intelligence and machine learning?",
        "How do neural networks work?",
        "Python programming for beginners",
        "Database management and SQL queries"
    ]
    
    # Initialize search engines
    search_engines = {
        "BM25": KeywordSearchEngine("bm25"),
        "TF-IDF": KeywordSearchEngine("tfidf"), 
        "Semantic": SemanticSearchEngine("all-MiniLM-L6-v2", "simple"),
        "Hybrid": HybridSearchEngine(keyword_weight=0.4, semantic_weight=0.6)
    }
    
    # Index documents in all engines
    for name, engine in search_engines.items():
        print(f"\n{'-'*30} Indexing {name} {'-'*30}")
        engine.index_documents(documents)
    
    # Test each query with each search engine
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        for engine_name, engine in search_engines.items():
            print(f"\n{'-'*20} {engine_name} Results {'-'*20}")
            
            try:
                results = engine.search(query, top_k=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"{i}. [{result.doc_id}] Score: {result.score:.3f}")
                        print(f"   {result.content[:100]}...")
                        
                        if result.search_method == "hybrid":
                            kw_score = result.metadata.get("keyword_score", 0)
                            sem_score = result.metadata.get("semantic_score", 0)
                            print(f"   📊 Keyword: {kw_score:.3f}, Semantic: {sem_score:.3f}")
                        
                        print()
                else:
                    print("   No results found")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Performance comparison
    print(f"\n{'='*80}")
    print("SEARCH METHOD COMPARISON")
    print(f"{'='*80}")
    
    comparison_metrics = {}
    test_query = test_queries[0]
    
    for engine_name, engine in search_engines.items():
        try:
            results = engine.search(test_query, top_k=5)
            scores = [r.score for r in results]
            
            comparison_metrics[engine_name] = {
                "num_results": len(results),
                "avg_score": np.mean(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "score_range": max(scores) - min(scores) if len(scores) > 1 else 0
            }
        except:
            comparison_metrics[engine_name] = {"error": True}
    
    # Print comparison table
    print(f"{'Method':<12} {'Results':<8} {'Avg Score':<10} {'Max Score':<10} {'Range':<8}")
    print("-" * 60)
    
    for method, metrics in comparison_metrics.items():
        if "error" not in metrics:
            print(f"{method:<12} {metrics['num_results']:<8} {metrics['avg_score']:<10.3f} "
                  f"{metrics['max_score']:<10.3f} {metrics['score_range']:<8.3f}")
        else:
            print(f"{method:<12} {'ERROR':<8}")
    
    print(f"\n{'='*80}")
    print("SEARCH VARIANT RECOMMENDATIONS")
    print(f"{'='*80}")
    print("🎯 BM25: Best for exact keyword matching, fast, good baseline")
    print("🎯 TF-IDF: Good for term frequency analysis, interpretable scores")
    print("🎯 Semantic: Captures meaning and context, handles synonyms well")
    print("🎯 Hybrid: Combines strengths of both approaches, highest accuracy")
    print("\n💡 For production systems, start with hybrid search and tune weights based on your domain")

if __name__ == "__main__":
    demo_search_variants()