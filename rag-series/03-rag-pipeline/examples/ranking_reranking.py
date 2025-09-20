"""
Ranking and Re-ranking Pipeline for RAG Systems

This module demonstrates advanced ranking and re-ranking techniques
to improve retrieval quality in RAG systems.

Components covered:
1. First-Pass Ranking - Initial retrieval (ANN search, keyword matching)
2. Re-ranking Strategies:
   - Bi-Encoder Re-ranking - Improved semantic similarity
   - Cross-Encoder Re-ranking - Deep interaction modeling  
   - LLM-as-Re-ranker - Using language models for relevance
   - Fusion Re-ranking - Combining multiple ranking signals

The pipeline follows: Initial Retrieval → Re-ranking → Final Results
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
from abc import ABC, abstractmethod

# Transformers for re-ranking
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Statistical fusion methods
from scipy.stats import rankdata
from scipy.special import softmax

@dataclass 
class RankedResult:
    """Represents a ranked search result"""
    doc_id: str
    content: str
    initial_score: float
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RankingSignal:
    """Represents a ranking signal (score from different methods)"""
    signal_name: str
    scores: List[float]
    weight: float = 1.0
    
class Reranker(ABC):
    """Abstract base class for re-ranking strategies"""
    
    @abstractmethod
    def rerank(self, query: str, results: List[RankedResult], top_k: int = None) -> List[RankedResult]:
        """Re-rank the initial results"""
        pass

class BiEncoderReranker(Reranker):
    """Re-ranking using bi-encoder models for improved semantic matching"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize bi-encoder re-ranker
        
        Args:
            model_name: Sentence transformer model for re-ranking
        """
        self.model_name = model_name
        print(f"🧠 Loading bi-encoder model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
    
    def rerank(self, query: str, results: List[RankedResult], top_k: int = None) -> List[RankedResult]:
        """Re-rank using bi-encoder semantic similarity"""
        print(f"🔄 Bi-encoder re-ranking {len(results)} results")
        
        if not results:
            return results
        
        # Encode query and documents
        query_embedding = self.encoder.encode([query], convert_to_tensor=True)
        doc_texts = [result.content for result in results]
        doc_embeddings = self.encoder.encode(doc_texts, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = cosine_similarity(
            query_embedding.cpu().numpy(),
            doc_embeddings.cpu().numpy()
        )[0]
        
        # Update results with re-ranking scores
        reranked_results = []
        for i, result in enumerate(results):
            new_result = RankedResult(
                doc_id=result.doc_id,
                content=result.content,
                initial_score=result.initial_score,
                rerank_score=float(similarities[i]),
                metadata={
                    **result.metadata,
                    "rerank_method": "bi_encoder",
                    "model": self.model_name
                }
            )
            reranked_results.append(new_result)
        
        # Sort by re-ranking score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Apply top-k if specified
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
            result.final_score = result.rerank_score
        
        print(f"✅ Re-ranked to top {len(reranked_results)} results")
        return reranked_results

class CrossEncoderReranker(Reranker):
    """Re-ranking using cross-encoder models for deep query-document interaction"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder re-ranker
        
        Args:
            model_name: Cross-encoder model for re-ranking
        """
        self.model_name = model_name
        print(f"🔗 Loading cross-encoder model: {model_name}")
        self.cross_encoder = CrossEncoder(model_name)
    
    def rerank(self, query: str, results: List[RankedResult], top_k: int = None) -> List[RankedResult]:
        """Re-rank using cross-encoder relevance scores"""
        print(f"🔄 Cross-encoder re-ranking {len(results)} results")
        
        if not results:
            return results
        
        # Prepare query-document pairs
        query_doc_pairs = [[query, result.content] for result in results]
        
        # Get relevance scores
        relevance_scores = self.cross_encoder.predict(query_doc_pairs)
        
        # Update results with re-ranking scores
        reranked_results = []
        for i, result in enumerate(results):
            new_result = RankedResult(
                doc_id=result.doc_id,
                content=result.content,
                initial_score=result.initial_score,
                rerank_score=float(relevance_scores[i]),
                metadata={
                    **result.metadata,
                    "rerank_method": "cross_encoder",
                    "model": self.model_name
                }
            )
            reranked_results.append(new_result)
        
        # Sort by re-ranking score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Apply top-k if specified
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        # Update ranks and final scores
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
            result.final_score = result.rerank_score
        
        print(f"✅ Re-ranked to top {len(reranked_results)} results")
        return reranked_results

class LLMReranker(Reranker):
    """Re-ranking using Large Language Models"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", batch_size: int = 5):
        """
        Initialize LLM re-ranker
        
        Args:
            model_name: LLM model for re-ranking
            batch_size: Number of documents to rank in each LLM call
        """
        self.model_name = model_name
        self.batch_size = batch_size
    
    def rerank(self, query: str, results: List[RankedResult], top_k: int = None) -> List[RankedResult]:
        """Re-rank using LLM relevance assessment"""
        print(f"🤖 LLM re-ranking {len(results)} results with {self.model_name}")
        
        if not results:
            return results
        
        # Process in batches to avoid context window limits
        all_scores = []
        
        for i in range(0, len(results), self.batch_size):
            batch_results = results[i:i + self.batch_size]
            batch_scores = self._rank_batch(query, batch_results)
            all_scores.extend(batch_scores)
        
        # Update results with LLM scores
        reranked_results = []
        for i, result in enumerate(results):
            new_result = RankedResult(
                doc_id=result.doc_id,
                content=result.content,
                initial_score=result.initial_score,
                rerank_score=all_scores[i],
                metadata={
                    **result.metadata,
                    "rerank_method": "llm",
                    "model": self.model_name
                }
            )
            reranked_results.append(new_result)
        
        # Sort by LLM scores
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Apply top-k if specified
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        # Update ranks and final scores
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
            result.final_score = result.rerank_score
        
        print(f"✅ LLM re-ranked to top {len(reranked_results)} results")
        return reranked_results
    
    def _rank_batch(self, query: str, results: List[RankedResult]) -> List[float]:
        """Rank a batch of results using LLM"""
        
        # Create documents text for LLM prompt
        docs_text = ""
        for i, result in enumerate(results):
            docs_text += f"\nDocument {i+1}: {result.content[:200]}..."
        
        prompt = f"""
        Given the following query and documents, score each document's relevance to the query on a scale of 0-10.
        
        Query: {query}
        
        Documents:{docs_text}
        
        Please provide only the scores as a comma-separated list (e.g., "8.5, 6.2, 9.1, 7.0").
        Consider semantic relevance, factual accuracy, and how well each document answers the query.
        """
        
        try:
            # Simulate LLM call (replace with actual API call)
            # For demo purposes, we'll generate pseudo-realistic scores
            scores = self._simulate_llm_scoring(query, results)
            
            # In real implementation, parse LLM response:
            # response = openai.ChatCompletion.create(
            #     model=self.model_name,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.1
            # )
            # scores_text = response.choices[0].message.content
            # scores = [float(s.strip()) for s in scores_text.split(",")]
            
            return scores
            
        except Exception as e:
            print(f"⚠️ LLM ranking failed: {e}")
            # Fallback to initial scores
            return [result.initial_score for result in results]
    
    def _simulate_llm_scoring(self, query: str, results: List[RankedResult]) -> List[float]:
        """Simulate LLM scoring for demonstration"""
        # Create realistic scores based on initial scores with some variation
        scores = []
        query_words = set(query.lower().split())
        
        for result in results:
            doc_words = set(result.content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            
            # Base score on word overlap and initial score
            base_score = result.initial_score * 10  # Scale to 0-10
            overlap_bonus = overlap * 0.5
            noise = np.random.normal(0, 0.3)  # Add some noise
            
            final_score = max(0, min(10, base_score + overlap_bonus + noise))
            scores.append(final_score)
        
        return scores

class FusionReranker(Reranker):
    """Fusion re-ranking combining multiple ranking signals"""
    
    def __init__(self, fusion_method: str = "rrf", rrf_k: int = 60):
        """
        Initialize fusion re-ranker
        
        Args:
            fusion_method: "rrf" (Reciprocal Rank Fusion), "weighted_sum", or "borda_count"
            rrf_k: Parameter for RRF (typically 20-100)
        """
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.ranking_signals = []
    
    def add_ranking_signal(self, signal_name: str, scores: List[float], weight: float = 1.0):
        """Add a ranking signal to the fusion"""
        signal = RankingSignal(signal_name, scores, weight)
        self.ranking_signals.append(signal)
    
    def rerank(self, query: str, results: List[RankedResult], top_k: int = None) -> List[RankedResult]:
        """Re-rank using fusion of multiple signals"""
        print(f"🔀 Fusion re-ranking with {len(self.ranking_signals)} signals using {self.fusion_method}")
        
        if not results or not self.ranking_signals:
            return results
        
        n_docs = len(results)
        
        if self.fusion_method == "rrf":
            final_scores = self._reciprocal_rank_fusion(n_docs)
        elif self.fusion_method == "weighted_sum":
            final_scores = self._weighted_sum_fusion(n_docs)
        elif self.fusion_method == "borda_count":
            final_scores = self._borda_count_fusion(n_docs)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Update results with fusion scores
        reranked_results = []
        for i, result in enumerate(results):
            new_result = RankedResult(
                doc_id=result.doc_id,
                content=result.content,
                initial_score=result.initial_score,
                rerank_score=final_scores[i],
                final_score=final_scores[i],
                metadata={
                    **result.metadata,
                    "rerank_method": "fusion",
                    "fusion_type": self.fusion_method,
                    "num_signals": len(self.ranking_signals)
                }
            )
            reranked_results.append(new_result)
        
        # Sort by fusion score
        reranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply top-k if specified
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
        
        print(f"✅ Fusion re-ranked to top {len(reranked_results)} results")
        return reranked_results
    
    def _reciprocal_rank_fusion(self, n_docs: int) -> List[float]:
        """Reciprocal Rank Fusion (RRF)"""
        fusion_scores = np.zeros(n_docs)
        
        for signal in self.ranking_signals:
            # Convert scores to ranks (1-based)
            ranks = rankdata(-np.array(signal.scores), method='ordinal')
            
            # Apply RRF formula: 1 / (k + rank)
            rrf_scores = signal.weight / (self.rrf_k + ranks)
            fusion_scores += rrf_scores
        
        return fusion_scores.tolist()
    
    def _weighted_sum_fusion(self, n_docs: int) -> List[float]:
        """Weighted sum of normalized scores"""
        fusion_scores = np.zeros(n_docs)
        
        for signal in self.ranking_signals:
            # Normalize scores to [0, 1]
            scores = np.array(signal.scores)
            if scores.max() > scores.min():
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized_scores = scores
            
            fusion_scores += signal.weight * normalized_scores
        
        return fusion_scores.tolist()
    
    def _borda_count_fusion(self, n_docs: int) -> List[float]:
        """Borda count fusion"""
        fusion_scores = np.zeros(n_docs)
        
        for signal in self.ranking_signals:
            # Convert scores to ranks (0-based, descending)
            ranks = rankdata(-np.array(signal.scores), method='ordinal') - 1
            
            # Borda count: n_docs - rank
            borda_scores = signal.weight * (n_docs - ranks)
            fusion_scores += borda_scores
        
        return fusion_scores.tolist()

class RankingPipeline:
    """Complete ranking and re-ranking pipeline"""
    
    def __init__(self):
        self.rerankers = []
    
    def add_reranker(self, reranker: Reranker):
        """Add a re-ranker to the pipeline"""
        self.rerankers.append(reranker)
    
    def rank(self, query: str, initial_results: List[RankedResult], top_k: int = 10) -> List[RankedResult]:
        """Execute the complete ranking pipeline"""
        print(f"🚀 Starting ranking pipeline with {len(self.rerankers)} re-rankers")
        
        current_results = initial_results.copy()
        
        # Apply each re-ranker in sequence
        for i, reranker in enumerate(self.rerankers):
            print(f"\n--- Stage {i+1}: {type(reranker).__name__} ---")
            current_results = reranker.rerank(query, current_results, top_k)
        
        print(f"\n✅ Pipeline complete. Final top-{len(current_results)} results:")
        for i, result in enumerate(current_results[:5]):  # Show top 5
            print(f"  {i+1}. [{result.doc_id}] Score: {result.final_score:.3f}")
        
        return current_results

def create_sample_results() -> List[RankedResult]:
    """Create sample search results for demonstration"""
    
    sample_results = [
        RankedResult(
            doc_id="ml_basics",
            content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions.",
            initial_score=0.85,
            metadata={"category": "AI", "length": 150}
        ),
        RankedResult(
            doc_id="deep_learning", 
            content="Deep learning uses neural networks with multiple layers to model complex patterns. It has revolutionized computer vision, natural language processing, and speech recognition.",
            initial_score=0.78,
            metadata={"category": "AI", "length": 120}
        ),
        RankedResult(
            doc_id="data_science",
            content="Data science combines statistics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, analysis, and visualization.",
            initial_score=0.72,
            metadata={"category": "Data", "length": 130}
        ),
        RankedResult(
            doc_id="python_programming",
            content="Python is a versatile programming language popular in data science and machine learning. It offers extensive libraries like pandas, numpy, and scikit-learn.",
            initial_score=0.68,
            metadata={"category": "Programming", "length": 110}
        ),
        RankedResult(
            doc_id="statistics",
            content="Statistics provides the mathematical foundation for data analysis and machine learning. Key concepts include probability, hypothesis testing, and regression analysis.",
            initial_score=0.64,
            metadata={"category": "Math", "length": 140}
        ),
        RankedResult(
            doc_id="algorithms",
            content="Algorithms are step-by-step procedures for solving problems. In machine learning, common algorithms include linear regression, decision trees, and neural networks.",
            initial_score=0.60,
            metadata={"category": "CS", "length": 125}
        )
    ]
    
    return sample_results

def demo_ranking_reranking():
    """Demonstrate ranking and re-ranking techniques"""
    
    print("=" * 80)
    print("RANKING AND RE-RANKING PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Sample query and results
    query = "What is machine learning and how does it work?"
    initial_results = create_sample_results()
    
    print(f"Query: {query}")
    print(f"Initial results: {len(initial_results)}")
    
    # Show initial ranking
    print(f"\n{'='*50}")
    print("INITIAL RANKING")
    print(f"{'='*50}")
    for i, result in enumerate(initial_results, 1):
        print(f"{i}. [{result.doc_id}] Score: {result.initial_score:.3f}")
        print(f"   {result.content[:80]}...")
    
    # Test individual re-rankers
    rerankers = {
        "Bi-Encoder": BiEncoderReranker("all-MiniLM-L6-v2"),
        "Cross-Encoder": CrossEncoderReranker("cross-encoder/ms-marco-TinyBERT-L-2-v2"),
        "LLM": LLMReranker("gpt-3.5-turbo", batch_size=3)
    }
    
    reranker_results = {}
    
    for name, reranker in rerankers.items():
        print(f"\n{'='*50}")
        print(f"{name.upper()} RE-RANKING")
        print(f"{'='*50}")
        
        try:
            results = reranker.rerank(query, initial_results.copy(), top_k=6)
            reranker_results[name] = results
            
            print("Top 3 results:")
            for i, result in enumerate(results[:3], 1):
                print(f"{i}. [{result.doc_id}] Score: {result.rerank_score:.3f}")
                print(f"   Initial: {result.initial_score:.3f} → Re-rank: {result.rerank_score:.3f}")
        
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
    
    # Fusion re-ranking demonstration
    if len(reranker_results) >= 2:
        print(f"\n{'='*50}")
        print("FUSION RE-RANKING")
        print(f"{'='*50}")
        
        # Create fusion reranker
        fusion_reranker = FusionReranker("rrf", rrf_k=60)
        
        # Add signals from different rerankers
        for name, results in reranker_results.items():
            scores = [r.rerank_score for r in results]
            fusion_reranker.add_ranking_signal(name.lower(), scores, weight=1.0)
        
        # Add initial scores as a signal
        initial_scores = [r.initial_score for r in initial_results]
        fusion_reranker.add_ranking_signal("initial", initial_scores, weight=0.5)
        
        fusion_results = fusion_reranker.rerank(query, initial_results.copy(), top_k=6)
        
        print("Fusion results (top 3):")
        for i, result in enumerate(fusion_results[:3], 1):
            print(f"{i}. [{result.doc_id}] Score: {result.final_score:.3f}")
    
    # Complete pipeline demonstration
    print(f"\n{'='*50}")
    print("COMPLETE RANKING PIPELINE")
    print(f"{'='*50}")
    
    # Create pipeline with multiple stages
    pipeline = RankingPipeline()
    
    try:
        # Add re-rankers in order of computational cost
        pipeline.add_reranker(BiEncoderReranker("all-MiniLM-L6-v2"))
        pipeline.add_reranker(CrossEncoderReranker("cross-encoder/ms-marco-TinyBERT-L-2-v2"))
        
        # Execute pipeline
        final_results = pipeline.rank(query, initial_results.copy(), top_k=5)
        
        print(f"\n🎯 Final ranking comparison:")
        print(f"{'Rank':<4} {'Doc ID':<15} {'Initial':<8} {'Final':<8} {'Change':<6}")
        print("-" * 50)
        
        initial_lookup = {r.doc_id: (i+1, r.initial_score) for i, r in enumerate(initial_results)}
        
        for i, result in enumerate(final_results, 1):
            initial_rank, initial_score = initial_lookup[result.doc_id]
            rank_change = initial_rank - i
            change_symbol = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "="
            
            print(f"{i:<4} {result.doc_id:<15} {initial_score:<8.3f} {result.final_score:<8.3f} {change_symbol}{abs(rank_change):<5}")
    
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
    
    print(f"\n{'='*80}")
    print("RE-RANKING STRATEGY RECOMMENDATIONS")
    print(f"{'='*80}")
    print("🎯 Bi-Encoder: Fast, good for initial filtering, moderate accuracy improvement")
    print("🎯 Cross-Encoder: High accuracy, slower, best for final re-ranking")
    print("🎯 LLM Re-ranking: Highest quality, expensive, use for critical applications")
    print("🎯 Fusion: Combines strengths, robust, good for production systems")
    print("\n💡 Recommended pipeline: Initial Retrieval → Bi-Encoder → Cross-Encoder → LLM (top-3)")

if __name__ == "__main__":
    demo_ranking_reranking()