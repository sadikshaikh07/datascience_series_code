"""
RAG Evaluation Framework

This module demonstrates comprehensive evaluation techniques for RAG systems,
covering both retrieval and generation quality assessment.

Evaluation components covered:
1. Human-Centric Metrics - Faithfulness, Grounding, Correctness, Coverage
2. Retrieval Metrics - Precision@k, Recall@k, nDCG, MRR
3. Answer Metrics - Exact Match, F1 Score, BLEU, ROUGE
4. Framework Integration - RAGAS, TruLens, LangSmith simulation
5. Custom Evaluation Pipeline - End-to-end assessment

This framework enables systematic RAG system improvement through
comprehensive quality measurement.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import json
from collections import Counter
import math

# Handle optional dependencies gracefully
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas not available. Some features will be limited.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Using fallback similarity calculations.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not available. Semantic similarity will be limited.")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available. BLEU scores will use simplified calculation.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️ rouge-score not available. ROUGE metrics will use simplified calculation.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib/seaborn not available. Visualization features disabled.")

@dataclass
class EvaluationQuery:
    """Represents a query for evaluation"""
    query_id: str
    question: str
    ground_truth_answer: str
    relevant_doc_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResponse:
    """Represents a RAG system response"""
    query_id: str
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    confidence_score: float = 0.0
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Results of RAG evaluation"""
    query_id: str
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    human_centric_metrics: Dict[str, float]
    overall_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class RetrievalEvaluator:
    """Evaluates retrieval quality using IR metrics"""
    
    def __init__(self):
        self.metrics = [
            "precision_at_k",
            "recall_at_k", 
            "ndcg_at_k",
            "mrr",
            "map"
        ]
    
    def evaluate(self, retrieved_docs: List[Dict[str, Any]], 
                 relevant_doc_ids: List[str], k: int = 10) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        
        retrieved_ids = [doc.get('doc_id', '') for doc in retrieved_docs[:k]]
        relevant_set = set(relevant_doc_ids)
        
        metrics = {}
        
        # Precision@k
        relevant_retrieved = len([doc_id for doc_id in retrieved_ids if doc_id in relevant_set])
        metrics['precision_at_k'] = relevant_retrieved / min(k, len(retrieved_ids)) if retrieved_ids else 0.0
        
        # Recall@k  
        metrics['recall_at_k'] = relevant_retrieved / len(relevant_set) if relevant_set else 0.0
        
        # nDCG@k
        metrics['ndcg_at_k'] = self._calculate_ndcg(retrieved_ids, relevant_set, k)
        
        # MRR (Mean Reciprocal Rank)
        metrics['mrr'] = self._calculate_mrr(retrieved_ids, relevant_set)
        
        # MAP (Mean Average Precision) - simplified for single query
        metrics['map'] = self._calculate_average_precision(retrieved_ids, relevant_set)
        
        return metrics
    
    def _calculate_ndcg(self, retrieved_ids: List[str], relevant_set: Set[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not relevant_set:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0
        
        # IDCG calculation (perfect ranking)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, retrieved_ids: List[str], relevant_set: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_average_precision(self, retrieved_ids: List[str], relevant_set: Set[str]) -> float:
        """Calculate Average Precision"""
        if not relevant_set:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_set) if relevant_set else 0.0

class GenerationEvaluator:
    """Evaluates answer generation quality"""
    
    def __init__(self):
        """Initialize generation evaluator"""
        
        # Initialize NLTK if available
        self.nltk_available = False
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
                self.nltk_available = True
            except LookupError:
                try:
                    print("Downloading NLTK punkt tokenizer...")
                    nltk.download('punkt', quiet=True)
                    self.nltk_available = True
                except:
                    print("⚠️ Could not download NLTK data")
                    self.nltk_available = False
        
        # Initialize ROUGE scorer
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            except Exception as e:
                print(f"⚠️ Could not initialize ROUGE scorer: {e}")
                self.rouge_scorer = None
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer: {e}")
                self.sentence_model = None
    
    def evaluate(self, generated_answer: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate generated answer quality"""
        
        metrics = {}
        
        # Exact Match
        metrics['exact_match'] = float(generated_answer.strip().lower() == ground_truth.strip().lower())
        
        # F1 Score (token-level)
        metrics['f1_score'] = self._calculate_f1_score(generated_answer, ground_truth)
        
        # BLEU Score
        metrics['bleu_score'] = self._calculate_bleu_score(generated_answer, ground_truth)
        
        # ROUGE Scores
        rouge_scores = self._calculate_rouge_scores(generated_answer, ground_truth)
        metrics.update(rouge_scores)
        
        # Semantic Similarity
        metrics['semantic_similarity'] = self._calculate_semantic_similarity(generated_answer, ground_truth)
        
        # Length Ratio
        metrics['length_ratio'] = len(generated_answer.split()) / max(1, len(ground_truth.split()))
        
        return metrics
    
    def _calculate_f1_score(self, generated: str, ground_truth: str) -> float:
        """Calculate token-level F1 score"""
        gen_tokens = set(generated.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if not gen_tokens and not gt_tokens:
            return 1.0
        
        if not gen_tokens or not gt_tokens:
            return 0.0
        
        common_tokens = gen_tokens.intersection(gt_tokens)
        
        precision = len(common_tokens) / len(gen_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_bleu_score(self, generated: str, ground_truth: str) -> float:
        """Calculate BLEU score"""
        if not self.nltk_available:
            # Fallback to simple n-gram overlap
            return self._simple_bleu_score(generated, ground_truth)
        
        reference = [ground_truth.lower().split()]
        candidate = generated.lower().split()
        
        try:
            smoothing_function = SmoothingFunction().method1
            bleu = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
            return bleu
        except:
            return self._simple_bleu_score(generated, ground_truth)
    
    def _simple_bleu_score(self, generated: str, ground_truth: str) -> float:
        """Simple BLEU-like score using unigram overlap"""
        gen_tokens = generated.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if not gen_tokens:
            return 0.0
        
        matches = sum(1 for token in gen_tokens if token in gt_tokens)
        return matches / len(gen_tokens) if gen_tokens else 0.0
    
    def _calculate_rouge_scores(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not self.rouge_scorer:
            # Fallback to simple overlap-based scores
            return self._simple_rouge_scores(generated, ground_truth)
        
        try:
            scores = self.rouge_scorer.score(ground_truth, generated)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure
            }
        except:
            return self._simple_rouge_scores(generated, ground_truth)
    
    def _simple_rouge_scores(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """Simple ROUGE-like scores using token overlap"""
        gen_tokens = generated.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if not gen_tokens and not gt_tokens:
            return {'rouge1_f': 1.0, 'rouge2_f': 1.0, 'rougeL_f': 1.0}
        
        if not gen_tokens or not gt_tokens:
            return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}
        
        # Simple unigram overlap (ROUGE-1 approximation)
        common_tokens = set(gen_tokens).intersection(set(gt_tokens))
        rouge1_f = 2 * len(common_tokens) / (len(gen_tokens) + len(gt_tokens))
        
        return {
            'rouge1_f': rouge1_f,
            'rouge2_f': rouge1_f * 0.8,  # Approximation
            'rougeL_f': rouge1_f * 0.9   # Approximation
        }
    
    def _calculate_semantic_similarity(self, generated: str, ground_truth: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        if not self.sentence_model:
            # Fallback to token overlap similarity
            return self._token_overlap_similarity(generated, ground_truth)
        
        try:
            embeddings = self.sentence_model.encode([generated, ground_truth])
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            else:
                # Fallback cosine similarity
                dot_product = np.dot(embeddings[0], embeddings[1])
                norms = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                similarity = dot_product / norms if norms > 0 else 0
            return float(similarity)
        except:
            return self._token_overlap_similarity(generated, ground_truth)
    
    def _token_overlap_similarity(self, generated: str, ground_truth: str) -> float:
        """Fallback similarity based on token overlap"""
        gen_tokens = set(generated.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if not gen_tokens and not gt_tokens:
            return 1.0
        
        if not gen_tokens or not gt_tokens:
            return 0.0
        
        intersection = gen_tokens.intersection(gt_tokens)
        union = gen_tokens.union(gt_tokens)
        
        return len(intersection) / len(union) if union else 0.0

class HumanCentricEvaluator:
    """Evaluates human-centric quality metrics"""
    
    def __init__(self):
        """Initialize human-centric evaluator"""
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer for human-centric evaluator: {e}")
                self.sentence_model = None
    
    def evaluate(self, answer: str, retrieved_docs: List[Dict[str, Any]], 
                 query: str, ground_truth: str = None) -> Dict[str, float]:
        """Evaluate human-centric metrics"""
        
        metrics = {}
        
        # Faithfulness - how well answer is supported by retrieved documents
        metrics['faithfulness'] = self._calculate_faithfulness(answer, retrieved_docs)
        
        # Grounding - how well answer uses retrieved information
        metrics['grounding'] = self._calculate_grounding(answer, retrieved_docs)
        
        # Relevance - how relevant answer is to the query
        metrics['relevance'] = self._calculate_relevance(answer, query)
        
        # Completeness - how complete the answer is
        metrics['completeness'] = self._calculate_completeness(answer, query, ground_truth)
        
        # Clarity - how clear and well-structured the answer is
        metrics['clarity'] = self._calculate_clarity(answer)
        
        # Coherence - how coherent the answer is
        metrics['coherence'] = self._calculate_coherence(answer)
        
        return metrics
    
    def _calculate_faithfulness(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness - answer supported by documents"""
        if not retrieved_docs:
            return 0.0
        
        # Extract claims from answer (simplified)
        answer_sentences = self._split_into_sentences(answer)
        if not answer_sentences:
            return 0.0
        
        # Check how many claims are supported by documents
        doc_contents = [doc.get('content', '') for doc in retrieved_docs]
        all_doc_text = ' '.join(doc_contents)
        
        supported_count = 0
        for sentence in answer_sentences:
            if self._is_supported_by_context(sentence, all_doc_text):
                supported_count += 1
        
        return supported_count / len(answer_sentences)
    
    def _calculate_grounding(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate grounding - answer uses retrieved information"""
        if not retrieved_docs:
            return 0.0
        
        doc_contents = [doc.get('content', '') for doc in retrieved_docs]
        all_doc_text = ' '.join(doc_contents)
        
        # Calculate semantic similarity between answer and documents
        if not all_doc_text.strip():
            return 0.0
        
        if not self.sentence_model:
            # Fallback to token overlap
            return self._token_overlap_similarity(answer, all_doc_text)
        
        try:
            embeddings = self.sentence_model.encode([answer, all_doc_text])
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            else:
                # Fallback cosine similarity
                dot_product = np.dot(embeddings[0], embeddings[1])
                norms = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                similarity = dot_product / norms if norms > 0 else 0
            return float(similarity)
        except:
            return self._token_overlap_similarity(answer, all_doc_text)
    
    def _calculate_relevance(self, answer: str, query: str) -> float:
        """Calculate relevance to query"""
        if not self.sentence_model:
            # Fallback to token overlap
            return self._token_overlap_similarity(answer, query)
        
        try:
            embeddings = self.sentence_model.encode([answer, query])
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            else:
                # Fallback cosine similarity
                dot_product = np.dot(embeddings[0], embeddings[1])
                norms = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                similarity = dot_product / norms if norms > 0 else 0
            return float(similarity)
        except:
            return self._token_overlap_similarity(answer, query)
    
    def _calculate_completeness(self, answer: str, query: str, ground_truth: str = None) -> float:
        """Calculate completeness of answer"""
        if ground_truth:
            # Compare with ground truth
            return self._calculate_coverage(answer, ground_truth)
        else:
            # Heuristic based on answer length and query complexity
            query_words = len(query.split())
            answer_words = len(answer.split())
            
            # Simple heuristic: longer answers for complex queries
            expected_length = max(10, query_words * 2)
            length_score = min(1.0, answer_words / expected_length)
            
            return length_score
    
    def _calculate_clarity(self, answer: str) -> float:
        """Calculate clarity of answer"""
        if not answer.strip():
            return 0.0
        
        score = 1.0
        
        # Penalize very short answers
        if len(answer.split()) < 5:
            score -= 0.3
        
        # Reward proper sentence structure
        sentences = self._split_into_sentences(answer)
        if len(sentences) > 1:
            score += 0.1
        
        # Penalize excessive length
        if len(answer.split()) > 200:
            score -= 0.2
        
        # Check for clear structure (simple heuristic)
        if any(word in answer.lower() for word in ['first', 'second', 'finally', 'therefore']):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_coherence(self, answer: str) -> float:
        """Calculate coherence of answer"""
        sentences = self._split_into_sentences(answer)
        if len(sentences) <= 1:
            return 1.0
        
        if not self.sentence_model:
            # Simple fallback: assume coherent if sentences share words
            similarities = []
            for i in range(len(sentences) - 1):
                sim = self._token_overlap_similarity(sentences[i], sentences[i+1])
                similarities.append(sim)
            return np.mean(similarities) if similarities else 0.0
        
        # Calculate semantic similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            try:
                embeddings = self.sentence_model.encode([sentences[i], sentences[i+1]])
                if SKLEARN_AVAILABLE:
                    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                else:
                    # Fallback cosine similarity
                    dot_product = np.dot(embeddings[0], embeddings[1])
                    norms = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                    sim = dot_product / norms if norms > 0 else 0
                similarities.append(sim)
            except:
                sim = self._token_overlap_similarity(sentences[i], sentences[i+1])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_supported_by_context(self, claim: str, context: str) -> bool:
        """Check if claim is supported by context (simplified)"""
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        
        # Simple overlap-based support check
        overlap = len(claim_words.intersection(context_words))
        return overlap >= len(claim_words) * 0.5  # At least 50% overlap
    
    def _calculate_coverage(self, answer: str, ground_truth: str) -> float:
        """Calculate how well answer covers ground truth"""
        answer_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        if not gt_words:
            return 1.0
        
        covered_words = answer_words.intersection(gt_words)
        return len(covered_words) / len(gt_words)
    
    def _token_overlap_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity based on token overlap"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0

class RAGEvaluationFramework:
    """Comprehensive RAG evaluation framework"""
    
    def __init__(self):
        """Initialize RAG evaluation framework"""
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.human_centric_evaluator = HumanCentricEvaluator()
        
        self.evaluation_history = []
    
    def evaluate_single_query(self, query: EvaluationQuery, response: RAGResponse) -> EvaluationResult:
        """Evaluate a single query-response pair"""
        
        print(f"📊 Evaluating query: {query.query_id}")
        
        # Retrieval evaluation
        retrieval_metrics = self.retrieval_evaluator.evaluate(
            response.retrieved_docs,
            query.relevant_doc_ids,
            k=10
        )
        
        # Generation evaluation
        generation_metrics = self.generation_evaluator.evaluate(
            response.answer,
            query.ground_truth_answer
        )
        
        # Human-centric evaluation
        human_centric_metrics = self.human_centric_evaluator.evaluate(
            response.answer,
            response.retrieved_docs,
            query.question,
            query.ground_truth_answer
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            retrieval_metrics,
            generation_metrics, 
            human_centric_metrics
        )
        
        result = EvaluationResult(
            query_id=query.query_id,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            human_centric_metrics=human_centric_metrics,
            overall_score=overall_score
        )
        
        self.evaluation_history.append(result)
        
        return result
    
    def evaluate_batch(self, queries: List[EvaluationQuery], 
                      responses: List[RAGResponse]) -> List[EvaluationResult]:
        """Evaluate a batch of queries and responses"""
        
        print(f"🔍 Evaluating batch of {len(queries)} queries")
        
        results = []
        for query, response in zip(queries, responses):
            if query.query_id == response.query_id:
                result = self.evaluate_single_query(query, response)
                results.append(result)
            else:
                print(f"⚠️ Query ID mismatch: {query.query_id} vs {response.query_id}")
        
        return results
    
    def _calculate_overall_score(self, retrieval_metrics: Dict[str, float],
                                generation_metrics: Dict[str, float],
                                human_centric_metrics: Dict[str, float]) -> float:
        """Calculate overall evaluation score"""
        
        # Weighted combination of different metric categories
        weights = {
            'retrieval': 0.3,
            'generation': 0.4,
            'human_centric': 0.3
        }
        
        # Key metrics for each category
        retrieval_score = np.mean([
            retrieval_metrics.get('precision_at_k', 0),
            retrieval_metrics.get('recall_at_k', 0),
            retrieval_metrics.get('ndcg_at_k', 0)
        ])
        
        generation_score = np.mean([
            generation_metrics.get('f1_score', 0),
            generation_metrics.get('semantic_similarity', 0),
            generation_metrics.get('rouge1_f', 0)
        ])
        
        human_centric_score = np.mean([
            human_centric_metrics.get('faithfulness', 0),
            human_centric_metrics.get('relevance', 0),
            human_centric_metrics.get('completeness', 0)
        ])
        
        overall_score = (
            weights['retrieval'] * retrieval_score +
            weights['generation'] * generation_score + 
            weights['human_centric'] * human_centric_score
        )
        
        return overall_score
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        if not results:
            return {"error": "No evaluation results available"}
        
        report = {
            "summary": {
                "total_queries": len(results),
                "average_overall_score": np.mean([r.overall_score for r in results]),
                "score_std": np.std([r.overall_score for r in results])
            },
            "retrieval_performance": {},
            "generation_performance": {},
            "human_centric_performance": {}
        }
        
        # Aggregate retrieval metrics
        retrieval_metrics = {}
        for metric in ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'mrr', 'map']:
            values = [r.retrieval_metrics.get(metric, 0) for r in results]
            retrieval_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        report["retrieval_performance"] = retrieval_metrics
        
        # Aggregate generation metrics
        generation_metrics = {}
        for metric in ['f1_score', 'bleu_score', 'rouge1_f', 'semantic_similarity']:
            values = [r.generation_metrics.get(metric, 0) for r in results]
            generation_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        report["generation_performance"] = generation_metrics
        
        # Aggregate human-centric metrics
        human_metrics = {}
        for metric in ['faithfulness', 'relevance', 'completeness', 'clarity']:
            values = [r.human_centric_metrics.get(metric, 0) for r in results]
            human_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        report["human_centric_performance"] = human_metrics
        
        return report

def create_test_evaluation_data() -> Tuple[List[EvaluationQuery], List[RAGResponse]]:
    """Create test data for evaluation demonstration"""
    
    queries = [
        EvaluationQuery(
            query_id="q1",
            question="What is machine learning?",
            ground_truth_answer="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            relevant_doc_ids=["doc1", "doc2"]
        ),
        EvaluationQuery(
            query_id="q2", 
            question="How do neural networks work?",
            ground_truth_answer="Neural networks work by processing data through layers of interconnected nodes that adjust their connections based on training data to learn patterns.",
            relevant_doc_ids=["doc2", "doc3"]
        ),
        EvaluationQuery(
            query_id="q3",
            question="What is the difference between supervised and unsupervised learning?",
            ground_truth_answer="Supervised learning uses labeled training data to learn patterns, while unsupervised learning finds patterns in data without labels.",
            relevant_doc_ids=["doc1", "doc4"]
        )
    ]
    
    responses = [
        RAGResponse(
            query_id="q1",
            answer="Machine learning is a branch of AI that allows computers to learn from data automatically without explicit programming.",
            retrieved_docs=[
                {"doc_id": "doc1", "content": "Machine learning enables computers to learn from data", "score": 0.9},
                {"doc_id": "doc2", "content": "AI subset that improves through experience", "score": 0.8},
                {"doc_id": "doc5", "content": "Programming paradigm for software development", "score": 0.3}
            ],
            confidence_score=0.85
        ),
        RAGResponse(
            query_id="q2",
            answer="Neural networks process information through layers of nodes that are connected and learn by adjusting these connections.",
            retrieved_docs=[
                {"doc_id": "doc3", "content": "Neural networks have layers of interconnected nodes", "score": 0.9},
                {"doc_id": "doc2", "content": "Networks learn by adjusting connections", "score": 0.7},
                {"doc_id": "doc1", "content": "Machine learning fundamentals", "score": 0.4}
            ],
            confidence_score=0.78
        ),
        RAGResponse(
            query_id="q3",
            answer="Supervised learning uses labeled data while unsupervised learning works with unlabeled data to find hidden patterns.",
            retrieved_docs=[
                {"doc_id": "doc4", "content": "Supervised learning requires labeled training data", "score": 0.9},
                {"doc_id": "doc1", "content": "Unsupervised learning finds patterns without labels", "score": 0.8},
                {"doc_id": "doc6", "content": "Deep learning architectures", "score": 0.2}
            ],
            confidence_score=0.92
        )
    ]
    
    return queries, responses

def demo_rag_evaluation():
    """Demonstrate comprehensive RAG evaluation"""
    
    print("=" * 80)
    print("RAG EVALUATION FRAMEWORK DEMONSTRATION") 
    print("=" * 80)
    
    # Initialize evaluation framework
    evaluator = RAGEvaluationFramework()
    
    # Create test data
    queries, responses = create_test_evaluation_data()
    
    print(f"📋 Created {len(queries)} test queries and responses")
    
    # Evaluate individual queries
    print(f"\n{'='*60}")
    print("INDIVIDUAL QUERY EVALUATION")
    print(f"{'='*60}")
    
    individual_results = []
    for query, response in zip(queries, responses):
        print(f"\n{'-'*40}")
        print(f"Query: {query.question}")
        print(f"Ground Truth: {query.ground_truth_answer[:80]}...")
        print(f"Generated Answer: {response.answer[:80]}...")
        
        result = evaluator.evaluate_single_query(query, response)
        individual_results.append(result)
        
        # Display key metrics
        print(f"📊 Overall Score: {result.overall_score:.3f}")
        print(f"🔍 Retrieval - Precision@k: {result.retrieval_metrics['precision_at_k']:.3f}")
        print(f"📝 Generation - F1: {result.generation_metrics['f1_score']:.3f}")
        print(f"🧠 Human-Centric - Faithfulness: {result.human_centric_metrics['faithfulness']:.3f}")
    
    # Batch evaluation  
    print(f"\n{'='*60}")
    print("BATCH EVALUATION")
    print(f"{'='*60}")
    
    batch_results = evaluator.evaluate_batch(queries, responses)
    
    # Generate comprehensive report
    report = evaluator.generate_report(batch_results)
    
    print(f"\n📊 EVALUATION REPORT SUMMARY")
    print(f"{'-'*40}")
    print(f"Total Queries: {report['summary']['total_queries']}")
    print(f"Average Overall Score: {report['summary']['average_overall_score']:.3f}")
    print(f"Score Standard Deviation: {report['summary']['score_std']:.3f}")
    
    # Display detailed metrics
    print(f"\n🔍 RETRIEVAL PERFORMANCE")
    print(f"{'-'*30}")
    for metric, stats in report['retrieval_performance'].items():
        print(f"{metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    print(f"\n📝 GENERATION PERFORMANCE")
    print(f"{'-'*30}")
    for metric, stats in report['generation_performance'].items():
        print(f"{metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    print(f"\n🧠 HUMAN-CENTRIC PERFORMANCE")
    print(f"{'-'*30}")
    for metric, stats in report['human_centric_performance'].items():
        print(f"{metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Framework comparison simulation
    print(f"\n{'='*60}")
    print("EVALUATION FRAMEWORK COMPARISON")
    print(f"{'='*60}")
    
    frameworks = {
        "RAGAS": {"faithfulness": 0.85, "answer_relevancy": 0.78, "context_relevancy": 0.82},
        "TruLens": {"groundedness": 0.79, "query_answer_relevance": 0.83, "context_relevance": 0.77},
        "Custom": {"overall_score": report['summary']['average_overall_score']}
    }
    
    for framework, metrics in frameworks.items():
        print(f"\n{framework}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.3f}")
    
    print(f"\n{'='*80}")
    print("RAG EVALUATION RECOMMENDATIONS")
    print(f"{'='*80}")
    print("🎯 Retrieval Metrics: Focus on Precision@k and nDCG for ranking quality")
    print("🎯 Generation Metrics: F1 and semantic similarity for answer quality")
    print("🎯 Human-Centric: Faithfulness and relevance for user satisfaction")
    print("🎯 Framework Integration: Use RAGAS/TruLens for production monitoring")
    print("🎯 Continuous Evaluation: Implement automated evaluation pipelines")
    print("\n💡 Recommended evaluation frequency: Daily for production, continuous for development")

if __name__ == "__main__":
    demo_rag_evaluation()