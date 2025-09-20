"""
Complete RAG Pipeline with LLM Integration

This module demonstrates a production-ready RAG pipeline that integrates
all components from previous modules into a cohesive system.

Pipeline components:
1. Document Processing - Chunking, embedding, and indexing
2. Query Processing - Analysis and expansion
3. Retrieval Engine - Multi-stage search with ranking
4. Safety & Filtering - Content validation and PII protection
5. Context Management - Window optimization for LLM
6. LLM Integration - Answer generation with multiple providers
7. Evaluation & Monitoring - Quality assessment and logging

This represents the complete end-to-end RAG workflow as described
in Blog 2.3: "From Retrieval to Answers - The Full RAG Pipeline".
"""

import asyncio
import time
import logging
import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import uuid
import numpy as np
from collections import Counter


# Add parent directories to Python path for shared imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import shared LLM providers
try:
    from shared.llm_providers import get_default_provider, BaseLLMProvider, LLMResponse
    from shared.llm_providers import OpenAIProvider, AnthropicProvider
    SHARED_LLM_AVAILABLE = True
except ImportError:
    SHARED_LLM_AVAILABLE = False
    print("⚠️ Shared LLM providers not available. Using fallback implementations.")

# Handle optional dependencies gracefully
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not available. Using fallback embeddings.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Using simple vector search.")

# Legacy imports for fallback (if shared providers not available)
if not SHARED_LLM_AVAILABLE:
    try:
        import openai
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        print("⚠️ OpenAI client not available.")

    try:
        from anthropic import Anthropic
        ANTHROPIC_AVAILABLE = True
    except ImportError:
        ANTHROPIC_AVAILABLE = False
        print("⚠️ Anthropic client not available.")

@dataclass
class Document:
    """Document with metadata"""
    doc_id: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    chunk_id: Optional[str] = None

@dataclass
class RAGQuery:
    """RAG query with processing metadata"""
    query_id: str
    original_query: str
    processed_query: str = ""
    query_type: str = "factual"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResponse:
    """Complete RAG response"""
    query_id: str
    answer: str
    retrieved_documents: List[Document]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

# LLM Provider wrapper for RAG pipeline compatibility
class RAGLLMProvider:
    """Wrapper for shared LLM providers to maintain RAG pipeline compatibility"""
    
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        
    async def generate_response(self, prompt: str, context: str = None, **kwargs) -> str:
        """Generate response using shared LLM provider"""
        try:
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            
            if SHARED_LLM_AVAILABLE:
                response = await self.provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    context=context
                )
                return response.content
            else:
                # Fallback simulation
                return f"[Simulated response for: {prompt[:50]}...]"
                
        except Exception as e:
            logging.error(f"LLM API error: {e}")
            return f"Error generating response: {e}"

# Legacy LLM Provider classes (fallback when shared providers not available)
if not SHARED_LLM_AVAILABLE:
    class LLMProvider(ABC):
        """Abstract base class for LLM providers"""
        
        @abstractmethod
        async def generate_response(self, prompt: str, **kwargs) -> str:
            """Generate response from LLM"""
            pass

    class FallbackOpenAIProvider(LLMProvider):
        """Fallback OpenAI GPT provider"""
        
        def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
            self.model = model
            self.client = None
            if OPENAI_AVAILABLE and api_key:
                self.client = openai.AsyncOpenAI(api_key=api_key)
            
        async def generate_response(self, prompt: str, **kwargs) -> str:
            """Generate response using OpenAI API"""
            if not self.client:
                return f"[Simulated OpenAI response for: {prompt[:50]}...]"
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 500)
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                return f"Error generating response: {e}"

    class FallbackAnthropicProvider(LLMProvider):
        """Fallback Anthropic Claude provider"""
        
        def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: str = None):
            self.model = model
            self.client = None
            if ANTHROPIC_AVAILABLE and api_key:
                self.client = Anthropic(api_key=api_key)
            
        async def generate_response(self, prompt: str, **kwargs) -> str:
            """Generate response using Anthropic API"""
            if not self.client:
                return f"[Simulated Anthropic response for: {prompt[:50]}...]"
            
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=kwargs.get("max_tokens", 500),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                logging.error(f"Anthropic API error: {e}")
                return f"Error generating response: {e}"

class DocumentProcessor:
    """Processes documents for RAG pipeline"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer: {e}")
                self.encoder = None
        
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents: chunk, embed, and prepare for indexing"""
        print(f"📄 Processing {len(documents)} documents...")
        
        processed_docs = []
        
        for doc in documents:
            # Chunk the document
            chunks = self._chunk_document(doc)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            
            if self.encoder:
                embeddings = self.encoder.encode(chunk_texts, convert_to_numpy=True)
            else:
                # Fallback: create random embeddings for demo
                print("⚠️ Using random embeddings (sentence-transformers not available)")
                embeddings = [np.random.rand(384).astype('float32') for _ in chunk_texts]
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                processed_docs.append(chunk)
        
        print(f"✅ Created {len(processed_docs)} document chunks")
        return processed_docs
    
    def _chunk_document(self, document: Document) -> List[Document]:
        """Split document into chunks"""
        content = document.content
        chunks = []
        
        # Simple word-based chunking
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            chunk = Document(
                doc_id=document.doc_id,
                content=chunk_content,
                source=document.source,
                metadata={
                    **document.metadata,
                    "chunk_index": len(chunks),
                    "parent_doc": document.doc_id
                },
                chunk_id=f"{document.doc_id}_chunk_{len(chunks)}"
            )
            chunks.append(chunk)
        
        return chunks

class VectorIndex:
    """Vector search index using FAISS"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents = []
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            self.use_faiss = True
        else:
            print("⚠️ Using simple vector search (FAISS not available)")
            self.index = None
            self.use_faiss = False
            self.embeddings_matrix = None
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the index"""
        embeddings = np.array([doc.embedding for doc in documents if doc.embedding is not None])
        
        if len(embeddings) > 0:
            if self.use_faiss:
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings.astype('float32'))
                total_docs = self.index.ntotal
            else:
                # Simple numpy-based storage
                if self.embeddings_matrix is None:
                    self.embeddings_matrix = embeddings
                else:
                    self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings])
                total_docs = len(self.embeddings_matrix)
            
            self.documents.extend(documents)
            print(f"📚 Indexed {len(embeddings)} documents. Total: {total_docs}")
        else:
            print(f"📚 No valid embeddings to index")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if (self.use_faiss and self.index.ntotal == 0) or (not self.use_faiss and self.embeddings_matrix is None):
            return []
        
        results = []
        
        if self.use_faiss:
            # FAISS search
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
        else:
            # Simple cosine similarity search
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            
            similarities = np.dot(doc_norms, query_norm)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(similarities[idx])))
        
        return results

class QueryProcessor:
    """Processes and optimizes queries"""
    
    def __init__(self):
        self.encoder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer for query processing: {e}")
                self.encoder = None
    
    def process_query(self, query: str) -> RAGQuery:
        """Process and analyze query"""
        query_id = str(uuid.uuid4())
        
        # Basic query processing
        processed_query = query.strip()
        
        # Classify query type (simplified)
        query_type = self._classify_query(query)
        
        rag_query = RAGQuery(
            query_id=query_id,
            original_query=query,
            processed_query=processed_query,
            query_type=query_type,
            metadata={
                "query_length": len(query.split()),
                "has_question_words": any(word in query.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who'])
            }
        )
        
        return rag_query
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return "definitional"
        elif any(word in query_lower for word in ['how', 'steps', 'process']):
            return "procedural"
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            return "causal"
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return "comparative"
        else:
            return "factual"

class SafetyFilter:
    """Simplified safety filter for demo"""
    
    def __init__(self):
        self.blocked_terms = ['harmful', 'dangerous', 'illegal']
    
    def filter_query(self, query: str) -> Tuple[bool, str]:
        """Filter query for safety"""
        query_lower = query.lower()
        
        for term in self.blocked_terms:
            if term in query_lower:
                return False, f"Query contains blocked term: {term}"
        
        return True, "Query passed safety check"
    
    def filter_response(self, response: str) -> Tuple[bool, str]:
        """Filter response for safety"""
        response_lower = response.lower()
        
        for term in self.blocked_terms:
            if term in response_lower:
                return False, f"Response contains blocked term: {term}"
        
        return True, "Response passed safety check"

class ContextManager:
    """Manages context window for LLM"""
    
    def __init__(self, max_context_tokens: int = 3000):
        self.max_context_tokens = max_context_tokens
    
    def prepare_context(self, query: str, documents: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        current_tokens = 0
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        query_tokens = len(query) // 4
        
        for doc in documents:
            doc_tokens = len(doc.content) // 4
            
            if current_tokens + doc_tokens + query_tokens < self.max_context_tokens:
                context_parts.append(f"Source: {doc.source}\n{doc.content}")
                current_tokens += doc_tokens
            else:
                break
        
        return "\n\n---\n\n".join(context_parts)

class CompleteRAGPipeline:
    """Complete RAG pipeline orchestrating all components"""
    
    def __init__(self, llm_provider=None):
        # Use shared LLM provider if available, otherwise fallback
        if llm_provider is None:
            if SHARED_LLM_AVAILABLE:
                shared_provider = get_default_provider()
                self.llm_provider = RAGLLMProvider(shared_provider)
            else:
                # Fallback to simulation
                self.llm_provider = FallbackOpenAIProvider()
        elif SHARED_LLM_AVAILABLE and isinstance(llm_provider, BaseLLMProvider):
            self.llm_provider = RAGLLMProvider(llm_provider)
        else:
            self.llm_provider = llm_provider
        self.document_processor = DocumentProcessor()
        self.vector_index = VectorIndex()
        self.query_processor = QueryProcessor()
        self.safety_filter = SafetyFilter()
        self.context_manager = ContextManager()
        
        # Pipeline state
        self.is_indexed = False
        self.query_history = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def index_documents(self, documents: List[Document]):
        """Index documents for retrieval"""
        print("🚀 Starting document indexing...")
        
        # Process documents
        processed_docs = self.document_processor.process_documents(documents)
        
        # Add to vector index
        self.vector_index.add_documents(processed_docs)
        
        self.is_indexed = True
        print("✅ Document indexing complete")
    
    async def query(self, query_text: str, top_k: int = 5) -> RAGResponse:
        """Execute complete RAG pipeline"""
        start_time = time.time()
        
        if not self.is_indexed:
            raise ValueError("Pipeline not ready. Call index_documents() first.")
        
        self.logger.info(f"Processing query: {query_text}")
        
        # Step 1: Process query
        query = self.query_processor.process_query(query_text)
        
        # Step 2: Safety check on query
        is_safe, safety_msg = self.safety_filter.filter_query(query.processed_query)
        if not is_safe:
            return RAGResponse(
                query_id=query.query_id,
                answer=f"Query rejected: {safety_msg}",
                retrieved_documents=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": "safety_violation"}
            )
        
        # Step 3: Retrieve relevant documents
        if self.query_processor.encoder:
            query_embedding = self.query_processor.encoder.encode([query.processed_query], convert_to_numpy=True)[0]
        else:
            # Fallback: create random embedding for demo
            print("⚠️ Using random query embedding (sentence-transformers not available)")
            query_embedding = np.random.rand(384).astype('float32')
        
        search_results = self.vector_index.search(query_embedding, top_k)
        
        retrieved_docs = [doc for doc, score in search_results]
        
        # Step 4: Prepare context
        context = self.context_manager.prepare_context(query.processed_query, retrieved_docs)
        
        # Step 5: Generate prompt
        prompt = self._create_prompt(query.processed_query, context)
        
        # Step 6: Generate answer
        raw_answer = await self.llm_provider.generate_response(
            prompt=query.processed_query, 
            context=context,
            system_prompt="You are a helpful assistant. Use the provided context to answer questions accurately."
        )
        
        # Step 7: Safety check on response
        is_safe, safety_msg = self.safety_filter.filter_response(raw_answer)
        if not is_safe:
            raw_answer = "I cannot provide an answer that meets safety guidelines."
        
        # Step 8: Calculate confidence score
        confidence_score = self._calculate_confidence(query, retrieved_docs, raw_answer)
        
        # Step 9: Create response
        processing_time = time.time() - start_time
        
        response = RAGResponse(
            query_id=query.query_id,
            answer=raw_answer,
            retrieved_documents=retrieved_docs,
            confidence_score=confidence_score,
            processing_time=processing_time,
            metadata={
                "query_type": query.query_type,
                "retrieved_count": len(retrieved_docs),
                "context_length": len(context),
                "safety_passed": is_safe
            }
        )
        
        # Log the interaction
        self.query_history.append(response)
        self.logger.info(f"Query processed in {processing_time:.2f}s with confidence {confidence_score:.2f}")
        
        return response
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM"""
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so
- Be concise but complete
- Use a professional and helpful tone

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, query: RAGQuery, docs: List[Document], answer: str) -> float:
        """Calculate confidence score for the response"""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Number of retrieved documents
        if len(docs) >= 3:
            confidence += 0.2
        elif len(docs) >= 1:
            confidence += 0.1
        
        # Factor 2: Answer length (not too short, not too long)
        answer_words = len(answer.split())
        if 10 <= answer_words <= 100:
            confidence += 0.2
        
        # Factor 3: Query-answer relevance (simplified)
        if any(word in answer.lower() for word in query.processed_query.lower().split()):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        if not self.query_history:
            return {"error": "No queries processed yet"}
        
        response_times = [r.processing_time for r in self.query_history]
        confidence_scores = [r.confidence_score for r in self.query_history]
        
        return {
            "total_queries": len(self.query_history),
            "avg_response_time": np.mean(response_times),
            "avg_confidence": np.mean(confidence_scores),
            "total_documents_indexed": self.vector_index.index.ntotal,
            "query_types": Counter([r.metadata.get("query_type", "unknown") for r in self.query_history])
        }

def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration"""
    
    sample_docs = [
        Document(
            doc_id="ml_intro",
            content="""Machine Learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.""",
            source="AI Textbook Chapter 1",
            metadata={"category": "fundamentals", "difficulty": "beginner"}
        ),
        Document(
            doc_id="neural_networks",
            content="""Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. Such systems learn to perform tasks by considering examples, generally without being programmed with task-specific rules. A neural network is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons.""",
            source="Deep Learning Fundamentals",
            metadata={"category": "algorithms", "difficulty": "intermediate"}
        ),
        Document(
            doc_id="supervised_learning",
            content="""Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.""",
            source="ML Algorithms Guide",
            metadata={"category": "learning_types", "difficulty": "intermediate"}
        ),
        Document(
            doc_id="unsupervised_learning",
            content="""Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision. In contrast to supervised learning that usually makes use of human-labeled data, unsupervised learning, also known as self-organization allows for modeling of probability densities over inputs. It forms one of the three main categories of machine learning, along with supervised and reinforcement learning.""",
            source="ML Algorithms Guide",
            metadata={"category": "learning_types", "difficulty": "intermediate"}
        ),
        Document(
            doc_id="deep_learning",
            content="""Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics and drug design.""",
            source="Advanced AI Concepts",
            metadata={"category": "advanced", "difficulty": "advanced"}
        )
    ]
    
    return sample_docs

async def demo_complete_rag_pipeline():
    """Demonstrate the complete RAG pipeline"""
    
    print("=" * 80)
    print("COMPLETE RAG PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize pipeline with shared LLM providers
    # This will automatically use the best available provider (OpenAI, Anthropic, or simulation)
    pipeline = CompleteRAGPipeline()
    
    print(f"🤖 LLM Provider: {pipeline.llm_provider.__class__.__name__}")
    if hasattr(pipeline.llm_provider, 'provider'):
        print(f"📡 Backend Provider: {pipeline.llm_provider.provider.__class__.__name__}")
        print(f"🔑 Provider Available: {pipeline.llm_provider.provider.is_available()}")
    
    # Create and index documents
    documents = create_sample_documents()
    print(f"📚 Created {len(documents)} sample documents")
    
    pipeline.index_documents(documents)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?", 
        "What is the difference between supervised and unsupervised learning?",
        "Explain deep learning and its applications",
        "What are the main types of machine learning?"
    ]
    
    print(f"\n{'='*60}")
    print("TESTING RAG PIPELINE WITH SAMPLE QUERIES")
    print(f"{'='*60}")
    
    # Process each query
    for i, query_text in enumerate(test_queries, 1):
        print(f"\n{'-'*50}")
        print(f"Query {i}: {query_text}")
        print(f"{'-'*50}")
        
        try:
            response = await pipeline.query(query_text, top_k=3)
            
            print(f"📝 Answer: {response.answer}")
            print(f"🎯 Confidence: {response.confidence_score:.2f}")
            print(f"⏱️ Processing Time: {response.processing_time:.2f}s")
            print(f"📊 Retrieved Documents: {len(response.retrieved_documents)}")
            
            # Show retrieved document sources
            if response.retrieved_documents:
                print("📚 Sources:")
                for doc in response.retrieved_documents[:2]:  # Show top 2
                    print(f"   - {doc.source}: {doc.content[:80]}...")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Show pipeline statistics
    print(f"\n{'='*60}")
    print("PIPELINE PERFORMANCE STATISTICS")
    print(f"{'='*60}")
    
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Demonstrate different LLM providers
    print(f"\n{'='*60}")
    print("MULTI-LLM PROVIDER DEMONSTRATION")
    print(f"{'='*60}")
    
    # Test different providers if shared LLM providers are available
    providers = {}
    
    if SHARED_LLM_AVAILABLE:
        try:
            from shared.llm_providers import get_openai_provider, get_anthropic_provider
            providers["OpenAI"] = get_openai_provider()
            providers["Anthropic"] = get_anthropic_provider()
        except Exception as e:
            print(f"⚠️ Could not load additional providers: {e}")
            providers["Simulation"] = get_default_provider()
    else:
        providers["Fallback"] = FallbackOpenAIProvider()
    
    test_query = "What is machine learning?"
    
    for provider_name, provider in providers.items():
        print(f"\n--- Testing {provider_name} ---")
        
        # Create new pipeline with different provider
        if SHARED_LLM_AVAILABLE:
            temp_pipeline = CompleteRAGPipeline(provider)
        else:
            temp_pipeline = CompleteRAGPipeline(provider)
        temp_pipeline.vector_index = pipeline.vector_index  # Reuse index
        temp_pipeline.is_indexed = True
        
        try:
            response = await temp_pipeline.query(test_query, top_k=2)
            print(f"✅ {provider_name} Response: {response.answer[:100]}...")
        except Exception as e:
            print(f"❌ {provider_name} Error: {e}")
    
    print(f"\n{'='*80}")
    print("RAG PIPELINE ARCHITECTURE SUMMARY")
    print(f"{'='*80}")
    print("🏗️ Document Processing: Chunking → Embedding → Indexing")
    print("🔍 Query Processing: Analysis → Safety Check → Vector Search")
    print("📝 Response Generation: Context Assembly → LLM Call → Safety Filter")
    print("📊 Quality Assurance: Confidence Scoring → Logging → Monitoring")
    print("🔄 Multi-Provider: OpenAI, Anthropic, Custom LLM Support")
    
    print(f"\n💡 Production Considerations:")
    print("  - Implement proper API key management")
    print("  - Add comprehensive error handling")
    print("  - Scale vector index for large document collections")
    print("  - Add caching for improved performance")
    print("  - Implement real-time monitoring and alerts")

if __name__ == "__main__":
    # Import Counter for statistics
    
    # Run the demo
    asyncio.run(demo_complete_rag_pipeline())