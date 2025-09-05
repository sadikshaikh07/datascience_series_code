from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from typing import List, Tuple

class RAGConceptsDemo:
    """
    Demonstrates core RAG concepts with practical examples:
    - Corpus, Documents, Chunks
    - Embeddings and Vectors  
    - Different similarity metrics
    - Top-k retrieval
    """
    
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def demonstrate_chunking(self):
        """
        Shows how documents are split into chunks for RAG
        """
        print("="*60)
        print("CHUNKING DEMONSTRATION")
        print("="*60)
        
        # A longer document that needs chunking
        full_document = """
        Our company employee handbook covers several important policies. 
        First, regarding leave policies: Our company provides 26 weeks of maternity leave 
        with full pay and benefits. Employees are eligible for 2 weeks of paternity leave. 
        Sick leave accrues at 8 hours per month for full-time employees.
        
        Second, regarding office operations: The office remains closed on all national holidays 
        including New Year's Day, Independence Day, Labor Day, and Christmas. 
        Office hours are Monday through Friday, 9 AM to 6 PM.
        
        Third, regarding benefits: Health insurance covers all employees and dependents. 
        Dental and vision coverage are also provided. The company contributes to a 401k 
        retirement plan with up to 4% matching.
        """
        
        # Simulate chunking (in real RAG, this would be more sophisticated)
        chunks = [
            "Our company provides 26 weeks of maternity leave with full pay and benefits. Employees are eligible for 2 weeks of paternity leave. Sick leave accrues at 8 hours per month.",
            
            "The office remains closed on all national holidays including New Year's Day, Independence Day, Labor Day, and Christmas. Office hours are Monday through Friday, 9 AM to 6 PM.",
            
            "Health insurance covers all employees and dependents. Dental and vision coverage are also provided. The company contributes to a 401k retirement plan with up to 4% matching."
        ]
        
        print("Original Document Length:", len(full_document), "characters")
        print(f"Split into {len(chunks)} chunks:")
        print()
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} ({len(chunk)} chars):")
            print(f"  {chunk}")
            print()
        
        return chunks
    
    def demonstrate_embeddings_and_vectors(self, chunks: List[str]):
        """
        Shows what embeddings look like and their properties
        """
        print("="*60)  
        print("EMBEDDINGS AND VECTORS")
        print("="*60)
        
        # Create embeddings
        embeddings = self.model.encode(chunks)
        
        print(f"Embedding model: {self.model.get_sentence_embedding_dimension()} dimensions")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Embeddings shape: {embeddings.shape}")
        print()
        
        # Show first few dimensions of first embedding
        print("First embedding (first 10 dimensions):")
        print(embeddings[0][:10])
        print()
        
        # Show embedding properties
        print("Embedding Properties:")
        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            print(f"  Chunk {i+1}: L2 norm = {norm:.3f}, Min = {emb.min():.3f}, Max = {emb.max():.3f}")
        
        return embeddings
    
    def compare_similarity_metrics(self, chunks: List[str], embeddings: np.ndarray):
        """
        Compares different similarity metrics used in RAG
        """
        print("\n" + "="*60)
        print("SIMILARITY METRICS COMPARISON")
        print("="*60)
        
        query = "What is the maternity leave policy?"
        query_emb = self.model.encode([query])
        
        # Calculate different similarity metrics
        cosine_scores = cosine_similarity(query_emb, embeddings)[0]
        dot_scores = np.dot(query_emb, embeddings.T)[0]
        euclidean_dists = euclidean_distances(query_emb, embeddings)[0]
        euclidean_similarities = 1 / (1 + euclidean_dists)  # Convert distance to similarity
        
        print(f"Query: {query}")
        print()
        print("Similarity Scores by Metric:")
        print("-" * 80)
        print(f"{'Chunk':<8} {'Cosine':<8} {'Dot Prod':<10} {'Euclidean':<10} {'Content':<50}")
        print("-" * 80)
        
        for i, chunk in enumerate(chunks):
            print(f"{i+1:<8} {cosine_scores[i]:<8.3f} {dot_scores[i]:<10.3f} {euclidean_similarities[i]:<10.3f} {chunk[:47]+'...' if len(chunk) > 47 else chunk}")
        
        # Show rankings
        print("\nRankings by each metric:")
        cosine_rank = np.argsort(cosine_scores)[::-1] + 1
        dot_rank = np.argsort(dot_scores)[::-1] + 1  
        euclidean_rank = np.argsort(euclidean_similarities)[::-1] + 1
        
        for i in range(len(chunks)):
            print(f"Chunk {i+1}: Cosine=#{cosine_rank[i]}, Dot=#{dot_rank[i]}, Euclidean=#{euclidean_rank[i]}")
    
    def demonstrate_top_k_retrieval(self, chunks: List[str]):
        """
        Shows how top-k retrieval works with different k values
        """
        print("\n" + "="*60)
        print("TOP-K RETRIEVAL")
        print("="*60)
        
        embeddings = self.model.encode(chunks)
        query = "What employee benefits are available?"
        query_emb = self.model.encode([query])
        
        scores = cosine_similarity(query_emb, embeddings)[0]
        
        print(f"Query: {query}")
        print()
        
        # Show different k values
        for k in [1, 2, 3]:
            print(f"Top-{k} Retrieval:")
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            for rank, idx in enumerate(top_k_indices, 1):
                print(f"  #{rank} (Score: {scores[idx]:.3f}): {chunks[idx]}")
            print()
        
        # Explain the trade-off
        print("Key Trade-offs:")
        print("• k=1: Most focused, but might miss relevant context")
        print("• k=2-3: Good balance of relevance and context") 
        print("• k>3: More context, but might include irrelevant information")
    
    def full_rag_pipeline_demo(self):
        """
        Demonstrates the complete RAG pipeline from query to retrieval
        """
        print("\n" + "="*70)
        print("COMPLETE RAG PIPELINE DEMONSTRATION")  
        print("="*70)
        
        # Step 1: Chunking
        print("STEP 1: DOCUMENT CHUNKING")
        chunks = self.demonstrate_chunking()
        
        # Step 2: Embedding Creation  
        print("\nSTEP 2: EMBEDDING CREATION")
        embeddings = self.demonstrate_embeddings_and_vectors(chunks)
        
        # Step 3: Query Processing
        query = "What leave policies does the company have?"
        print(f"\nSTEP 3: QUERY PROCESSING")
        print(f"Query: {query}")
        query_emb = self.model.encode([query])
        print(f"Query embedding shape: {query_emb.shape}")
        
        # Step 4: Similarity Search
        print(f"\nSTEP 4: SIMILARITY SEARCH")
        scores = cosine_similarity(query_emb, embeddings)[0]
        top_2_indices = np.argsort(scores)[::-1][:2]
        
        print("Retrieved documents:")
        retrieved_chunks = []
        for i, idx in enumerate(top_2_indices):
            chunk = chunks[idx]
            retrieved_chunks.append(chunk)
            print(f"  #{i+1} (Score: {scores[idx]:.3f}): {chunk}")
        
        # Step 5: Generation (simulated)
        print(f"\nSTEP 5: ANSWER GENERATION (Simulated)")
        print("In a real RAG system, these chunks would be sent to an LLM with the prompt:")
        print(f"\"Answer this question: '{query}' using the following context:")
        for chunk in retrieved_chunks:
            print(f"- {chunk}")
        print("\"")
        
        print("\nExpected LLM response:")
        print("\"Based on the company policies, we offer several leave options:")
        print("- Maternity leave: 26 weeks with full pay and benefits")  
        print("- Paternity leave: 2 weeks for eligible employees")
        print("- Sick leave: 8 hours accrued per month for full-time staff\"")

def main():
    """
    Run all RAG concept demonstrations
    """
    demo = RAGConceptsDemo()
    
    print("RAG FUNDAMENTALS - CORE CONCEPTS DEMONSTRATION")
    print("This demo shows all the key concepts from the blog post in action")
    
    # Run chunking demo
    chunks = demo.demonstrate_chunking()
    
    # Run embeddings demo  
    embeddings = demo.demonstrate_embeddings_and_vectors(chunks)
    
    # Compare similarity metrics
    demo.compare_similarity_metrics(chunks, embeddings)
    
    # Show top-k retrieval
    demo.demonstrate_top_k_retrieval(chunks)
    
    # Complete pipeline
    demo.full_rag_pipeline_demo()
    
    print("\n" + "="*70)
    print("SUMMARY OF RAG CONCEPTS")
    print("="*70)
    print("✓ Corpus: Complete knowledge base (all documents)")
    print("✓ Documents: Individual files in the corpus")  
    print("✓ Chunks: Smaller pieces of documents (200-500 words)")
    print("✓ Embeddings: Vector representations capturing semantic meaning")
    print("✓ Similarity Metrics: Ways to measure closeness (cosine, dot product, euclidean)")
    print("✓ Top-k Retrieval: Getting the k most relevant chunks")
    print("✓ RAG Pipeline: Query → Retrieve → Generate")

if __name__ == "__main__":
    main()