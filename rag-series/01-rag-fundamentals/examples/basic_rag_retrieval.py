from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def basic_rag_retrieval_demo():
    """
    Demonstrates the basic RAG retrieval process using sentence transformers.
    This example shows how to:
    1. Load an embedding model
    2. Create embeddings for documents
    3. Perform similarity search for a query
    """
    
    # Step 1: Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Step 2: Corpus (3 small docs)
    docs = [
        "Our company provides 26 weeks of maternity leave.",
        "Employees are eligible for 2 weeks of paternity leave.",
        "The office remains closed on all national holidays."
    ]
    
    print(f"\nCorpus contains {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"  Doc {i}: {doc}")
    
    # Step 3: Encode docs
    print("\nCreating document embeddings...")
    embeddings = model.encode(docs)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Step 4: Encode query
    query = "What is the maternity leave policy?"
    print(f"\nQuery: {query}")
    query_emb = model.encode([query])
    
    # Step 5: Similarity search
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_idx = int(np.argmax(scores))
    
    # Display results
    print("\n" + "="*50)
    print("RETRIEVAL RESULTS")
    print("="*50)
    print(f"Query: {query}")
    print(f"Most Relevant Doc: {docs[top_idx]}")
    print(f"Cosine Scores: {np.round(scores, 3)}")
    
    # Show all rankings
    print("\nAll documents ranked by relevance:")
    ranked_indices = np.argsort(scores)[::-1]
    for i, idx in enumerate(ranked_indices):
        print(f"  Rank {i+1}: Doc {idx} (Score: {scores[idx]:.3f}) - {docs[idx]}")
    
    return docs, embeddings, query, query_emb, scores

def top_k_retrieval_demo():
    """
    Demonstrates top-k retrieval (keeping multiple relevant documents)
    In real RAG, you usually keep top_k > 1 for better context.
    """
    print("\n" + "="*60)
    print("TOP-K RETRIEVAL DEMO")
    print("="*60)
    
    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Larger corpus for better demonstration
    docs = [
        "Our company provides 26 weeks of maternity leave.",
        "Employees are eligible for 2 weeks of paternity leave.", 
        "The office remains closed on all national holidays.",
        "Maternity leave can be extended with doctor's approval.",
        "New fathers can take additional unpaid leave if needed.",
        "Holiday schedules are published at the beginning of each year.",
        "Health insurance covers prenatal care for expecting mothers.",
        "Remote work is available during the last month of pregnancy."
    ]
    
    query = "What benefits are available for new parents?"
    k = 3  # Top-3 retrieval
    
    # Encode
    embeddings = model.encode(docs)
    query_emb = model.encode([query])
    
    # Get top-k
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    print(f"Query: {query}")
    print(f"Retrieving top-{k} documents:")
    print()
    
    retrieved_docs = []
    for i, idx in enumerate(top_k_indices):
        retrieved_docs.append(docs[idx])
        print(f"Rank {i+1} (Score: {scores[idx]:.3f}): {docs[idx]}")
    
    print("\n" + "-"*40)
    print("These documents would be sent to the LLM for answer generation.")
    
    return retrieved_docs

if __name__ == "__main__":
    # Run basic retrieval demo
    docs, embeddings, query, query_emb, scores = basic_rag_retrieval_demo()
    
    # Run top-k retrieval demo  
    retrieved_docs = top_k_retrieval_demo()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("• RAG retrieval finds the most semantically similar documents")
    print("• Cosine similarity measures the angle between embedding vectors")  
    print("• Top-k retrieval provides more context than single document retrieval")
    print("• In production RAG, these retrieved docs are fed to an LLM for generation")