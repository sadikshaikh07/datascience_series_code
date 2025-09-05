import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def visualize_embeddings():
    """
    Visualizes embeddings in 2D space to show how RAG retrieval works.
    This helps understand why certain documents are retrieved for queries.
    """
    
    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Documents and query
    docs = [
        "Our company provides 26 weeks of maternity leave.",
        "Employees are eligible for 2 weeks of paternity leave.",
        "The office remains closed on all national holidays."
    ]
    
    query = "What is the maternity leave policy?"
    
    # Create embeddings
    embeddings = model.encode(docs)
    query_emb = model.encode([query])
    
    # Combine for visualization
    all_embeddings = np.vstack([embeddings, query_emb])
    labels = [f"Doc {i}: {doc[:30]}..." for i, doc in enumerate(docs)] + ["Query"]
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot documents
    doc_points = reduced[:-1]
    query_point = reduced[-1]
    
    plt.scatter(doc_points[:, 0], doc_points[:, 1], 
                c=['blue', 'green', 'orange'], s=100, alpha=0.7, label='Documents')
    plt.scatter(query_point[0], query_point[1], 
                c='red', s=150, marker='*', label='Query', edgecolors='black', linewidth=2)
    
    # Add labels
    for i, (x, y) in enumerate(reduced):
        if i < len(docs):
            plt.annotate(f"Doc {i}", (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
        else:
            plt.annotate("Query", (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=12, fontweight='bold', color='red')
    
    # Draw lines from query to each document
    for i, (x, y) in enumerate(doc_points):
        plt.plot([query_point[0], x], [query_point[1], y], 
                'gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.title("RAG Retrieval: Query and Documents in 2D Embedding Space", fontsize=14, fontweight='bold')
    plt.xlabel(f"First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with explanation
    textstr = '\n'.join([
        "Closer points = More similar meaning",
        "Doc 0 (maternity) is closest to query",
        "Doc 2 (holidays) is farthest from query"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    # Print distance information
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(query_emb, embeddings)[0]
    
    print("="*50)
    print("EMBEDDING SIMILARITY ANALYSIS")
    print("="*50)
    print(f"Query: {query}")
    print()
    
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Doc {i} (Similarity: {score:.3f}): {doc}")
    
    print(f"\nMost relevant document: Doc {np.argmax(scores)}")
    print(f"Least relevant document: Doc {np.argmin(scores)}")
    
    return reduced, labels, scores

def compare_different_queries():
    """
    Shows how different queries retrieve different documents
    """
    print("\n" + "="*60)
    print("COMPARING DIFFERENT QUERIES")
    print("="*60)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    docs = [
        "Our company provides 26 weeks of maternity leave.",
        "Employees are eligible for 2 weeks of paternity leave.",
        "The office remains closed on all national holidays.",
        "Health insurance covers all employees and dependents.",
        "Remote work is available on Fridays."
    ]
    
    queries = [
        "What is the maternity leave policy?",
        "When is the office closed?", 
        "What are the remote work options?",
        "What healthcare benefits do we have?"
    ]
    
    embeddings = model.encode(docs)
    
    for query in queries:
        query_emb = model.encode([query])
        scores = cosine_similarity(query_emb.reshape(1, -1), embeddings)[0]
        best_doc_idx = np.argmax(scores)
        
        print(f"\nQuery: {query}")
        print(f"Best match: Doc {best_doc_idx} (Score: {scores[best_doc_idx]:.3f})")
        print(f"Content: {docs[best_doc_idx]}")

if __name__ == "__main__":
    
    print("RAG EMBEDDING VISUALIZATION DEMO")
    print("This shows how embeddings work in RAG retrieval")
    
    # Main visualization
    reduced, labels, scores = visualize_embeddings()
    
    # Compare different queries
    compare_different_queries()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("• Documents with similar meaning cluster together in embedding space")
    print("• RAG retrieval finds documents closest to the query vector")
    print("• Different queries will retrieve different relevant documents")
    print("• This visualization shows embeddings in 2D, but real embeddings have 384+ dimensions")