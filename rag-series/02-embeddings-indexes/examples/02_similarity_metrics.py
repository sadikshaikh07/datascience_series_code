#!/usr/bin/env python3
"""
Blog 2.2 - Section 2: Similarity Metrics
========================================

This script demonstrates different similarity metrics used in RAG:
- Cosine similarity
- Dot product similarity
- Euclidean (L2) distance
- When to use each metric and their properties
- Normalization effects on similarity rankings

Covers blog sections:
- 2️⃣ Similarity Metrics: How We Compare Vectors
- 📐 Cosine similarity
- ⚡ Dot product
- 📏 Euclidean (L2) distance
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns


def demonstrate_all_similarity_metrics():
    """
    Shows the three main similarity metrics with practical examples
    """
    print("=" * 60)
    print("SIMILARITY METRICS COMPARISON")
    print("=" * 60)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Test documents
    docs = [
        "Our company provides 26 weeks of maternity leave",
        "Employees get 2 weeks of paternity leave", 
        "The office is closed on national holidays",
        "Pizza recipe with homemade dough"
    ]
    
    query = "What is the maternity leave policy?"
    
    print(f"Query: '{query}'")
    print(f"\nDocuments:")
    for i, doc in enumerate(docs):
        print(f"  {i}: {doc}")
    
    # Get embeddings (both raw and normalized)
    embeddings_raw = model.encode(docs)
    query_emb_raw = model.encode([query])
    
    # Normalized versions
    embeddings_norm = embeddings_raw / np.linalg.norm(embeddings_raw, axis=1, keepdims=True)
    query_emb_norm = query_emb_raw / np.linalg.norm(query_emb_raw)
    
    print(f"\nEmbedding properties:")
    print(f"  Raw embeddings shape: {embeddings_raw.shape}")
    print(f"  Normalized embeddings shape: {embeddings_norm.shape}")
    
    # Demonstrate each metric
    demonstrate_cosine_similarity(query_emb_raw, embeddings_raw, query_emb_norm, embeddings_norm, docs)
    demonstrate_dot_product(query_emb_raw, embeddings_raw, query_emb_norm, embeddings_norm, docs)
    demonstrate_euclidean_distance(query_emb_raw, embeddings_raw, query_emb_norm, embeddings_norm, docs)
    
    return query_emb_raw, embeddings_raw, query_emb_norm, embeddings_norm, docs


def demonstrate_cosine_similarity(query_raw, docs_raw, query_norm, docs_norm, doc_texts):
    """
    Demonstrates cosine similarity calculation and properties
    """
    print(f"\n" + "=" * 50)
    print("📐 COSINE SIMILARITY")
    print("=" * 50)
    
    print("Formula: cos(θ) = (A · B) / (||A|| ||B||)")
    print("Range: [-1, 1] (1 = identical, 0 = orthogonal, -1 = opposite)")
    print()
    
    # Calculate cosine similarity
    cos_scores_raw = cosine_similarity(query_raw.reshape(1, -1), docs_raw)[0]
    cos_scores_norm = cosine_similarity(query_norm.reshape(1, -1), docs_norm)[0]
    
    print("Raw embeddings:")
    for i, (doc, score) in enumerate(zip(doc_texts, cos_scores_raw)):
        print(f"  Doc {i}: {score:.3f} - {doc[:50]}...")
    
    print("\nNormalized embeddings:")
    for i, (doc, score) in enumerate(zip(doc_texts, cos_scores_norm)):
        print(f"  Doc {i}: {score:.3f} - {doc[:50]}...")
    
    # Show ranking
    raw_ranking = np.argsort(cos_scores_raw)[::-1]
    norm_ranking = np.argsort(cos_scores_norm)[::-1]
    
    print(f"\nRankings:")
    print(f"  Raw: {raw_ranking.tolist()}")
    print(f"  Normalized: {norm_ranking.tolist()}")
    
    print(f"\nKey insights:")
    print(f"• Cosine similarity measures angle between vectors")
    print(f"• Ignores vector magnitude (length)")
    print(f"• Same ranking for raw and normalized embeddings")
    print(f"• Perfect for semantic similarity in RAG")


def demonstrate_dot_product(query_raw, docs_raw, query_norm, docs_norm, doc_texts):
    """
    Demonstrates dot product similarity and its relationship to cosine
    """
    print(f"\n" + "=" * 50)
    print("⚡ DOT PRODUCT SIMILARITY")
    print("=" * 50)
    
    print("Formula: A · B = Σ(Ai × Bi)")
    print("Range: [−∞, +∞] (higher = more similar)")
    print()
    
    # Calculate dot product
    dot_scores_raw = np.dot(query_raw, docs_raw.T)[0]
    dot_scores_norm = np.dot(query_norm.reshape(1, -1), docs_norm.T)[0]
    
    print("Raw embeddings:")
    for i, (doc, score) in enumerate(zip(doc_texts, dot_scores_raw)):
        print(f"  Doc {i}: {score:.3f} - {doc[:50]}...")
    
    print("\nNormalized embeddings:")
    for i, (doc, score) in enumerate(zip(doc_texts, dot_scores_norm)):
        print(f"  Doc {i}: {score:.3f} - {doc[:50]}...")
    
    # Show ranking
    raw_ranking = np.argsort(dot_scores_raw)[::-1]
    norm_ranking = np.argsort(dot_scores_norm)[::-1]
    
    print(f"\nRankings:")
    print(f"  Raw: {raw_ranking.tolist()}")
    print(f"  Normalized: {norm_ranking.tolist()}")
    
    # Compare with cosine
    cos_scores_norm = cosine_similarity(query_norm.reshape(1, -1), docs_norm)[0]
    
    print(f"\nDot product vs Cosine (normalized):")
    for i in range(len(doc_texts)):
        print(f"  Doc {i}: Dot={dot_scores_norm[i]:.3f}, Cos={cos_scores_norm[i]:.3f}")
    
    print(f"\nKey insights:")
    print(f"• Dot product = cosine × vector magnitudes")
    print(f"• For normalized vectors: dot product = cosine similarity")
    print(f"• Often faster to compute than cosine")
    print(f"• Used in FAISS IndexFlatIP (Inner Product)")


def demonstrate_euclidean_distance(query_raw, docs_raw, query_norm, docs_norm, doc_texts):
    """
    Demonstrates Euclidean (L2) distance and its relationship to cosine
    """
    print(f"\n" + "=" * 50)
    print("📏 EUCLIDEAN (L2) DISTANCE")
    print("=" * 50)
    
    print("Formula: ||A - B||₂ = √(Σ(Ai - Bi)²)")
    print("Range: [0, +∞] (0 = identical, higher = more different)")
    print()
    
    # Calculate Euclidean distance
    eucl_dist_raw = euclidean_distances(query_raw.reshape(1, -1), docs_raw)[0]
    eucl_dist_norm = euclidean_distances(query_norm.reshape(1, -1), docs_norm)[0]
    
    print("Raw embeddings:")
    for i, (doc, dist) in enumerate(zip(doc_texts, eucl_dist_raw)):
        print(f"  Doc {i}: {dist:.3f} - {doc[:50]}...")
    
    print("\nNormalized embeddings:")
    for i, (doc, dist) in enumerate(zip(doc_texts, eucl_dist_norm)):
        print(f"  Doc {i}: {dist:.3f} - {doc[:50]}...")
    
    # Show ranking (ascending for distance)
    raw_ranking = np.argsort(eucl_dist_raw)
    norm_ranking = np.argsort(eucl_dist_norm)
    
    print(f"\nRankings (closest first):")
    print(f"  Raw: {raw_ranking.tolist()}")
    print(f"  Normalized: {norm_ranking.tolist()}")
    
    # Relationship to cosine for normalized vectors
    cos_scores_norm = cosine_similarity(query_norm.reshape(1, -1), docs_norm)[0]
    eucl_from_cosine = np.sqrt(2 * (1 - cos_scores_norm))
    
    print(f"\nEuclidean vs Cosine relationship (normalized):")
    print(f"Formula: ||A - B||₂ = √(2(1 - cos(θ))) for unit vectors")
    for i in range(len(doc_texts)):
        print(f"  Doc {i}: Euclidean={eucl_dist_norm[i]:.3f}, From_Cosine={eucl_from_cosine[i]:.3f}")
    
    print(f"\nKey insights:")
    print(f"• Euclidean measures straight-line distance")
    print(f"• For normalized vectors: consistent ranking with cosine")
    print(f"• √(2(1-cos(θ))) formula shows mathematical relationship")
    print(f"• Used in FAISS IndexFlatL2")


def normalization_impact_demo():
    """
    Shows the impact of normalization on similarity rankings
    """
    print(f"\n" + "=" * 60)
    print("NORMALIZATION IMPACT ON RANKINGS")
    print("=" * 60)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create embeddings with different magnitudes
    docs = [
        "Short text",  # Will have smaller magnitude
        "This is a much longer text with many more words that will result in a larger embedding magnitude",  # Larger magnitude
        "Medium length policy text about leave"  # Medium magnitude
    ]
    
    query = "leave policy"
    
    embeddings = model.encode(docs)
    query_emb = model.encode([query])
    
    print(f"Query: '{query}'")
    print(f"\nDocuments and their properties:")
    for i, doc in enumerate(docs):
        emb = embeddings[i]
        norm = np.linalg.norm(emb)
        print(f"  Doc {i}: L2 norm = {norm:.3f}")
        print(f"         Text: '{doc}'")
    
    # Compare metrics before and after normalization
    print(f"\n" + "-" * 40)
    print("BEFORE NORMALIZATION:")
    
    # Cosine (already normalized internally)
    cos_scores = cosine_similarity(query_emb.reshape(1, -1), embeddings)[0]
    
    # Dot product (raw)
    dot_scores = np.dot(query_emb, embeddings.T)[0]
    
    # L2 distance (raw)
    l2_distances = euclidean_distances(query_emb.reshape(1, -1), embeddings)[0]
    
    print("Cosine similarity:")
    for i, score in enumerate(cos_scores):
        print(f"  Doc {i}: {score:.3f}")
    
    print("Dot product:")
    for i, score in enumerate(dot_scores):
        print(f"  Doc {i}: {score:.3f}")
    
    print("L2 distance:")
    for i, dist in enumerate(l2_distances):
        print(f"  Doc {i}: {dist:.3f}")
    
    print(f"\n" + "-" * 40)
    print("AFTER NORMALIZATION:")
    
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_emb_norm = query_emb / np.linalg.norm(query_emb)
    
    # Recalculate metrics
    cos_scores_norm = cosine_similarity(query_emb_norm.reshape(1, -1), embeddings_norm)[0]
    dot_scores_norm = np.dot(query_emb_norm, embeddings_norm.T)[0]
    l2_distances_norm = euclidean_distances(query_emb_norm.reshape(1, -1), embeddings_norm)[0]
    
    print("Cosine similarity (should be same):")
    for i, score in enumerate(cos_scores_norm):
        print(f"  Doc {i}: {score:.3f}")
    
    print("Dot product (now equals cosine):")
    for i, score in enumerate(dot_scores_norm):
        print(f"  Doc {i}: {score:.3f}")
    
    print("L2 distance (consistent with cosine):")
    for i, dist in enumerate(l2_distances_norm):
        print(f"  Doc {i}: {dist:.3f}")
    
    print(f"\nKey takeaway:")
    print(f"• Normalization ensures consistent rankings across metrics")
    print(f"• Essential for fair comparison of embeddings")
    print(f"• Most RAG systems use normalized embeddings")


def similarity_metrics_visualization():
    """
    Visualizes how different similarity metrics behave
    """
    print(f"\n" + "=" * 60)
    print("SIMILARITY METRICS VISUALIZATION")
    print("=" * 60)
    
    # Create simple 2D vectors for visualization
    query_vec = np.array([1, 0])  # Reference vector
    
    # Test vectors at different angles and magnitudes
    angles = np.linspace(0, np.pi, 19)  # 0 to 180 degrees
    magnitudes = [0.5, 1.0, 2.0]  # Different magnitudes
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for mag_idx, magnitude in enumerate(magnitudes):
        ax = axes[mag_idx]
        
        cosine_sims = []
        dot_products = []
        l2_distances = []
        
        for angle in angles:
            # Create test vector
            test_vec = magnitude * np.array([np.cos(angle), np.sin(angle)])
            
            # Calculate similarities
            cos_sim = np.dot(query_vec, test_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(test_vec))
            dot_prod = np.dot(query_vec, test_vec)
            l2_dist = np.linalg.norm(query_vec - test_vec)
            
            cosine_sims.append(cos_sim)
            dot_products.append(dot_prod)
            l2_distances.append(l2_dist)
        
        # Convert angles to degrees for plotting
        angles_deg = np.degrees(angles)
        
        # Plot
        ax.plot(angles_deg, cosine_sims, 'b-', label='Cosine Similarity', linewidth=2)
        ax.plot(angles_deg, dot_products, 'r--', label='Dot Product', linewidth=2)
        ax.plot(angles_deg, np.array(l2_distances) / max(l2_distances), 'g:', 
                label='L2 Distance (normalized)', linewidth=2)
        
        ax.set_title(f'Magnitude = {magnitude}')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Similarity Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
    
    plt.tight_layout()
    plt.suptitle('How Similarity Metrics Behave with Angle and Magnitude', 
                 fontsize=16, y=1.02)
    plt.show()
    
    print("Observations from the visualization:")
    print("• Cosine similarity only depends on angle (magnitude-invariant)")
    print("• Dot product depends on both angle and magnitude")  
    print("• L2 distance increases with both angle and magnitude difference")
    print("• Normalization makes all metrics focus on angle only")


def practical_similarity_demo():
    """
    Practical demonstration with real HR policy examples
    """
    print(f"\n" + "=" * 60)
    print("PRACTICAL SIMILARITY DEMO - HR POLICIES")
    print("=" * 60)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # HR policy documents
    policies = [
        "Our company provides 26 weeks of maternity leave with full pay",
        "Employees are entitled to 2 weeks of paternity leave after childbirth", 
        "The office remains closed on all national holidays including Christmas",
        "Health insurance covers medical, dental, and vision care for all employees",
        "Remote work is available up to 2 days per week with manager approval"
    ]
    
    # Different types of queries
    queries = [
        "What is the maternity leave policy?",
        "When is the office closed?", 
        "What health benefits are provided?",
        "Can employees work from home?"
    ]
    
    # Get embeddings
    embeddings = model.encode(policies, normalize_embeddings=True)  # Pre-normalized
    
    print("Policy documents:")
    for i, policy in enumerate(policies):
        print(f"  {i}: {policy}")
    
    print(f"\n" + "-" * 50)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Get query embedding
        query_emb = model.encode([query], normalize_embeddings=True)
        
        # Calculate similarities (all equivalent for normalized vectors)
        cosine_scores = cosine_similarity(query_emb, embeddings)[0]
        dot_scores = np.dot(query_emb, embeddings.T)[0]
        l2_distances = euclidean_distances(query_emb, embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(cosine_scores)
        
        print(f"Best match: Policy {best_idx}")
        print(f"  Text: {policies[best_idx]}")
        print(f"  Cosine: {cosine_scores[best_idx]:.3f}")
        print(f"  Dot product: {dot_scores[best_idx]:.3f}")
        print(f"  L2 distance: {l2_distances[best_idx]:.3f}")
        
        # Show top 3 for this query
        top_indices = np.argsort(cosine_scores)[::-1][:3]
        print(f"  Top 3 matches:")
        for rank, idx in enumerate(top_indices):
            print(f"    {rank+1}. Policy {idx} (score: {cosine_scores[idx]:.3f})")
    
    print(f"\nKey insights:")
    print(f"• With normalized embeddings, all metrics give consistent rankings")
    print(f"• Semantic matching works across different phrasings")
    print(f"• Higher scores indicate better semantic match")
    print(f"• This forms the foundation of RAG retrieval")


def main():
    """
    Run all similarity metrics demonstrations
    """
    print("📐 BLOG 2.2 - SECTION 2: SIMILARITY METRICS")
    print("This covers all similarity metrics from the blog post")
    print()
    
    # 1. Basic comparison of all metrics
    query_raw, docs_raw, query_norm, docs_norm, doc_texts = demonstrate_all_similarity_metrics()
    
    # 2. Normalization impact
    normalization_impact_demo()
    
    # 3. Visual comparison
    similarity_metrics_visualization()
    
    # 4. Practical demo
    practical_similarity_demo()
    
    print("\n" + "=" * 60)
    print("✅ SIMILARITY METRICS COMPLETE")
    print("=" * 60)
    print("Key concepts demonstrated:")
    print("• Cosine similarity: angle between vectors, magnitude-invariant")
    print("• Dot product: includes both angle and magnitude effects")  
    print("• Euclidean distance: straight-line distance in vector space")
    print("• Normalization makes all metrics consistent")
    print("• Choice of metric affects retrieval results")
    print()
    print("👉 Rule of thumb: Normalize embeddings, then cosine = dot = L2 rankings")
    print()
    print("Next: Run 03_indexing_structures.py to explore ANN indexes")


if __name__ == "__main__":
    main()