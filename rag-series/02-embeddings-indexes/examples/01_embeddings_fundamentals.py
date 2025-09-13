#!/usr/bin/env python3
"""
Blog 2.2 - Section 1: Embeddings Fundamentals
=============================================

This script demonstrates core embedding concepts:
- What embeddings are and how they capture semantic meaning
- Creating embeddings with different models
- Understanding embedding dimensions and properties
- Visualizing embeddings in 2D space

Covers blog sections:
- 1️⃣ Embeddings: The Meaning Map
- 🗺️ What are embeddings?
- ⚙️ How are they created?
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def demonstrate_embedding_basics():
    """
    Shows what embeddings are and their basic properties
    """
    print("=" * 60)
    print("EMBEDDING FUNDAMENTALS DEMONSTRATION")
    print("=" * 60)
    
    # Load embedding model
    print("Loading embedding model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Example sentences from blog
    sentences = [
        "maternity leave 26 weeks",
        "parental leave policy", 
        "pizza recipe",
        "Our company provides 26 weeks of maternity leave",
        "Employee vacation time policy",
        "How to make homemade pizza dough"
    ]
    
    print(f"\nCreating embeddings for {len(sentences)} sentences:")
    for i, sentence in enumerate(sentences):
        print(f"  {i+1}. {sentence}")
    
    # Create embeddings
    embeddings = model.encode(sentences)
    
    print(f"\nEmbedding Properties:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dimensions: {embeddings.shape[1]}")
    print(f"  Data type: {embeddings.dtype}")
    
    # Show first embedding sample
    print(f"\nFirst embedding sample (first 10 dimensions):")
    print(f"  {embeddings[0][:10]}")
    
    # Embedding statistics
    print(f"\nEmbedding Statistics:")
    for i, (sentence, emb) in enumerate(zip(sentences, embeddings)):
        norm = np.linalg.norm(emb)
        mean_val = np.mean(emb)
        std_val = np.std(emb)
        print(f"  {i+1}. L2 norm: {norm:.3f}, Mean: {mean_val:.3f}, Std: {std_val:.3f}")
    
    return sentences, embeddings, model


def visualize_embeddings_2d(sentences, embeddings, method='pca'):
    """
    Visualize high-dimensional embeddings in 2D space
    """
    print(f"\n" + "=" * 60)
    print(f"2D EMBEDDING VISUALIZATION ({method.upper()})")
    print("=" * 60)
    
    # Reduce dimensions to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        title = f"PCA Visualization (explained variance: {reducer.explained_variance_ratio_.sum():.2%})"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sentences)-1))
        reduced = reducer.fit_transform(embeddings)
        title = "t-SNE Visualization"
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Define colors for different categories
    colors = ['red', 'red', 'blue', 'red', 'green', 'blue']  # Similar concepts same color
    labels = ['Work Policy', 'Work Policy', 'Food', 'Work Policy', 'Work Policy', 'Food']
    
    # Plot points
    for i, (x, y) in enumerate(reduced):
        plt.scatter(x, y, c=colors[i], s=100, alpha=0.7)
        plt.annotate(f"{i+1}", (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='red', s=100, alpha=0.7, label='Work Policy'),
        plt.scatter([], [], c='green', s=100, alpha=0.7, label='General Policy'), 
        plt.scatter([], [], c='blue', s=100, alpha=0.7, label='Food Recipe')
    ]
    plt.legend(handles=legend_elements)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, sentence in enumerate(sentences):
        x, y = reduced[i]
        plt.annotate(sentence[:30] + "..." if len(sentence) > 30 else sentence, 
                    (x, y), xytext=(10, -10), textcoords='offset points',
                    fontsize=8, alpha=0.8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
    
    plt.tight_layout()
    plt.show()
    
    print("Key observations:")
    print("• Similar concepts (work policies) cluster together")
    print("• Dissimilar concepts (food recipes) are far apart")
    print("• Semantic similarity is captured in spatial proximity")


def compare_embedding_models():
    """
    Compare different embedding models and their characteristics
    """
    print(f"\n" + "=" * 60)
    print("COMPARING DIFFERENT EMBEDDING MODELS")
    print("=" * 60)
    
    # Different models to compare
    models = {
        "all-MiniLM-L6-v2": "Lightweight, fast, 384 dimensions",
        "all-mpnet-base-v2": "High quality, slower, 768 dimensions",  
        "paraphrase-MiniLM-L6-v2": "Paraphrase detection, 384 dimensions"
    }
    
    test_sentence = "Our company provides 26 weeks of maternity leave"
    
    print(f"Test sentence: '{test_sentence}'")
    print()
    
    for model_name, description in models.items():
        try:
            print(f"Loading {model_name}...")
            model = SentenceTransformer(model_name)
            
            # Create embedding
            embedding = model.encode([test_sentence])
            
            # Model properties
            print(f"  Description: {description}")
            print(f"  Dimensions: {embedding.shape[1]}")
            print(f"  L2 norm: {np.linalg.norm(embedding):.3f}")
            print(f"  Mean value: {np.mean(embedding):.3f}")
            print(f"  Std deviation: {np.std(embedding):.3f}")
            print(f"  Sample values: {embedding[0][:5]}")
            print()
            
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            print()


def demonstrate_embedding_properties():
    """
    Shows key properties of embeddings mentioned in the blog
    """
    print(f"\n" + "=" * 60)
    print("EMBEDDING PROPERTIES ANALYSIS")
    print("=" * 60)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Test semantic relationships
    test_pairs = [
        # Similar meaning pairs
        ("maternity leave policy", "parental leave benefits"),
        ("employee vacation time", "worker holiday schedule"), 
        ("pizza recipe", "cooking instructions"),
        
        # Dissimilar pairs
        ("maternity leave policy", "pizza recipe"),
        ("employee vacation time", "cooking instructions"),
        ("parental leave benefits", "homemade pizza")
    ]
    
    print("Semantic Similarity Analysis:")
    print("(Higher cosine similarity = more similar meaning)")
    print()
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    for i, (text1, text2) in enumerate(test_pairs):
        # Get embeddings
        emb1 = model.encode([text1])
        emb2 = model.encode([text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        # Determine relationship type
        relationship = "Similar" if i < 3 else "Different"
        
        print(f"{relationship} concepts:")
        print(f"  '{text1}'")
        print(f"  '{text2}'")
        print(f"  Cosine similarity: {similarity:.3f}")
        print()
    
    print("Key insights:")
    print("• Embeddings capture semantic relationships")
    print("• Similar concepts have high cosine similarity (>0.5)")
    print("• Unrelated concepts have low similarity (<0.3)")
    print("• This enables semantic search beyond keyword matching")


def demonstrate_contrastive_learning_concept():
    """
    Illustrates the contrastive learning concept mentioned in the blog
    """
    print(f"\n" + "=" * 60)
    print("CONTRASTIVE LEARNING CONCEPT DEMONSTRATION")
    print("=" * 60)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Anchor sentence
    anchor = "Our company provides maternity leave benefits"
    
    # Positive examples (similar meaning)
    positives = [
        "The company offers parental leave policies",
        "Maternity benefits are provided by our organization",
        "We have leave policies for new mothers"
    ]
    
    # Negative examples (different meaning)  
    negatives = [
        "How to make homemade pizza dough",
        "The weather is sunny today",
        "Programming languages are useful tools"
    ]
    
    print(f"Anchor: '{anchor}'")
    print()
    
    # Get embeddings
    anchor_emb = model.encode([anchor])
    pos_embs = model.encode(positives)
    neg_embs = model.encode(negatives)
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("Positive examples (should be similar to anchor):")
    for i, (pos, pos_emb) in enumerate(zip(positives, pos_embs)):
        similarity = cosine_similarity(anchor_emb, [pos_emb])[0][0]
        print(f"  {i+1}. Similarity: {similarity:.3f} - '{pos}'")
    
    print("\nNegative examples (should be different from anchor):")
    for i, (neg, neg_emb) in enumerate(zip(negatives, neg_embs)):
        similarity = cosine_similarity(anchor_emb, [neg_emb])[0][0]
        print(f"  {i+1}. Similarity: {similarity:.3f} - '{neg}'")
    
    # Calculate average similarities
    avg_pos_sim = np.mean([cosine_similarity(anchor_emb, [emb])[0][0] for emb in pos_embs])
    avg_neg_sim = np.mean([cosine_similarity(anchor_emb, [emb])[0][0] for emb in neg_embs])
    
    print(f"\nSummary:")
    print(f"  Average positive similarity: {avg_pos_sim:.3f}")
    print(f"  Average negative similarity: {avg_neg_sim:.3f}")
    print(f"  Separation margin: {avg_pos_sim - avg_neg_sim:.3f}")
    
    print("\nContrastive learning insight:")
    print("• Training pulls similar examples closer (higher similarity)")
    print("• Training pushes dissimilar examples apart (lower similarity)")
    print("• This creates meaningful semantic representations")


def main():
    """
    Run all embedding fundamentals demonstrations
    """
    print("🔍 BLOG 2.2 - SECTION 1: EMBEDDINGS FUNDAMENTALS")
    print("This covers all embedding concepts from the blog post")
    print()
    
    # 1. Basic embedding demonstration
    sentences, embeddings, model = demonstrate_embedding_basics()
    
    # 2. 2D visualization 
    visualize_embeddings_2d(sentences, embeddings, method='pca')
    
    # 3. Model comparison
    compare_embedding_models()
    
    # 4. Embedding properties
    demonstrate_embedding_properties()
    
    # 5. Contrastive learning concept
    demonstrate_contrastive_learning_concept()
    
    print("\n" + "=" * 60)
    print("✅ EMBEDDINGS FUNDAMENTALS COMPLETE")
    print("=" * 60)
    print("Key concepts demonstrated:")
    print("• Embeddings capture semantic meaning as vectors")
    print("• High-dimensional representations enable similarity search")
    print("• Contrastive learning creates meaningful semantic spaces")
    print("• Different models offer different trade-offs")
    print()
    print("Next: Run 02_similarity_metrics.py to explore similarity measures")


if __name__ == "__main__":
    main()