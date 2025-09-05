#!/usr/bin/env python3
"""
RAG Fundamentals Demo - All Examples
====================================

This script runs all the RAG fundamental examples from Blog 2.1.
It demonstrates:
- Basic RAG retrieval process
- Embedding visualization  
- Core RAG concepts (chunking, embeddings, similarity metrics)
- Complete RAG pipeline

Run this script to see all RAG fundamentals concepts in action.

Usage:
    python demo_all_rag_fundamentals.py
"""

import sys
import traceback
from pathlib import Path

def run_example(example_name: str, run_function):
    """
    Runs a single example with error handling and formatting
    """
    print("\n" + "🚀 " + "="*80)
    print(f"RUNNING: {example_name}")
    print("="*80)
    
    try:
        run_function()
        print(f"\n✅ {example_name} completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in {example_name}:")
        print(f"   {str(e)}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """
    Run all RAG fundamentals examples
    """
    print("🔍 RAG FUNDAMENTALS - COMPLETE DEMONSTRATION")
    print("=" * 80)
    print("This demo covers all concepts from Blog 2.1: RAG Fundamentals")
    print("- Basic retrieval with sentence transformers")
    print("- Embedding visualization in 2D space")
    print("- Core concepts: chunking, embeddings, similarity metrics")
    print("- Complete RAG pipeline walkthrough")
    print("=" * 80)
    
    examples_run = 0
    examples_passed = 0
    
    # Import examples
    try:
        from basic_rag_retrieval import basic_rag_retrieval_demo, top_k_retrieval_demo
        from embedding_visualization import visualize_embeddings, compare_different_queries  
        from rag_concepts_demo import RAGConceptsDemo
        
    except ImportError as e:
        print(f"❌ Failed to import examples: {e}")
        print("Make sure you're running this script from the examples directory")
        print("and all required packages are installed (run: pip install -r requirements.txt)")
        return
    
    # Example 1: Basic RAG Retrieval
    examples_run += 1
    if run_example("Basic RAG Retrieval", basic_rag_retrieval_demo):
        examples_passed += 1
    
    # Example 2: Top-K Retrieval
    examples_run += 1  
    if run_example("Top-K Retrieval", top_k_retrieval_demo):
        examples_passed += 1
    
    # Example 3: Embedding Visualization
    examples_run += 1
    if run_example("Embedding Visualization", visualize_embeddings):
        examples_passed += 1
    
    # Example 4: Query Comparison
    examples_run += 1
    if run_example("Different Queries Comparison", compare_different_queries):
        examples_passed += 1
    
    # Example 5: Complete RAG Concepts Demo
    examples_run += 1
    demo = RAGConceptsDemo()
    if run_example("Complete RAG Concepts Demo", demo.full_rag_pipeline_demo):
        examples_passed += 1
    
    # Final Summary
    print("\n" + "🎯 " + "="*80)
    print("FINAL SUMMARY") 
    print("="*80)
    print(f"Examples run: {examples_run}")
    print(f"Examples passed: {examples_passed}")
    print(f"Success rate: {examples_passed/examples_run*100:.1f}%")
    
    if examples_passed == examples_run:
        print("\n🎉 All RAG fundamentals examples completed successfully!")
        print("\nWhat you've learned:")
        print("✓ How RAG retrieval works with sentence transformers")
        print("✓ How embeddings capture semantic meaning") 
        print("✓ Different similarity metrics and their trade-offs")
        print("✓ The importance of top-k retrieval for context")
        print("✓ The complete RAG pipeline from query to generation")
        
        print("\n📚 Key Concepts Reinforced:")
        print("• Corpus → Complete knowledge base")
        print("• Documents → Individual files")
        print("• Chunks → Smaller document pieces") 
        print("• Embeddings → Vector representations of meaning")
        print("• Retriever → System that finds relevant chunks")
        print("• Generator → LLM that creates answers from context")
        
        print("\n🔮 Next Steps:")
        print("In Blog 2.2, we'll explore:")
        print("• Vector databases (FAISS, Pinecone, Chroma)")
        print("• Advanced indexing techniques")
        print("• Performance optimization")
        print("• Production RAG systems")
        
    else:
        print(f"\n⚠️  {examples_run - examples_passed} example(s) had issues.")
        print("Check the error messages above and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()