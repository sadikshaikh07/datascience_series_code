# RAG Fundamentals: Solving the Knowledge Problem

This directory contains practical code examples for **Blog 2.1: RAG Fundamentals** from the Data Science Series.

## 🔍 What You'll Learn

- **What is RAG?** - Retrieval-Augmented Generation explained with real code
- **The RAG Pipeline** - Query → Retrieve → Generate process  
- **Core Concepts** - Embeddings, similarity search, top-k retrieval
- **Hands-on Implementation** - Working examples with sentence transformers

## 📁 Code Examples

### Core Examples

1. **`basic_rag_retrieval.py`** - Basic RAG retrieval demonstration
   - Load embedding models
   - Create document embeddings  
   - Perform similarity search
   - Top-k retrieval examples

2. **`embedding_visualization.py`** - Visualize how embeddings work
   - 2D visualization of embedding space
   - See why certain documents are retrieved
   - Compare different queries

3. **`rag_concepts_demo.py`** - Complete concepts walkthrough
   - Document chunking process
   - Embedding properties and vectors
   - Different similarity metrics comparison
   - Full RAG pipeline demonstration

4. **`demo_all_rag_fundamentals.py`** - Run all examples together
   - Master script that runs all demos
   - Complete walkthrough of RAG fundamentals
   - Progress tracking and error handling

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install sentence-transformers scikit-learn numpy matplotlib
```

### Running the Examples

**Option 1: Run all examples (Recommended)**
```bash
python demo_all_rag_fundamentals.py
```

**Option 2: Run individual examples**
```bash
python basic_rag_retrieval.py
python embedding_visualization.py  
python rag_concepts_demo.py
```

## 📊 What Each Example Demonstrates

### Basic RAG Retrieval
- Loading sentence transformer models
- Creating embeddings for a document corpus
- Performing cosine similarity search
- Understanding retrieval scores and ranking

**Key Output:**
```
Query: What is the maternity leave policy?
Most Relevant Doc: Our company provides 26 weeks of maternity leave.
Cosine Scores: [0.945 0.721 0.102]
```

### Embedding Visualization
- 2D visualization of high-dimensional embeddings
- Visual understanding of similarity search
- How different queries retrieve different documents

**Key Insight:** Documents with similar meaning cluster together in embedding space.

### RAG Concepts Demo
- Document chunking strategies
- Embedding properties and dimensions
- Comparison of similarity metrics (cosine, dot product, euclidean)
- Complete RAG pipeline from query to answer generation

## 🧠 Key Concepts Covered

### RAG Components
- **Corpus** - Your complete knowledge base
- **Documents** - Individual files in your corpus  
- **Chunks** - Smaller pieces of documents (200-500 words)
- **Embeddings** - Vector representations capturing meaning
- **Retriever** - System that finds relevant chunks
- **Generator** - LLM that creates answers using context

### Technical Concepts  
- **Similarity Metrics** - Cosine, dot product, euclidean distance
- **Top-k Retrieval** - Getting multiple relevant documents for context
- **Embedding Models** - sentence-transformers for semantic search
- **Vector Operations** - How similarity search works mathematically

## 💡 Real-World Applications

The examples simulate real RAG scenarios:
- **HR Policy Bot** - Query employee policies and get accurate answers
- **Document Search** - Find relevant information in large document collections
- **FAQ Systems** - Automatically answer questions from knowledge bases

## 🔗 Related Blog Content

This code supports [Blog 2.1: RAG Fundamentals](https://medium.com/@sadikkhadeer/rag-fundamentals-solving-the-knowledge-problem-7d4f6b0eda3a)

**Blog Series Context:**
- **Series 1:** AI Fundamentals & Agent Basics (Prerequisites)
- **Series 2:** RAG Systems ← You are here
- **Future:** Vector Databases, Advanced RAG, Production Systems

## 🛠 Troubleshooting

**Common Issues:**

1. **Import Error: No module named 'sentence_transformers'**
   ```bash
   pip install sentence-transformers
   ```

2. **Visualization not showing:** 
   - Make sure matplotlib is installed
   - Use `plt.show()` or run in Jupyter notebook

3. **Slow first run:** 
   - Sentence transformer models download on first use
   - Subsequent runs will be faster

## 🎯 Learning Path

**Before this:** Complete Series 1 (AI Fundamentals & Agent Basics)
**After this:** Move to Blog 2.2 (Embeddings, Indexes & Retrieval Mechanics)

**Next topics:**
- Vector databases (FAISS, Pinecone, Chroma)
- Advanced indexing techniques  
- Production RAG systems
- Performance optimization

## 📈 Expected Outcomes

After running these examples, you should understand:
- ✅ How RAG solves LLM knowledge limitations
- ✅ The complete RAG retrieval pipeline
- ✅ How embeddings capture semantic meaning
- ✅ Why top-k retrieval provides better context
- ✅ Different similarity metrics and their trade-offs
- ✅ How to implement basic RAG retrieval from scratch

---

*Part of the [Data Science Series](https://medium.com/@sadikkhadeer/data-science-series-complete-learning-path-updated-weekly-83611dea41fb) - Comprehensive learning path from AI fundamentals to advanced specializations.*