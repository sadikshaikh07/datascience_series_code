#!/usr/bin/env python3
"""
Blog 2.2 - Section 3: Indexing Structures (ANN)
===============================================

This script demonstrates different Approximate Nearest Neighbor (ANN) indexing structures:
- Flat (Exact) search
- HNSW (Hierarchical Navigable Small World)
- IVF (Inverted File Index)  
- PQ (Product Quantization)
- Performance comparisons and trade-offs

Covers blog sections:
- 3️⃣ Indexes (ANN): The Real Engine of Retrieval
- 🎯 Flat (Exact), 🕸️ HNSW, 📂 IVF, 🗜️ PQ
- 3.8 Choosing an index (cheat-sheet)
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class IndexBenchmarker:
    """
    Comprehensive benchmarker for different FAISS index types
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.queries = None
        self.indexes = {}
        self.benchmark_results = {}
    
    def create_corpus(self, size: str = "medium") -> Tuple[List[str], np.ndarray]:
        """
        Create a corpus of different sizes for benchmarking
        """
        print(f"Creating {size} corpus...")
        
        base_docs = [
            "Our company provides 26 weeks of maternity leave with full pay and benefits",
            "Employees are eligible for 2 weeks of paternity leave immediately after birth", 
            "The office remains closed on all national holidays including New Year's Day",
            "Health insurance coverage includes medical, dental, and vision benefits",
            "Remote work arrangements are available up to 2 days per week",
            "Employee training budget provides $2000 annually for professional development",
            "Sick leave accrues at 8 hours per month for all full-time employees",
            "Parental leave can be extended with doctor approval for medical reasons",
            "Holiday schedules are published at the beginning of each fiscal year",
            "Performance reviews are conducted annually with merit-based increases"
        ]
        
        # Generate different corpus sizes
        if size == "small":
            # Duplicate and vary base docs to create ~100 documents
            docs = []
            for i in range(10):
                for base_doc in base_docs:
                    variation = f"{base_doc}. Additional context for document {i}."
                    docs.append(variation)
            target_size = 100
        
        elif size == "medium":
            # Create ~1000 documents
            docs = []
            for i in range(100):
                for base_doc in base_docs:
                    variation = f"{base_doc}. Department variation {i//10}. Document ID: {i}."
                    docs.append(variation)
            target_size = 1000
            
        elif size == "large":
            # Create ~10000 documents
            docs = []
            for i in range(1000):
                for base_doc in base_docs:
                    variation = f"{base_doc}. Sector {i//100}. Department {i//10}. Doc {i}."
                    docs.append(variation)
            target_size = 10000
        
        # Trim to exact size
        docs = docs[:target_size]
        
        print(f"  Generated {len(docs)} documents")
        print(f"  Sample document: {docs[0][:80]}...")
        
        # Create embeddings
        print("  Creating embeddings...")
        embeddings = self.model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
        
        print(f"  Embeddings shape: {embeddings.shape}")
        return docs, embeddings
    
    def create_test_queries(self) -> List[str]:
        """
        Create test queries for benchmarking
        """
        queries = [
            "What is the maternity leave policy?",
            "How much paternity leave do fathers get?",
            "When is the office closed for holidays?",
            "What health insurance benefits are provided?", 
            "Can employees work remotely from home?",
            "What is the employee training budget?",
            "How much sick leave do employees accrue?",
            "Can parental leave be extended?",
            "When are holiday schedules published?",
            "How often are performance reviews conducted?"
        ]
        
        print(f"Created {len(queries)} test queries")
        return queries


def demonstrate_flat_index():
    """
    Demonstrates Flat (Exact) indexing - the baseline
    """
    print("=" * 60)
    print("🎯 FLAT (EXACT) INDEX DEMONSTRATION")
    print("=" * 60)
    
    benchmarker = IndexBenchmarker()
    
    # Create small corpus for exact search demo
    docs, embeddings = benchmarker.create_corpus("small")
    queries = benchmarker.create_test_queries()
    
    # Create flat indexes for different metrics
    d = embeddings.shape[1]
    
    print(f"\nCreating Flat indexes...")
    print(f"  Dimension: {d}")
    print(f"  Number of documents: {len(embeddings)}")
    
    # Different flat indexes
    indexes = {
        "FlatL2": faiss.IndexFlatL2(d),      # Euclidean distance
        "FlatIP": faiss.IndexFlatIP(d),      # Inner Product (dot product)
    }
    
    # Add vectors to indexes
    for name, index in indexes.items():
        index.add(embeddings)
        print(f"  Added {index.ntotal} vectors to {name}")
    
    # Test each index
    for name, index in indexes.items():
        print(f"\n{name} Results:")
        
        # Test with first query
        query = queries[0]
        query_emb = benchmarker.model.encode([query], normalize_embeddings=True)
        
        # Search
        start_time = time.time()
        distances, indices = index.search(query_emb, 5)
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"  Query: '{query}'")
        print(f"  Search time: {search_time:.2f} ms")
        print(f"  Top 5 results:")
        
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            print(f"    {rank+1}. Doc {idx}: {dist:.3f} - {docs[idx][:60]}...")
    
    print(f"\nFlat Index Properties:")
    print(f"• Perfect recall (finds all true nearest neighbors)")
    print(f"• O(N·d) search complexity - scales linearly") 
    print(f"• No training required")
    print(f"• Memory: stores all vectors in full precision")
    print(f"• Best for: <10k vectors or as accuracy baseline")
    
    return embeddings, queries


def demonstrate_hnsw_index():
    """
    Demonstrates HNSW (Hierarchical Navigable Small World) indexing
    """
    print(f"\n" + "=" * 60)
    print("🕸️ HNSW INDEX DEMONSTRATION") 
    print("=" * 60)
    
    benchmarker = IndexBenchmarker()
    docs, embeddings = benchmarker.create_corpus("medium")
    queries = benchmarker.create_test_queries()
    
    d = embeddings.shape[1]
    
    print(f"\nHNSW Index Configuration:")
    print(f"  Dimensions: {d}")
    print(f"  Corpus size: {len(embeddings)} vectors")
    
    # Create HNSW index with different parameters
    M = 32  # Graph degree
    hnsw = faiss.IndexHNSWFlat(d, M)
    
    # Set construction parameters
    hnsw.hnsw.efConstruction = 200  # Build quality
    
    print(f"  M (graph degree): {M}")
    print(f"  efConstruction: {hnsw.hnsw.efConstruction}")
    
    # Add vectors (this builds the graph)
    print(f"\nBuilding HNSW graph...")
    start_build = time.time()
    hnsw.add(embeddings)
    build_time = time.time() - start_build
    
    print(f"  Build time: {build_time:.2f} seconds")
    print(f"  Vectors in index: {hnsw.ntotal}")
    
    # Create flat index for comparison
    flat = faiss.IndexFlatIP(d)
    flat.add(embeddings)
    
    # Test different efSearch values
    ef_search_values = [16, 32, 64, 128, 256]
    
    print(f"\nTesting different efSearch values:")
    print(f"{'efSearch':<10} {'Latency (ms)':<12} {'Recall@5':<10} {'Top-5 Results'}")
    print("-" * 70)
    
    query = queries[0]
    query_emb = benchmarker.model.encode([query], normalize_embeddings=True)
    
    # Get ground truth from flat index
    gt_distances, gt_indices = flat.search(query_emb, 5)
    gt_set = set(gt_indices[0])
    
    for ef in ef_search_values:
        hnsw.hnsw.efSearch = ef
        
        # Benchmark search
        start_time = time.time()
        distances, indices = hnsw.search(query_emb, 5)
        search_time = (time.time() - start_time) * 1000
        
        # Calculate recall
        retrieved_set = set(indices[0])
        recall = len(retrieved_set.intersection(gt_set)) / len(gt_set)
        
        print(f"{ef:<10} {search_time:<12.2f} {recall:<10.2f} {list(indices[0])}")
    
    # Demonstrate the recall-latency trade-off
    plot_hnsw_tradeoffs(hnsw, flat, query_emb, ef_search_values)
    
    print(f"\nHNSW Properties:")
    print(f"• Multi-layer graph structure (highways → streets)")
    print(f"• Excellent recall-latency trade-off")
    print(f"• Memory: ~M × 4 bytes per vector (edges)")
    print(f"• Search complexity: ~O(log N)")
    print(f"• Best for: 100k-100M vectors with good RAM")


def plot_hnsw_tradeoffs(hnsw_index, flat_index, query_emb, ef_values):
    """
    Plot recall vs latency trade-offs for HNSW
    """
    print(f"\nCreating recall-latency visualization...")
    
    latencies = []
    recalls = []
    
    # Get ground truth
    gt_distances, gt_indices = flat_index.search(query_emb, 10)
    gt_set = set(gt_indices[0])
    
    for ef in ef_values:
        hnsw_index.hnsw.efSearch = ef
        
        # Measure multiple queries for stability
        times = []
        recall_sum = 0
        
        for _ in range(5):  # Average over 5 runs
            start = time.time()
            distances, indices = hnsw_index.search(query_emb, 10)
            times.append((time.time() - start) * 1000)
            
            # Calculate recall@10
            retrieved_set = set(indices[0])
            recall_sum += len(retrieved_set.intersection(gt_set)) / len(gt_set)
        
        latencies.append(np.mean(times))
        recalls.append(recall_sum / 5)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(latencies, recalls, 'bo-', linewidth=2, markersize=8)
    
    # Annotate points
    for i, (lat, rec, ef) in enumerate(zip(latencies, recalls, ef_values)):
        plt.annotate(f'ef={ef}', (lat, rec), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.xlabel('Latency (ms)')
    plt.ylabel('Recall@10')
    plt.title('HNSW: Recall vs Latency Trade-off')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Key insights from plot:")
    print(f"• Higher efSearch → better recall but higher latency")
    print(f"• Sweet spot typically around efSearch=64-128")
    print(f"• Diminishing returns beyond efSearch=200")


def demonstrate_ivf_index():
    """
    Demonstrates IVF (Inverted File Index) with parameter tuning
    """
    print(f"\n" + "=" * 60)
    print("📂 IVF (INVERTED FILE INDEX) DEMONSTRATION")
    print("=" * 60)
    
    benchmarker = IndexBenchmarker()
    docs, embeddings = benchmarker.create_corpus("medium") 
    queries = benchmarker.create_test_queries()
    
    d = embeddings.shape[1]
    n = len(embeddings)
    
    # IVF parameters
    nlist = 64  # Number of clusters (rule of thumb: 4*sqrt(n))
    
    print(f"\nIVF Index Configuration:")
    print(f"  Dimensions: {d}")
    print(f"  Corpus size: {n} vectors")
    print(f"  nlist (clusters): {nlist}")
    print(f"  Heuristic nlist ≈ 4√N = {int(4 * np.sqrt(n))}")
    
    # Create IVF index
    quantizer = faiss.IndexFlatIP(d)  # Coarse quantizer
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the index (learn centroids)
    print(f"\nTraining IVF index...")
    start_train = time.time()
    ivf.train(embeddings)
    train_time = time.time() - start_train
    
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Centroids trained: {ivf.nlist}")
    
    # Add vectors to index
    print(f"Adding vectors to index...")
    start_add = time.time()
    ivf.add(embeddings)
    add_time = time.time() - start_add
    
    print(f"  Add time: {add_time:.2f} seconds")
    print(f"  Vectors in index: {ivf.ntotal}")
    
    # Create flat index for ground truth
    flat = faiss.IndexFlatIP(d)
    flat.add(embeddings)
    
    # Test different nprobe values
    nprobe_values = [1, 2, 4, 8, 16, 32]
    
    print(f"\nTesting different nprobe values:")
    print(f"{'nprobe':<8} {'Latency (ms)':<12} {'Recall@5':<10} {'Lists searched':<15}")
    print("-" * 50)
    
    query = queries[0]
    query_emb = benchmarker.model.encode([query], normalize_embeddings=True)
    
    # Get ground truth
    gt_distances, gt_indices = flat.search(query_emb, 5)
    gt_set = set(gt_indices[0])
    
    for nprobe in nprobe_values:
        ivf.nprobe = nprobe
        
        # Benchmark search
        start_time = time.time()
        distances, indices = ivf.search(query_emb, 5)
        search_time = (time.time() - start_time) * 1000
        
        # Calculate recall
        retrieved_set = set(indices[0])
        recall = len(retrieved_set.intersection(gt_set)) / len(gt_set)
        
        lists_searched = f"{nprobe}/{nlist}"
        print(f"{nprobe:<8} {search_time:<12.2f} {recall:<10.2f} {lists_searched:<15}")
    
    print(f"\nIVF Properties:")
    print(f"• Clusters documents, searches only relevant clusters")
    print(f"• Requires training phase to learn centroids")
    print(f"• Memory efficient (no graph structure)")
    print(f"• Can miss neighbors on cluster boundaries")
    print(f"• Best for: millions to hundreds of millions of vectors")
    
    # Show cluster distribution
    analyze_ivf_clusters(ivf, embeddings, queries[:3])


def analyze_ivf_clusters(ivf_index, embeddings, sample_queries):
    """
    Analyze how vectors are distributed across IVF clusters
    """
    print(f"\nIVF CLUSTER ANALYSIS:")
    
    # Get cluster assignments for all vectors
    _, cluster_assignments = ivf_index.quantizer.search(embeddings, 1)
    cluster_assignments = cluster_assignments.flatten()
    
    # Count vectors per cluster
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    
    print(f"  Total clusters: {ivf_index.nlist}")
    print(f"  Clusters with vectors: {len(unique_clusters)}")
    print(f"  Average vectors per cluster: {np.mean(counts):.1f}")
    print(f"  Min/Max vectors per cluster: {np.min(counts)}/{np.max(counts)}")
    
    # Show cluster distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(counts, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Vectors per Cluster')
    plt.ylabel('Number of Clusters')
    plt.title('IVF Cluster Size Distribution')
    plt.grid(True, alpha=0.3)
    
    # Show which clusters are searched for sample queries
    plt.subplot(1, 2, 2)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    all_searched_clusters = set()
    for i, query in enumerate(sample_queries):
        query_emb = model.encode([query], normalize_embeddings=True)
        
        # Find nearest centroids
        ivf_index.nprobe = 4  # Search 4 clusters
        _, searched_clusters = ivf_index.quantizer.search(query_emb, ivf_index.nprobe)
        
        searched = searched_clusters[0]
        all_searched_clusters.update(searched)
        
        # Plot the searched clusters for this query
        y_pos = [i] * len(searched)
        plt.scatter(searched, y_pos, alpha=0.7, s=50, label=f'Query {i+1}')
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Query')
    plt.title('Clusters Searched by Sample Queries')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"  Unique clusters searched: {len(all_searched_clusters)}")
    print(f"  Cluster coverage: {len(all_searched_clusters)/ivf_index.nlist*100:.1f}%")


def demonstrate_pq_compression():
    """
    Demonstrates Product Quantization (PQ) for memory compression
    """
    print(f"\n" + "=" * 60)
    print("🗜️ PRODUCT QUANTIZATION (PQ) DEMONSTRATION")
    print("=" * 60)
    
    benchmarker = IndexBenchmarker()
    docs, embeddings = benchmarker.create_corpus("medium")
    queries = benchmarker.create_test_queries()
    
    d = embeddings.shape[1]
    n = len(embeddings)
    
    print(f"\nPQ Compression Setup:")
    print(f"  Dimensions: {d}")
    print(f"  Corpus size: {n} vectors")
    
    # PQ parameters
    m = 16  # Number of sub-quantizers (must divide d evenly)
    nbits = 8  # Bits per code (256 centroids per sub-quantizer)
    
    print(f"  Sub-quantizers (m): {m}")
    print(f"  Bits per code: {nbits}")
    print(f"  Sub-vector size: {d//m}")
    
    # Create indexes for comparison
    indexes = {
        "Flat": faiss.IndexFlatIP(d),
        "PQ": faiss.IndexPQ(d, m, nbits),
        "IVFPQ": None  # Will create after training
    }
    
    # Memory calculations
    float_mem = n * d * 4  # 4 bytes per float32
    pq_mem = n * m * (nbits // 8)  # Compressed memory
    compression_ratio = float_mem / pq_mem
    
    print(f"\nMemory Analysis:")
    print(f"  Float32 storage: {float_mem/1e6:.1f} MB")
    print(f"  PQ storage: {pq_mem/1e6:.1f} MB") 
    print(f"  Compression ratio: {compression_ratio:.1f}x smaller")
    
    # Train and populate indexes
    print(f"\nTraining indexes...")
    
    # Flat index (baseline)
    indexes["Flat"].add(embeddings)
    
    # PQ index
    print("  Training PQ...")
    start_train = time.time()
    indexes["PQ"].train(embeddings)
    pq_train_time = time.time() - start_train
    indexes["PQ"].add(embeddings)
    
    print(f"    PQ training time: {pq_train_time:.2f} seconds")
    
    # IVFPQ (best of both worlds)
    nlist = 64
    quantizer = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    
    print("  Training IVFPQ...")
    start_train = time.time()
    ivfpq.train(embeddings)
    ivfpq_train_time = time.time() - start_train
    ivfpq.add(embeddings)
    indexes["IVFPQ"] = ivfpq
    
    print(f"    IVFPQ training time: {ivfpq_train_time:.2f} seconds")
    
    # Benchmark all indexes
    print(f"\nBenchmarking indexes:")
    print(f"{'Index':<10} {'Latency (ms)':<12} {'Recall@5':<10} {'Memory':<15}")
    print("-" * 50)
    
    query = queries[0]
    query_emb = benchmarker.model.encode([query], normalize_embeddings=True)
    
    # Get ground truth
    gt_distances, gt_indices = indexes["Flat"].search(query_emb, 5)
    gt_set = set(gt_indices[0])
    
    for name, index in indexes.items():
        if name == "IVFPQ":
            index.nprobe = 8  # Search 8 clusters
        
        # Benchmark
        start_time = time.time()
        distances, indices = index.search(query_emb, 5)
        search_time = (time.time() - start_time) * 1000
        
        # Calculate recall
        retrieved_set = set(indices[0])
        recall = len(retrieved_set.intersection(gt_set)) / len(gt_set)
        
        # Memory estimate
        if name == "Flat":
            memory = f"{float_mem/1e6:.1f} MB"
        elif name == "PQ":
            memory = f"{pq_mem/1e6:.1f} MB"
        else:  # IVFPQ
            memory = f"{pq_mem/1e6:.1f} MB + idx"
        
        print(f"{name:<10} {search_time:<12.2f} {recall:<10.2f} {memory:<15}")
    
    print(f"\nPQ Properties:")
    print(f"• Massive memory savings (10-200x compression)")
    print(f"• Some accuracy loss due to quantization")
    print(f"• IVFPQ combines speed and compression")
    print(f"• Best for: 100M+ vectors with memory constraints")
    
    # Show PQ reconstruction error
    demonstrate_pq_accuracy(indexes["PQ"], embeddings[:100])


def demonstrate_pq_accuracy(pq_index, sample_embeddings):
    """
    Shows how PQ affects accuracy through reconstruction
    """
    print(f"\nPQ ACCURACY ANALYSIS:")
    
    n_samples = len(sample_embeddings)
    
    # Reconstruct vectors from PQ codes
    codes = pq_index.sa_encode(sample_embeddings)
    reconstructed = pq_index.sa_decode(codes)
    
    # Calculate reconstruction errors
    mse_errors = np.mean((sample_embeddings - reconstructed) ** 2, axis=1)
    cosine_similarities = []
    
    for orig, recon in zip(sample_embeddings, reconstructed):
        cos_sim = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon))
        cosine_similarities.append(cos_sim)
    
    print(f"  Samples analyzed: {n_samples}")
    print(f"  Average MSE: {np.mean(mse_errors):.6f}")
    print(f"  Average cosine similarity: {np.mean(cosine_similarities):.3f}")
    print(f"  Min cosine similarity: {np.min(cosine_similarities):.3f}")
    
    # Plot error distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(mse_errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('MSE Error')
    plt.ylabel('Count')
    plt.title('PQ Reconstruction Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2) 
    plt.hist(cosine_similarities, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Cosine Similarity (Original vs Reconstructed)')
    plt.ylabel('Count')
    plt.title('PQ Reconstruction Quality')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Key insights:")
    print(f"• PQ lossy compression trades accuracy for memory")
    print(f"• Higher m and nbits → better accuracy, more memory")
    print(f"• Cosine similarity usually >0.9 with good parameters")


def index_selection_guide():
    """
    Provides practical guidance on choosing indexes based on blog's cheat sheet
    """
    print(f"\n" + "=" * 60)
    print("📋 INDEX SELECTION GUIDE")
    print("=" * 60)
    
    selection_guide = {
        "≤ 50k vectors": {
            "recommendation": "IndexFlatIP/L2",
            "reasons": ["Perfect recall", "No training needed", "Simple setup"],
            "use_case": "Prototypes, small corpora, accuracy baseline"
        },
        "50k - 1M vectors": {
            "recommendation": "IndexHNSWFlat", 
            "reasons": ["Great recall-latency", "No training", "Good default"],
            "use_case": "Most production RAG systems"
        },
        "1M - 10M vectors": {
            "recommendation": "IndexHNSWFlat or IndexIVFFlat",
            "reasons": ["HNSW if RAM available", "IVF if memory constrained"],
            "use_case": "Large document collections"
        },
        "10M+ vectors": {
            "recommendation": "IndexIVFPQ",
            "reasons": ["Memory efficient", "Good speed", "Scalable"],
            "use_case": "Enterprise-scale deployments"
        },
        "Special cases": {
            "Read-heavy workload": "Annoy (static dataset)",
            "Extreme throughput": "ScaNN (Google)",
            "Disk-based": "DiskANN (Microsoft)",
            "GPU acceleration": "IndexIVFPQ with GPU"
        }
    }
    
    print("Index Selection Cheat Sheet:")
    print("=" * 40)
    
    for scenario, details in selection_guide.items():
        if scenario != "Special cases":
            print(f"\n{scenario}:")
            print(f"  → {details['recommendation']}")
            print(f"  Reasons: {', '.join(details['reasons'])}")
            print(f"  Use case: {details['use_case']}")
        else:
            print(f"\n{scenario}:")
            for case, rec in details.items():
                print(f"  • {case}: {rec}")
    
    print(f"\n" + "=" * 40)
    print("Decision Flow:")
    print("1. Start with Flat for baseline accuracy")
    print("2. Try HNSW for balanced performance")  
    print("3. Use IVF/PQ only when scale demands it")
    print("4. Always benchmark on your own data!")
    
    # Practical example
    print(f"\nPractical Example:")
    print(f"For a typical RAG system with 100k documents:")
    print(f"  → Start: IndexFlatIP (baseline)")
    print(f"  → Production: IndexHNSWFlat (M=32, efSearch=64)")
    print(f"  → If memory tight: IndexIVFFlat (nlist=400, nprobe=8)")


def comprehensive_benchmark():
    """
    Runs a comprehensive benchmark of all index types
    """
    print(f"\n" + "=" * 60)
    print("🏁 COMPREHENSIVE INDEX BENCHMARK")
    print("=" * 60)
    
    benchmarker = IndexBenchmarker()
    docs, embeddings = benchmarker.create_corpus("medium")
    queries = benchmarker.create_test_queries()[:5]  # Use 5 queries
    
    d = embeddings.shape[1]
    n = len(embeddings)
    
    print(f"Benchmark setup:")
    print(f"  Corpus size: {n} vectors")
    print(f"  Dimensions: {d}")
    print(f"  Test queries: {len(queries)}")
    
    # Create all index types
    indexes = {}
    
    # Flat (baseline)
    indexes["Flat"] = faiss.IndexFlatIP(d)
    indexes["Flat"].add(embeddings)
    
    # HNSW
    hnsw = faiss.IndexHNSWFlat(d, 32)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    hnsw.add(embeddings)
    indexes["HNSW"] = hnsw
    
    # IVF
    nlist = 64
    quantizer = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(embeddings)
    ivf.add(embeddings)
    ivf.nprobe = 8
    indexes["IVF"] = ivf
    
    # IVFPQ
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, 16, 8)
    ivfpq.train(embeddings)
    ivfpq.add(embeddings)
    ivfpq.nprobe = 8
    indexes["IVFPQ"] = ivfpq
    
    # Benchmark all indexes
    results = {}
    
    print(f"\nBenchmarking...")
    for name, index in indexes.items():
        print(f"  Testing {name}...")
        
        latencies = []
        recalls = []
        
        for query in queries:
            query_emb = benchmarker.model.encode([query], normalize_embeddings=True)
            
            # Get ground truth
            if name != "Flat":
                gt_distances, gt_indices = indexes["Flat"].search(query_emb, 10)
                gt_set = set(gt_indices[0])
            
            # Measure latency
            start_time = time.time()
            distances, indices = index.search(query_emb, 10)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            # Calculate recall
            if name != "Flat":
                retrieved_set = set(indices[0])
                recall = len(retrieved_set.intersection(gt_set)) / len(gt_set)
                recalls.append(recall)
            else:
                recalls.append(1.0)  # Perfect recall
        
        results[name] = {
            "latency": np.mean(latencies),
            "recall": np.mean(recalls),
            "std_latency": np.std(latencies),
            "std_recall": np.std(recalls)
        }
    
    # Display results
    print(f"\nBenchmark Results:")
    print(f"{'Index':<10} {'Latency (ms)':<15} {'Recall@10':<15} {'Efficiency'}")
    print("-" * 60)
    
    for name, result in results.items():
        lat = result["latency"]
        rec = result["recall"]
        efficiency = rec / (lat + 1)  # Simple efficiency metric
        
        print(f"{name:<10} {lat:.2f} ± {result['std_latency']:.2f}"
              f"    {rec:.3f} ± {result['std_recall']:.3f}"
              f"     {efficiency:.3f}")
    
    # Create visualization
    create_benchmark_plot(results)
    
    return results


def create_benchmark_plot(results):
    """
    Create a recall vs latency plot for all indexes
    """
    names = list(results.keys())
    latencies = [results[name]["latency"] for name in names]
    recalls = [results[name]["recall"] for name in names]
    
    plt.figure(figsize=(10, 6))
    
    # Different colors and markers for each index
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, name in enumerate(names):
        plt.scatter(latencies[i], recalls[i], 
                   c=colors[i], marker=markers[i], s=100, 
                   label=name, alpha=0.7)
        
        # Annotate points
        plt.annotate(name, (latencies[i], recalls[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Latency (ms)')
    plt.ylabel('Recall@10')
    plt.title('Index Performance: Recall vs Latency Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add efficiency frontiers
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% recall target')
    plt.axvline(x=10, color='blue', linestyle='--', alpha=0.5, label='10ms latency target')
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey insights from benchmark:")
    print("• Flat: Perfect recall but highest latency")
    print("• HNSW: Best balance for most use cases") 
    print("• IVF: Good speed, slight recall drop")
    print("• IVFPQ: Fastest but lowest recall")


def main():
    """
    Run all indexing structure demonstrations
    """
    print("🔍 BLOG 2.2 - SECTION 3: INDEXING STRUCTURES (ANN)")
    print("This covers all ANN indexing concepts from the blog post")
    print()
    
    # 1. Flat (Exact) baseline
    embeddings, queries = demonstrate_flat_index()
    
    # 2. HNSW demonstration
    demonstrate_hnsw_index()
    
    # 3. IVF demonstration
    demonstrate_ivf_index()
    
    # 4. PQ compression
    demonstrate_pq_compression()
    
    # 5. Index selection guide
    index_selection_guide()
    
    # 6. Comprehensive benchmark
    benchmark_results = comprehensive_benchmark()
    
    print("\n" + "=" * 60)
    print("✅ INDEXING STRUCTURES COMPLETE")
    print("=" * 60)
    print("Key concepts demonstrated:")
    print("• Flat: Perfect recall baseline, O(N) search")
    print("• HNSW: Graph-based, excellent recall-latency balance")
    print("• IVF: Clustering-based, memory efficient")
    print("• PQ: Compression for massive scale")
    print("• Selection depends on size, latency, memory constraints")
    print()
    print("👉 Rule of thumb: Start Flat → try HNSW → scale with IVF/PQ")
    print()
    print("Next: Run 04_vector_databases.py to explore production vector DBs")


if __name__ == "__main__":
    main()