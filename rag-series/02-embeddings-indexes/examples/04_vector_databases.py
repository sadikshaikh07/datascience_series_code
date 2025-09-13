#!/usr/bin/env python3
"""
Blog 2.2 - Section 4: Vector Databases
======================================

This script demonstrates production-ready vector databases:
- Chroma (developer-friendly, local)
- Qdrant (production-ready, scalable)  
- Feature comparisons and use cases
- Metadata filtering and hybrid search
- When to move beyond FAISS libraries

Covers blog sections:
- 4️⃣ Vector Databases: When a Library Isn't Enough
- Production features: persistence, filtering, hybrid search
- Database comparison and selection
"""

import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
import time
import json
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def create_sample_policy_data():
    """
    Create sample HR policy data with rich metadata for demonstration
    """
    policies = [
        {
            "id": "hr_mat_001",
            "title": "Maternity Leave Policy", 
            "content": "Our company provides 26 weeks of paid maternity leave with full benefits. Leave can begin up to 4 weeks before due date and can be extended with medical approval.",
            "department": "HR",
            "policy_type": "leave",
            "year": 2024,
            "priority": "high",
            "applies_to": ["full-time", "part-time"],
            "last_updated": "2024-01-15"
        },
        {
            "id": "hr_pat_002", 
            "title": "Paternity Leave Policy",
            "content": "Employees are eligible for 2 weeks of paid paternity leave immediately following birth or adoption. Additional unpaid leave may be requested with manager approval.",
            "department": "HR",
            "policy_type": "leave", 
            "year": 2024,
            "priority": "high",
            "applies_to": ["full-time"],
            "last_updated": "2024-01-15"
        },
        {
            "id": "hr_vac_003",
            "title": "Vacation Time Policy",
            "content": "Employees receive 15 vacation days in year one, 20 days after 2 years, and 25 days after 5 years. Vacation must be approved in advance.",
            "department": "HR", 
            "policy_type": "vacation",
            "year": 2024,
            "priority": "medium",
            "applies_to": ["full-time", "part-time"],
            "last_updated": "2024-02-01"
        },
        {
            "id": "it_sec_001",
            "title": "Laptop Security Policy",
            "content": "All company laptops must be encrypted with BitLocker and have automatic screen locks after 10 minutes of inactivity.",
            "department": "IT",
            "policy_type": "security",
            "year": 2024,
            "priority": "critical",
            "applies_to": ["all"],
            "last_updated": "2024-03-01"
        },
        {
            "id": "it_rem_002",
            "title": "Remote Work IT Requirements", 
            "content": "Remote workers must use company VPN, have reliable internet (>25 Mbps), and secure home office setup with locked filing.",
            "department": "IT",
            "policy_type": "remote_work",
            "year": 2024,
            "priority": "high",
            "applies_to": ["full-time"],
            "last_updated": "2024-01-20"
        },
        {
            "id": "fin_exp_001",
            "title": "Expense Reporting Policy",
            "content": "All expenses must be submitted within 30 days with receipts. Travel expenses over $100 require pre-approval from manager.",
            "department": "Finance",
            "policy_type": "expense",
            "year": 2024, 
            "priority": "medium",
            "applies_to": ["all"],
            "last_updated": "2024-02-15"
        },
        {
            "id": "hr_hea_004",
            "title": "Health Insurance Benefits",
            "content": "Company provides comprehensive health insurance covering medical, dental, and vision. Employees pay 20% of premiums, company covers 80%.",
            "department": "HR",
            "policy_type": "benefits",
            "year": 2024,
            "priority": "high", 
            "applies_to": ["full-time"],
            "last_updated": "2024-01-01"
        },
        {
            "id": "hr_rem_005",
            "title": "Remote Work Policy",
            "content": "Employees may work remotely up to 2 days per week with manager approval. Core hours 9 AM to 3 PM must be maintained.",
            "department": "HR",
            "policy_type": "remote_work", 
            "year": 2024,
            "priority": "medium",
            "applies_to": ["full-time"],
            "last_updated": "2024-01-10"
        }
    ]
    
    return policies


def demonstrate_chroma_basics():
    """
    Demonstrates Chroma vector database features
    """
    print("=" * 60)
    print("📚 CHROMA VECTOR DATABASE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize Chroma client (local, persistent)
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete collection if exists (for clean demo)
    try:
        client.delete_collection("company_policies")
    except:
        pass
    
    # Create collection with metadata
    collection = client.create_collection(
        name="company_policies",
        metadata={
            "description": "Company policy documents with advanced filtering",
            "created_by": "RAG_demo",
            "version": "1.0"
        }
    )
    
    print(f"Created collection: {collection.name}")
    print(f"Collection metadata: {collection.metadata}")
    
    # Load sample data
    policies = create_sample_policy_data()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"\nAdding {len(policies)} policies to collection...")
    
    # Add documents to collection
    documents = []
    metadatas = []
    ids = []
    
    for policy in policies:
        # Combine title and content for embedding
        full_text = f"{policy['title']}: {policy['content']}"
        documents.append(full_text)
        
        # Metadata (exclude content to avoid duplication)
        metadata = {k: v for k, v in policy.items() if k != 'content'}
        
        # Convert list values to strings for Chroma compatibility
        if 'applies_to' in metadata and isinstance(metadata['applies_to'], list):
            metadata['applies_to'] = ', '.join(metadata['applies_to'])
        
        metadatas.append(metadata)
        
        ids.append(policy['id'])
    
    # Add to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Added {collection.count()} documents")
    
    # Basic queries
    demonstrate_chroma_queries(collection)
    
    # Advanced filtering
    demonstrate_chroma_filtering(collection)
    
    return collection


def demonstrate_chroma_queries(collection):
    """
    Shows basic querying capabilities in Chroma
    """
    print(f"\n" + "=" * 50)
    print("CHROMA: BASIC QUERIES")
    print("=" * 50)
    
    queries = [
        "What is the maternity leave policy?",
        "How do I set up remote work?", 
        "What are the security requirements for laptops?",
        "How do I report business expenses?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        print("Results:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"  {i+1}. {metadata['title']} ({metadata['department']})")
            print(f"     Priority: {metadata['priority']}")
            print(f"     {doc[:100]}...")


def demonstrate_chroma_filtering(collection):
    """
    Demonstrates advanced filtering capabilities
    """
    print(f"\n" + "=" * 50)
    print("CHROMA: METADATA FILTERING")
    print("=" * 50)
    
    # Filter by department
    print("1. Filter by HR department:")
    results = collection.query(
        query_texts=["employee benefits and policies"],
        where={"department": "HR"},
        n_results=5
    )
    
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        print(f"  • {metadata['title']} - {metadata['policy_type']}")
    
    # Filter by priority
    print("\n2. Filter by critical priority:")
    results = collection.query(
        query_texts=["important security requirements"],
        where={"priority": "critical"},
        n_results=3
    )
    
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        print(f"  • {metadata['title']} - {metadata['department']}")
    
    # Complex filter (AND condition)
    print("\n3. Complex filter (HR + high priority):")
    results = collection.query(
        query_texts=["leave policies"],
        where={"$and": [{"department": "HR"}, {"priority": "high"}]},
        n_results=3
    )
    
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        print(f"  • {metadata['title']} - {metadata['policy_type']}")
    
    # Filter by exact string match (since Chroma doesn't support $contains for substring)
    print("\n4. Filter by exact applies_to field:")
    results = collection.query(
        query_texts=["what policies apply to me"],
        where={"applies_to": "full-time"},
        n_results=3
    )
    
    if results['documents'][0]:
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            print(f"  • {metadata['title']} - applies to: {metadata['applies_to']}")
    else:
        print("  • No exact matches found for 'full-time' - showing all applies_to values:")
        all_results = collection.query(
            query_texts=["policies"],
            n_results=8
        )
        for metadata in all_results['metadatas'][0]:
            print(f"    - {metadata['title']}: {metadata['applies_to']}")


def demonstrate_qdrant_setup():
    """
    Demonstrates Qdrant vector database setup and basic operations
    """
    print(f"\n" + "=" * 60)
    print("🚀 QDRANT VECTOR DATABASE DEMONSTRATION") 
    print("=" * 60)
    
    # Initialize Qdrant client (in-memory for demo)
    client = QdrantClient(":memory:")
    collection_name = "company_policies"
    
    print(f"Initializing Qdrant client...")
    print(f"Collection name: {collection_name}")
    
    # Create collection with vector configuration
    vector_size = 384  # all-MiniLM-L6-v2 dimensions
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    
    print(f"✅ Created collection with {vector_size}D vectors, COSINE distance")
    
    # Load and prepare data
    policies = create_sample_policy_data()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"\nPreparing {len(policies)} policies for insertion...")
    
    # Create points for insertion
    points = []
    
    for i, policy in enumerate(policies):
        # Create full text for embedding
        full_text = f"{policy['title']}: {policy['content']}"
        
        # Generate embedding
        vector = model.encode(full_text).tolist()
        
        # Create point with payload (metadata + content)
        point = PointStruct(
            id=i,
            vector=vector,
            payload={
                "document_id": policy['id'],
                "title": policy['title'],
                "content": policy['content'],
                "department": policy['department'], 
                "policy_type": policy['policy_type'],
                "year": policy['year'],
                "priority": policy['priority'],
                "applies_to": policy['applies_to'],
                "last_updated": policy['last_updated']
            }
        )
        points.append(point)
    
    # Insert points
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    collection_info = client.get_collection(collection_name)
    print(f"✅ Inserted {collection_info.points_count} points")
    
    # Basic queries
    demonstrate_qdrant_queries(client, collection_name, model)
    
    # Advanced filtering
    demonstrate_qdrant_filtering(client, collection_name, model)
    
    return client, collection_name


def demonstrate_qdrant_queries(client, collection_name, model):
    """
    Shows basic querying in Qdrant
    """
    print(f"\n" + "=" * 50)
    print("QDRANT: BASIC QUERIES")
    print("=" * 50)
    
    queries = [
        "What parental leave benefits are available?",
        "How do I work remotely?",
        "What are the IT security requirements?", 
        "How do I handle business expenses?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Generate query vector
        query_vector = model.encode(query).tolist()
        
        # Search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3,
            with_payload=True
        )
        
        print("Results:")
        for i, result in enumerate(search_results):
            payload = result.payload
            score = result.score
            print(f"  {i+1}. {payload['title']} (Score: {score:.3f})")
            print(f"     Dept: {payload['department']} | Type: {payload['policy_type']}")
            print(f"     {payload['content'][:80]}...")


def demonstrate_qdrant_filtering(client, collection_name, model):
    """
    Demonstrates advanced filtering in Qdrant
    """
    print(f"\n" + "=" * 50)
    print("QDRANT: ADVANCED FILTERING")
    print("=" * 50)
    
    # 1. Filter by single field
    print("1. Filter by HR department:")
    query_vector = model.encode("employee leave policies").tolist()
    
    hr_filter = Filter(
        must=[FieldCondition(key="department", match=MatchValue(value="HR"))]
    )
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=hr_filter,
        limit=3,
        with_payload=True
    )
    
    for result in results:
        payload = result.payload
        print(f"  • {payload['title']} - {payload['policy_type']}")
    
    # 2. Filter by multiple conditions (AND)
    print("\n2. Filter by HR + high priority:")
    
    hr_high_filter = Filter(
        must=[
            FieldCondition(key="department", match=MatchValue(value="HR")),
            FieldCondition(key="priority", match=MatchValue(value="high"))
        ]
    )
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=hr_high_filter,
        limit=3,
        with_payload=True
    )
    
    for result in results:
        payload = result.payload
        print(f"  • {payload['title']} - Priority: {payload['priority']}")
    
    # 3. Filter by year range
    print("\n3. Filter by year and policy type:")
    
    leave_2024_filter = Filter(
        must=[
            FieldCondition(key="policy_type", match=MatchValue(value="leave")),
            FieldCondition(key="year", match=MatchValue(value=2024))
        ]
    )
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=leave_2024_filter,
        limit=5,
        with_payload=True
    )
    
    for result in results:
        payload = result.payload
        print(f"  • {payload['title']} - Updated: {payload['last_updated']}")
    
    # 4. Complex filter with OR condition
    print("\n4. Complex filter (IT or Finance departments):")
    
    it_or_finance_filter = Filter(
        should=[  # OR condition
            FieldCondition(key="department", match=MatchValue(value="IT")),
            FieldCondition(key="department", match=MatchValue(value="Finance"))
        ]
    )
    
    results = client.search(
        collection_name=collection_name, 
        query_vector=model.encode("technology and financial policies").tolist(),
        query_filter=it_or_finance_filter,
        limit=4,
        with_payload=True
    )
    
    for result in results:
        payload = result.payload
        print(f"  • {payload['title']} - {payload['department']}")


def compare_vector_databases():
    """
    Compares features and performance of different vector databases
    """
    print(f"\n" + "=" * 60)
    print("⚔️ VECTOR DATABASE COMPARISON")
    print("=" * 60)
    
    # Feature comparison matrix
    comparison = {
        "Feature": [
            "Ease of Setup",
            "Local Development", 
            "Production Ready",
            "Metadata Filtering",
            "Hybrid Search",
            "Persistence",
            "Clustering/Sharding",
            "REST API",
            "Language Support",
            "Memory Usage",
            "Query Performance",
            "Scalability",
            "Community Support",
            "Enterprise Features"
        ],
        "Chroma": [
            "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐",
            "⭐⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐",
            "⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐"
        ],
        "Qdrant": [
            "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐",
            "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐",
            "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐"
        ],
        "FAISS": [
            "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐", "⭐",
            "⭐", "⭐", "⭐", "⭐⭐⭐", "⭐⭐⭐⭐⭐",
            "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐"
        ]
    }
    
    print("Feature Comparison Matrix:")
    print(f"{'Feature':<25} {'Chroma':<15} {'Qdrant':<15} {'FAISS':<10}")
    print("-" * 70)
    
    for i, feature in enumerate(comparison["Feature"]):
        chroma_score = comparison["Chroma"][i]
        qdrant_score = comparison["Qdrant"][i]
        faiss_score = comparison["FAISS"][i]
        print(f"{feature:<25} {chroma_score:<15} {qdrant_score:<15} {faiss_score:<10}")
    
    # Use case recommendations
    print(f"\n" + "=" * 40)
    print("USE CASE RECOMMENDATIONS:")
    print("=" * 40)
    
    use_cases = {
        "Prototyping & Development": {
            "Best Choice": "Chroma",
            "Why": "Simple setup, great for local development and testing"
        },
        "Small-Medium Production (< 10M vectors)": {
            "Best Choice": "Qdrant",
            "Why": "Production-ready with excellent performance and features"
        },
        "Large Scale Production (> 10M vectors)": {
            "Best Choice": "Qdrant + FAISS",
            "Why": "Qdrant for features + FAISS for maximum performance"
        },
        "Maximum Performance & Custom Control": {
            "Best Choice": "FAISS",
            "Why": "Fastest search, full control, but requires more engineering"
        },
        "Enterprise with Complex Requirements": {
            "Best Choice": "Qdrant or Weaviate",
            "Why": "Advanced features, security, multi-tenancy, support"
        }
    }
    
    for use_case, recommendation in use_cases.items():
        print(f"\n{use_case}:")
        print(f"  → {recommendation['Best Choice']}")
        print(f"  Why: {recommendation['Why']}")


def demonstrate_production_features():
    """
    Shows production features that vector databases provide over raw FAISS
    """
    print(f"\n" + "=" * 60)
    print("🏭 PRODUCTION FEATURES DEMONSTRATION")
    print("=" * 60)
    
    features = {
        "Persistence": {
            "Problem": "FAISS indexes lost on restart", 
            "Solution": "Automatic persistence to disk",
            "Example": "Database survives server restarts, no data loss"
        },
        "Metadata Filtering": {
            "Problem": "FAISS only does vector similarity",
            "Solution": "Rich metadata filtering before/during search",
            "Example": "Find similar docs WHERE department='HR' AND year=2024"
        },
        "Hybrid Search": {
            "Problem": "Vector-only search misses exact keyword matches",
            "Solution": "Combine BM25 keyword + vector semantic search", 
            "Example": "Find 'maternity leave' (exact) + similar concepts"
        },
        "Multi-tenancy": {
            "Problem": "Single FAISS index for all users/tenants",
            "Solution": "Isolated collections per tenant with access control",
            "Example": "Company A can't see Company B's documents"
        },
        "Horizontal Scaling": {
            "Problem": "Single machine memory/compute limits",
            "Solution": "Distribute across multiple nodes with sharding",
            "Example": "Scale from 1M to 1B+ vectors across cluster"
        },
        "REST APIs": {
            "Problem": "FAISS requires Python/C++ integration", 
            "Solution": "Language-agnostic HTTP APIs",
            "Example": "Query from JavaScript, Go, Java, etc."
        },
        "Monitoring & Observability": {
            "Problem": "No built-in metrics or monitoring",
            "Solution": "Built-in metrics, logging, health checks",
            "Example": "Track query latency, memory usage, index health"
        },
        "Backup & Recovery": {
            "Problem": "Manual index backup and restoration",
            "Solution": "Automated backup/restore with point-in-time recovery", 
            "Example": "Restore database to any point in last 30 days"
        }
    }
    
    print("Production Feature Analysis:")
    print("=" * 40)
    
    for feature, details in features.items():
        print(f"\n🔧 {feature}:")
        print(f"  Problem: {details['Problem']}")
        print(f"  Solution: {details['Solution']}")
        print(f"  Example: {details['Example']}")
    
    print(f"\n" + "=" * 40)
    print("When to Move Beyond FAISS:")
    print("=" * 40)
    
    migration_triggers = [
        "Need persistence across restarts",
        "Require metadata/attribute filtering", 
        "Want hybrid keyword + semantic search",
        "Multiple users/tenants need isolation",
        "Need to scale beyond single machine",
        "Want REST APIs for non-Python clients",
        "Require production monitoring/alerting",
        "Need automated backup and recovery"
    ]
    
    for trigger in migration_triggers:
        print(f"  • {trigger}")
    
    print(f"\n💡 Rule of thumb: Start with FAISS for performance testing,")
    print(f"   move to vector DB when you need production features.")


def benchmark_vector_databases():
    """
    Simple benchmark comparing Chroma vs Qdrant performance
    """
    print(f"\n" + "=" * 60)
    print("⚡ VECTOR DATABASE PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create test data
    policies = create_sample_policy_data() * 10  # 80 documents
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    test_queries = [
        "What is the leave policy for new parents?",
        "How do I set up remote work arrangements?",
        "What are the IT security requirements?"
    ]
    
    print(f"Benchmark setup:")
    print(f"  Documents: {len(policies)}")
    print(f"  Test queries: {len(test_queries)}")
    print(f"  Metric: Average query latency")
    
    results = {}
    
    # Benchmark Chroma
    print(f"\n🔄 Benchmarking Chroma...")
    
    # Setup Chroma
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("benchmark")
    except:
        pass
    
    chroma_collection = chroma_client.create_collection("benchmark")
    
    # Add data
    documents = []
    metadatas = []
    ids = []
    
    for i, policy in enumerate(policies):
        full_text = f"{policy['title']}: {policy['content']}"
        documents.append(full_text)
        
        # Fix metadata for Chroma compatibility
        metadata = {k: v for k, v in policy.items() if k != 'content'}
        if 'applies_to' in metadata and isinstance(metadata['applies_to'], list):
            metadata['applies_to'] = ', '.join(metadata['applies_to'])
        
        metadatas.append(metadata)
        ids.append(f"doc_{i}")
    
    chroma_collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    # Benchmark queries
    chroma_times = []
    for query in test_queries:
        start = time.time()
        results_chroma = chroma_collection.query(query_texts=[query], n_results=5)
        chroma_times.append((time.time() - start) * 1000)
    
    results["Chroma"] = {
        "avg_latency": np.mean(chroma_times),
        "std_latency": np.std(chroma_times),
        "setup_time": "Quick (embedded)"
    }
    
    # Benchmark Qdrant
    print(f"🔄 Benchmarking Qdrant...")
    
    # Setup Qdrant
    qdrant_client = QdrantClient(":memory:")
    collection_name = "benchmark"
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Prepare points
    points = []
    for i, policy in enumerate(policies):
        full_text = f"{policy['title']}: {policy['content']}"
        vector = model.encode(full_text).tolist()
        
        point = PointStruct(
            id=i,
            vector=vector,
            payload=policy
        )
        points.append(point)
    
    qdrant_client.upsert(collection_name=collection_name, points=points)
    
    # Benchmark queries
    qdrant_times = []
    for query in test_queries:
        query_vector = model.encode(query).tolist()
        
        start = time.time()
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5,
            with_payload=True
        )
        qdrant_times.append((time.time() - start) * 1000)
    
    results["Qdrant"] = {
        "avg_latency": np.mean(qdrant_times),
        "std_latency": np.std(qdrant_times), 
        "setup_time": "Medium (client-server)"
    }
    
    # Display results
    print(f"\nBenchmark Results:")
    print(f"{'Database':<10} {'Avg Latency (ms)':<16} {'Std Dev':<10} {'Setup'}")
    print("-" * 50)
    
    for db_name, result in results.items():
        avg_lat = result["avg_latency"]
        std_lat = result["std_latency"]
        setup = result["setup_time"]
        print(f"{db_name:<10} {avg_lat:<16.2f} {std_lat:<10.2f} {setup}")
    
    print(f"\nKey insights:")
    print(f"• Both databases perform well for small-medium workloads")
    print(f"• Qdrant typically faster for pure vector search")
    print(f"• Chroma simpler for development and prototyping")
    print(f"• Performance differences more significant at scale")
    
    return results


def main():
    """
    Run all vector database demonstrations
    """
    print("📊 BLOG 2.2 - SECTION 4: VECTOR DATABASES")
    print("This covers production vector database concepts from the blog post")
    print()
    
    # 1. Chroma demonstration
    chroma_collection = demonstrate_chroma_basics()
    
    # 2. Qdrant demonstration  
    qdrant_client, qdrant_collection = demonstrate_qdrant_setup()
    
    # 3. Database comparison
    compare_vector_databases()
    
    # 4. Production features
    demonstrate_production_features()
    
    # 5. Performance benchmark
    benchmark_results = benchmark_vector_databases()
    
    print("\n" + "=" * 60)
    print("✅ VECTOR DATABASES COMPLETE")
    print("=" * 60)
    print("Key concepts demonstrated:")
    print("• Chroma: Developer-friendly, embedded database")
    print("• Qdrant: Production-ready, high-performance database") 
    print("• Production features: persistence, filtering, APIs")
    print("• Migration path: FAISS → Vector DB when complexity grows")
    print("• Feature vs performance trade-offs")
    print()
    print("👉 Decision: Use Chroma for dev, Qdrant for production")
    print()
    print("Next: Run 05_retrieval_mechanics.py for end-to-end RAG pipeline")


if __name__ == "__main__":
    main()