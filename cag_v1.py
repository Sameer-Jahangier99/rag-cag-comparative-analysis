import time
import json
import torch
import numpy as np
import psutil
import hashlib
import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import OrderedDict, defaultdict

# Load environment variables
load_dotenv()

# Core imports
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import uuid

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False
    print("Warning: NVML not available. GPU metrics will not be collected.")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CAGConfig:
    """Configuration for CAG-Enhanced RAG System"""
    # Model settings
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    use_groq_api: bool = True
    
    # Generation settings
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Retrieval settings
    use_bm25: bool = True
    use_dense: bool = True
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Cache settings
    prompt_cache_size: int = 10000
    semantic_cache_size: int = 10000
    similarity_threshold: float = 0.85
    ttl_hours: Optional[int] = 24
    kv_cache_enabled: bool = True
    eviction_policy: str = "hybrid"
    recreate_collection: bool = False
    
    # Qdrant Cloud settings (loaded from .env)
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = "cag_rag_docs"
    
    # Separate Qdrant Cloud for CAG (loaded from .env)
    qdrant_url_cag: str = ""
    qdrant_api_key_cag: str = ""
    collection_name_cag: str = "cag_rag_docs_cag"
    
    # Groq API settings
    groq_api_key: str = ""
    
    def __post_init__(self):
        """Load credentials from environment if not provided"""
        # Default Qdrant (for RAG)
        if not self.qdrant_url:
            self.qdrant_url = os.getenv("QDRANT_URL", "https://your-qdrant-instance.qdrant.io")
        if not self.qdrant_api_key:
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "your-api-key-here")
        
        # Separate Qdrant for CAG
        if not self.qdrant_url_cag:
            self.qdrant_url_cag = os.getenv("QDRANT_URL_1", self.qdrant_url)
        if not self.qdrant_api_key_cag:
            self.qdrant_api_key_cag = os.getenv("QDRANT_API_KEY_1", self.qdrant_api_key)
        if not self.collection_name_cag:
            self.collection_name_cag = os.getenv("QDRANT_COLLECTION_1", "cag_rag_docs_cag")
        
        # Groq API
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY", "")
    
    def save(self, path: str):
        """Save configuration to JSON (without credentials)"""
        config_dict = asdict(self)
        # Don't save sensitive data
        config_dict['qdrant_api_key'] = "***REDACTED***"
        config_dict['qdrant_api_key_cag'] = "***REDACTED***"
        config_dict['groq_api_key'] = "***REDACTED***"
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        # Reload credentials from environment
        data['qdrant_url'] = os.getenv("QDRANT_URL", data.get('qdrant_url', ''))
        data['qdrant_api_key'] = os.getenv("QDRANT_API_KEY", data.get('qdrant_api_key', ''))
        data['qdrant_url_cag'] = os.getenv("QDRANT_URL_1", data.get('qdrant_url_cag', ''))
        data['qdrant_api_key_cag'] = os.getenv("QDRANT_API_KEY_1", data.get('qdrant_api_key_cag', ''))
        data['collection_name_cag'] = os.getenv("QDRANT_COLLECTION_1", data.get('collection_name_cag', ''))
        data['groq_api_key'] = os.getenv("GROQ_API_KEY", data.get('groq_api_key', ''))
        return cls(**data)

# ============================================================================
# CACHE IMPLEMENTATIONS
# ============================================================================

class CacheEntry:
    """Single cache entry with metadata"""
    def __init__(self, key: str, value: Any, embedding: np.ndarray = None):
        self.key = key
        self.value = value
        self.embedding = embedding
        self.timestamp = time.time()
        self.hit_count = 0
        self.last_access = time.time()
    
    def is_expired(self, ttl_seconds: Optional[int]) -> bool:
        if ttl_seconds is None:
            return False
        return (time.time() - self.timestamp) > ttl_seconds

class PromptCache:
    """Exact prompt matching cache with LRU eviction"""
    def __init__(self, max_size: int, ttl_hours: Optional[int] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours else None
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.total_lookups = 0
    
    def _hash_key(self, prompt: str) -> str:
        """Generate hash key from prompt"""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[Any]:
        """Get cached value for exact prompt match"""
        self.total_lookups += 1
        key = self._hash_key(prompt)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired(self.ttl_seconds):
                del self.cache[key]
                self.misses += 1
                return None
            
            # Update access metadata
            entry.hit_count += 1
            entry.last_access = time.time()
            self.cache.move_to_end(key)
            self.hits += 1
            return entry.value
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, value: Any):
        """Store value in cache"""
        key = self._hash_key(prompt)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)  # LRU eviction
        
        self.cache[key] = CacheEntry(key, value)
        self.cache.move_to_end(key)
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_lookups = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            "type": "prompt_cache",
            "hits": self.hits,
            "misses": self.misses,
            "total_lookups": self.total_lookups,
            "hit_rate": self.hits / total if total > 0 else 0,
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0
        }

class SemanticCache:
    """Semantic similarity-based cache with fuzzy matching"""
    def __init__(self, max_size: int, similarity_threshold: float, 
                 ttl_hours: Optional[int] = None):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours else None
        self.entries: List[CacheEntry] = []
        self.hits = 0
        self.misses = 0
        self.total_lookups = 0
    
    def get(self, query_embedding: np.ndarray) -> Optional[Any]:
        """Get cached value for similar query"""
        self.total_lookups += 1
        
        if not self.entries:
            self.misses += 1
            return None
        
        # Remove expired entries
        self.entries = [e for e in self.entries 
                       if not e.is_expired(self.ttl_seconds)]
        
        if not self.entries:
            self.misses += 1
            return None
        
        # Compute similarities
        embeddings = np.array([e.embedding for e in self.entries])
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            embeddings
        )[0]
        
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        
        # Check if similarity exceeds threshold
        if max_sim >= self.similarity_threshold:
            entry = self.entries[max_idx]
            entry.hit_count += 1
            entry.last_access = time.time()
            self.hits += 1
            return entry.value
        
        self.misses += 1
        return None
    
    def set(self, query_embedding: np.ndarray, value: Any):
        """Store value with embedding"""
        # Evict if at capacity (hybrid LRU + LFU)
        if len(self.entries) >= self.max_size:
            # Sort by hit count (ascending) then last access (ascending)
            self.entries.sort(key=lambda e: (e.hit_count, e.last_access))
            self.entries.pop(0)
        
        entry = CacheEntry(
            key=f"semantic_{len(self.entries)}",
            value=value,
            embedding=query_embedding
        )
        self.entries.append(entry)
    
    def clear(self):
        """Clear all cache entries"""
        self.entries.clear()
        self.hits = 0
        self.misses = 0
        self.total_lookups = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            "type": "semantic_cache",
            "hits": self.hits,
            "misses": self.misses,
            "total_lookups": self.total_lookups,
            "hit_rate": self.hits / total if total > 0 else 0,
            "current_size": len(self.entries),
            "max_size": self.max_size,
            "utilization": len(self.entries) / self.max_size if self.max_size > 0 else 0,
            "similarity_threshold": self.similarity_threshold
        }

class CAGLayer:
    """Multi-tier Cache-Augmented Generation layer"""
    def __init__(self, config: CAGConfig):
        self.config = config
        self.prompt_cache = PromptCache(
            config.prompt_cache_size,
            config.ttl_hours
        )
        self.semantic_cache = SemanticCache(
            config.semantic_cache_size,
            config.similarity_threshold,
            config.ttl_hours
        )
        self.kv_cache_enabled = config.kv_cache_enabled
        
        print("CAG Layer initialized:")
        print(f"  - Prompt Cache: {config.prompt_cache_size} entries")
        print(f"  - Semantic Cache: {config.semantic_cache_size} entries (τ={config.similarity_threshold})")
        print(f"  - KV Cache: {'Enabled' if config.kv_cache_enabled else 'Disabled'}")
    
    def get_all_stats(self) -> Dict:
        """Get statistics from all cache layers"""
        prompt_stats = self.prompt_cache.get_stats()
        semantic_stats = self.semantic_cache.get_stats()
        
        # Combined statistics
        total_hits = prompt_stats['hits'] + semantic_stats['hits']
        total_lookups = prompt_stats['total_lookups'] + semantic_stats['total_lookups']
        
        return {
            "prompt_cache": prompt_stats,
            "semantic_cache": semantic_stats,
            "combined": {
                "total_hits": total_hits,
                "total_lookups": total_lookups,
                "overall_hit_rate": total_hits / total_lookups if total_lookups > 0 else 0
            }
        }
    
    def clear_all(self):
        """Clear all caches"""
        self.prompt_cache.clear()
        self.semantic_cache.clear()

# ============================================================================
# METRICS MONITORING
# ============================================================================

@dataclass
class CAGQueryMetrics:
    """Extended metrics for CAG system"""
    query_id: int
    query: str
    answer: str
    
    # Timing metrics
    total_latency_ms: float
    cache_lookup_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    
    # Cache metrics
    cache_hit: bool
    cache_type: Optional[str]  # "prompt", "semantic", None
    prompt_cache_checked: bool
    semantic_cache_checked: bool
    
    # Token metrics
    prompt_tokens: int
    context_tokens: int
    generation_tokens: int
    total_tokens: int
    tokens_saved: int  # Compared to no-cache baseline
    
    # Memory metrics
    cpu_memory_mb: float
    gpu_memory_mb: float
    
    # Retrieval metrics
    num_retrieved_docs: int
    retrieved_contexts: Optional[List[str]]
    
    # Quality metrics
    exact_match: Optional[float] = None
    f1_score: Optional[float] = None
    
    timestamp: float = time.time()

class CAGMetricsMonitor:
    """Monitor and track all CAG system metrics"""
    
    def __init__(self):
        self.query_metrics: List[CAGQueryMetrics] = []
        self.process = psutil.Process()
        
        if NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
        else:
            self.gpu_handle = None
        
        self.query_counter = 0
        self.start_time = time.time()
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current CPU and GPU memory usage in MB"""
        cpu_mem = self.process.memory_info().rss / (1024 ** 2)
        
        gpu_mem = 0.0
        if NVML_AVAILABLE and self.gpu_handle:
            try:
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem = gpu_info.used / (1024 ** 2)
            except:
                pass
        
        return cpu_mem, gpu_mem
    
    def record_query(self, metrics: CAGQueryMetrics):
        """Record metrics for a query"""
        self.query_metrics.append(metrics)
        self.query_counter += 1
    
    def get_summary_statistics(self) -> Dict:
        """Calculate comprehensive summary statistics"""
        if not self.query_metrics:
            return {}
        
        # Separate cache hits and misses
        cache_hits = [m for m in self.query_metrics if m.cache_hit]
        cache_misses = [m for m in self.query_metrics if not m.cache_hit]
        
        # Latency statistics
        all_latencies = [m.total_latency_ms for m in self.query_metrics]
        hit_latencies = [m.total_latency_ms for m in cache_hits]
        miss_latencies = [m.total_latency_ms for m in cache_misses]
        
        cache_lookup_latencies = [m.cache_lookup_latency_ms for m in self.query_metrics]
        retrieval_latencies = [m.retrieval_latency_ms for m in self.query_metrics if m.retrieval_latency_ms > 0]
        generation_latencies = [m.generation_latency_ms for m in self.query_metrics if m.generation_latency_ms > 0]
        
        # Token statistics
        all_tokens = [m.total_tokens for m in self.query_metrics]
        tokens_saved = [m.tokens_saved for m in self.query_metrics]
        
        # Cache statistics
        prompt_cache_hits = len([m for m in cache_hits if m.cache_type == "prompt"])
        semantic_cache_hits = len([m for m in cache_hits if m.cache_type == "semantic"])
        
        # Memory statistics
        cpu_memory = [m.cpu_memory_mb for m in self.query_metrics]
        gpu_memory = [m.gpu_memory_mb for m in self.query_metrics]
        
        return {
            "system_type": "CAG_Enhanced_RAG",
            "total_queries": len(self.query_metrics),
            "runtime_seconds": time.time() - self.start_time,
            
            # Cache performance
            "cache_performance": {
                "total_hits": len(cache_hits),
                "total_misses": len(cache_misses),
                "overall_hit_rate": len(cache_hits) / len(self.query_metrics),
                "prompt_cache_hits": prompt_cache_hits,
                "semantic_cache_hits": semantic_cache_hits,
                "cache_breakdown": {
                    "prompt": prompt_cache_hits / len(cache_hits) if cache_hits else 0,
                    "semantic": semantic_cache_hits / len(cache_hits) if cache_hits else 0
                }
            },
            
            # Latency metrics
            "latency": {
                "all_queries": {
                    "mean": np.mean(all_latencies),
                    "median": np.median(all_latencies),
                    "p50": np.percentile(all_latencies, 50),
                    "p95": np.percentile(all_latencies, 95),
                    "p99": np.percentile(all_latencies, 99),
                    "std": np.std(all_latencies),
                    "min": np.min(all_latencies),
                    "max": np.max(all_latencies)
                },
                "cache_hits": {
                    "mean": np.mean(hit_latencies) if hit_latencies else 0,
                    "median": np.median(hit_latencies) if hit_latencies else 0,
                    "p95": np.percentile(hit_latencies, 95) if hit_latencies else 0
                },
                "cache_misses": {
                    "mean": np.mean(miss_latencies) if miss_latencies else 0,
                    "median": np.median(miss_latencies) if miss_latencies else 0,
                    "p95": np.percentile(miss_latencies, 95) if miss_latencies else 0
                },
                "speedup_factor": (np.mean(miss_latencies) / np.mean(hit_latencies)) if hit_latencies and miss_latencies else 1.0,
                "cache_lookup": {
                    "mean": np.mean(cache_lookup_latencies),
                    "median": np.median(cache_lookup_latencies)
                },
                "retrieval": {
                    "mean": np.mean(retrieval_latencies) if retrieval_latencies else 0,
                    "median": np.median(retrieval_latencies) if retrieval_latencies else 0
                },
                "generation": {
                    "mean": np.mean(generation_latencies) if generation_latencies else 0,
                    "median": np.median(generation_latencies) if generation_latencies else 0
                }
            },
            
            # Token metrics
            "tokens": {
                "total": {
                    "sum": sum(all_tokens),
                    "mean": np.mean(all_tokens),
                    "median": np.median(all_tokens),
                    "std": np.std(all_tokens)
                },
                "saved": {
                    "total": sum(tokens_saved),
                    "mean": np.mean(tokens_saved),
                    "median": np.median(tokens_saved),
                    "percentage_saved": (sum(tokens_saved) / (sum(all_tokens) + sum(tokens_saved)) * 100) if (sum(all_tokens) + sum(tokens_saved)) > 0 else 0
                }
            },
            
            # Memory metrics
            "memory": {
                "cpu": {
                    "mean_mb": np.mean(cpu_memory),
                    "peak_mb": np.max(cpu_memory)
                },
                "gpu": {
                    "mean_mb": np.mean(gpu_memory),
                    "peak_mb": np.max(gpu_memory)
                }
            },
            
            # Throughput
            "throughput": {
                "queries_per_second": len(self.query_metrics) / (time.time() - self.start_time)
            }
        }
    
    def save_metrics(self, filepath: str):
        """Save all metrics to JSON file (NumPy-safe serialization)"""

        # ---- Converter to safely convert numpy types ----
        def to_python(o):
            if isinstance(o, dict):
                return {k: to_python(v) for k, v in o.items()}
            if isinstance(o, list):
                return [to_python(v) for v in o]
            if isinstance(o, tuple):
                return tuple(to_python(v) for v in o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        # ---- Build output JSON ----
        output = {
            "config": "CAG_Enhanced_RAG",
            "summary": to_python(self.get_summary_statistics()),
            "individual_queries": [
                to_python({
                    "query_id": m.query_id,
                    "query": m.query,
                    "answer": m.answer,
                    "cache_hit": m.cache_hit,
                    "cache_type": m.cache_type,
                    "total_latency_ms": m.total_latency_ms,
                    "cache_lookup_latency_ms": m.cache_lookup_latency_ms,
                    "retrieval_latency_ms": m.retrieval_latency_ms,
                    "generation_latency_ms": m.generation_latency_ms,
                    "total_tokens": m.total_tokens,
                    "tokens_saved": m.tokens_saved,
                    "cpu_memory_mb": m.cpu_memory_mb,
                    "gpu_memory_mb": m.gpu_memory_mb,
                    "timestamp": m.timestamp
                })
                for m in self.query_metrics
            ]
        }

        # ---- Write JSON safely ----
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nMetrics saved to: {filepath}")


# ============================================================================
# HYBRID RETRIEVER (BM25 + DENSE)
# ============================================================================

class HybridRetriever:
    """Hybrid retrieval with BM25 and Dense vectors using separate Qdrant Cloud"""
    
    def __init__(self, config: CAGConfig):
        self.config = config
        
        print(f"Initializing Hybrid Retriever...")
        print(f"  - BM25: {config.use_bm25}")
        print(f"  - Dense: {config.use_dense}")
        
        # Initialize embedding model
        if config.use_dense:
            print(f"Loading embedding model: {config.embedding_model}")
            self.embedding_model = SentenceTransformer(
                config.embedding_model,
                device=config.device
            )
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            self.embedding_model = None
            self.embedding_dim = None
        
        # Initialize Qdrant Cloud (CAG instance)
        if config.use_dense:
            print(f"Connecting to Qdrant Cloud (CAG)...")
            base_url_cag = config.qdrant_url_cag.replace(":6333", "")
            self.qdrant_client = QdrantClient(
                url=base_url_cag,
                api_key=config.qdrant_api_key_cag,
                timeout=60.0
            )
            self._init_collection()
        else:
            self.qdrant_client = None
        
        # BM25 components
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        
        print("Hybrid Retriever initialized successfully")
    
    def _init_collection(self):
        """Initialize Qdrant collection with user choice"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        collection_exists = self.config.collection_name_cag in collection_names
        
        if collection_exists:
            print(f"\n✓ Collection '{self.config.collection_name_cag}' already exists in CAG Qdrant")
            
            if self.config.recreate_collection:
                choice = "recreate"
            else:
                print("\nWhat would you like to do?")
                print("  1. Use existing collection (continue with current data)")
                print("  2. Recreate collection (delete and start fresh)")
                choice_input = input("\nEnter your choice (1 or 2): ").strip()
                
                if choice_input == "2":
                    choice = "recreate"
                else:
                    choice = "use"
            
            if choice == "recreate":
                print(f"Deleting existing collection: {self.config.collection_name_cag}")
                self.qdrant_client.delete_collection(
                    collection_name=self.config.collection_name_cag
                )
                print("Creating new collection...")
                self._create_collection()
            else:
                print(f"Using existing collection: {self.config.collection_name_cag}")
        else:
            print(f"Collection '{self.config.collection_name_cag}' does not exist in CAG Qdrant")
            print("Creating new collection...")
            self._create_collection()
    
    def _create_collection(self):
        """Create a new Qdrant collection"""
        self.qdrant_client.create_collection(
            collection_name=self.config.collection_name_cag,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        print(f"✓ Created Qdrant collection: {self.config.collection_name_cag}")
    
    def add_documents(self, documents: List[str]):
        """Add documents to the retriever (for BM25 and Dense)"""
        print(f"\nAdding {len(documents)} documents to CAG retriever...")
        self.documents.extend(documents)
        self.doc_ids.extend([str(i) for i in range(len(documents))])
        
        if self.config.use_bm25:
            self._init_bm25()
        
        if self.config.use_dense and self.embedding_model:
            self._index_dense_vectors(documents)
        
        print(f"✓ Added {len(documents)} documents to CAG retriever")
    
    def _init_bm25(self):
        """Initialize BM25 model"""
        print("Building BM25 index...")
        tokenized_docs = [doc.split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("✓ BM25 index built")
    
    def _index_dense_vectors(self, documents: List[str]):
        """Index documents with dense vectors in Qdrant Cloud"""
        print("Generating embeddings and indexing in Qdrant Cloud...")
        embeddings = self.embedding_model.encode(
            documents, 
            show_progress_bar=True, 
            batch_size=32
        )
        
        # Prepare points for Qdrant
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"text": doc, "doc_index": i}
                )
            )
        
        # Batch upsert to Qdrant Cloud
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:min(i + batch_size, len(points))]
            self.qdrant_client.upsert(
                collection_name=self.config.collection_name_cag,
                points=batch
            )
        
        print(f"✓ Indexed {len(points)} documents in Qdrant Cloud")
    
    def query(self, query: str) -> Dict:
        """Perform hybrid retrieval and return results with metrics"""
        start_time = time.perf_counter()
        
        # BM25 retrieval
        bm25_results = []
        if self.config.use_bm25 and self.bm25:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-(self.config.top_k * 2):][::-1]
            bm25_results = [(self.documents[idx], scores[idx]) for idx in top_indices if idx < len(self.documents)]
        
        # Dense retrieval
        dense_results = []
        if self.config.use_dense and self.embedding_model and self.qdrant_client:
            try:
                query_embedding = self.embedding_model.encode([query])[0]
                search_results = self.qdrant_client.search(
                    collection_name=self.config.collection_name_cag,
                    query_vector=query_embedding.tolist(),
                    limit=self.config.top_k * 2
                )
                dense_results = [(hit.payload['text'], hit.score) for hit in search_results]
            except Exception as e:
                print(f"  ⚠ Dense retrieval failed: {e}")
        
        # Combine results
        all_results = {}
        for text, score in bm25_results:
            all_results[text] = all_results.get(text, 0) + score * 0.5
        for text, score in dense_results:
            all_results[text] = all_results.get(text, 0) + score * 0.5
        
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        contexts = [text for text, _ in sorted_results[:self.config.top_k]]
        
        retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "retrieved_contexts": contexts,
            "retrieval_latency_ms": retrieval_time_ms,
            "total_tokens": sum([len(ctx.split()) for ctx in contexts]),
            "num_retrieved_docs": len(contexts)
        }

# ============================================================================
# LLM ENGINE (GROQ API)
# ============================================================================

class LLMEngine:
    """LLM generation engine using Groq Cloud API"""
    
    def __init__(self, config: CAGConfig):
        self.config = config
        
        print(f"\nInitializing LLM Engine...")
        print(f"Model: {config.llm_model}")
        print(f"API: Groq Cloud")
        
        # Initialize Groq client
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file. Please set it.")
        
        self.client = Groq(api_key=config.groq_api_key)
        
        print("LLM Engine initialized successfully with Groq API")
    
    def count_tokens(self, text: str) -> int:
        """Estimate tokens (Groq uses ~4 characters per token average)"""
        return len(text) // 4
    
    def generate(self, prompt: str) -> Tuple[str, float, int, int]:
        """Generate response using Groq API"""
        start_time = time.perf_counter()
        
        # Estimate prompt tokens
        prompt_tokens = self.count_tokens(prompt)
        
        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_new_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Get actual token usage from response
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                generation_tokens = response.usage.completion_tokens
            else:
                generation_tokens = self.count_tokens(answer)
        
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            answer = "Error generating response"
            generation_tokens = 0
        
        generation_time_ms = (time.perf_counter() - start_time) * 1000
        
        return answer, generation_time_ms, prompt_tokens, generation_tokens

# ============================================================================
# CAG ENHANCED RAG SYSTEM
# ============================================================================

class CAGEnhancedRAG:
    """End-to-end CAG-Enhanced RAG system"""
    
    def __init__(self, config: CAGConfig):
        self.config = config
        
        print("\nInitializing CAG-Enhanced RAG System...")
        
        # Initialize caches
        self.cag_layer = CAGLayer(config)
        
        # Initialize retriever
        self.retriever = HybridRetriever(config)
        
        # Initialize LLM engine
        self.llm_engine = LLMEngine(config)
        
        # Initialize metrics monitor
        self.metrics_monitor = CAGMetricsMonitor()
        
        self.query_counter = 0
    
    def add_documents(self, documents: List[str]):
        """Add documents to retriever and update caches"""
        self.retriever.add_documents(documents)
        print(f"✓ Added {len(documents)} documents to system caches")
    
    def query(self, query: str) -> Dict:
        """Perform query with caching, retrieval and generation"""
        query_start = time.perf_counter()
        cache_lookup_start = time.perf_counter()
        
        retrieval_time_ms = 0
        generation_time_ms = 0
        cache_hit = False
        cache_type = None
        answer = ""
        prompt_tokens = 0
        context_tokens = 0
        generation_tokens = 0
        retrieved_contexts = []
        num_retrieved_docs = 0
        tokens_saved = 0
        
        # Get memory BASELINE before this query
        cpu_mem_baseline, gpu_mem_baseline = self.metrics_monitor.get_memory_usage()
        
        # Step 1: Check Prompt Cache FIRST (exact match) - NO ENCODING NEEDED
        cached_result = self.cag_layer.prompt_cache.get(query)
        
        if cached_result and isinstance(cached_result, str):
            # PROMPT CACHE HIT - No memory overhead at all
            cache_hit = True
            cache_type = "prompt"
            answer = cached_result
            prompt_tokens = 0
            generation_tokens = 0
            context_tokens = 0
            tokens_saved = 500
            # Memory = baseline (no work done)
            cpu_mem = cpu_mem_baseline
            gpu_mem = gpu_mem_baseline
        
        # Step 2: ONLY check Semantic Cache if prompt cache misses
        # This avoids encoding queries that already hit prompt cache
        elif self.retriever.embedding_model:
            try:
                # NOW we encode - only if needed
                query_embedding = self.retriever.embedding_model.encode([query])[0]
                semantic_result = self.cag_layer.semantic_cache.get(query_embedding)
                
                if semantic_result and isinstance(semantic_result, str):
                    # SEMANTIC CACHE HIT
                    cache_hit = True
                    cache_type = "semantic"
                    answer = semantic_result
                    prompt_tokens = 0
                    generation_tokens = 0
                    context_tokens = 0
                    tokens_saved = 500
                    # Memory = baseline (minimal encoding work)
                    cpu_mem = cpu_mem_baseline
                    gpu_mem = gpu_mem_baseline
            except Exception as e:
                pass
        
        cache_lookup_time_ms = (time.perf_counter() - cache_lookup_start) * 1000
        
        # Step 3: Full pipeline on cache miss
        if not cache_hit:
            # Retrieval
            retrieval_start = time.perf_counter()
            retrieval_results = self.retriever.query(query)
            retrieval_time_ms = retrieval_results["retrieval_latency_ms"]
            retrieved_contexts = retrieval_results["retrieved_contexts"]
            num_retrieved_docs = retrieval_results["num_retrieved_docs"]
            
            # Get memory AFTER retrieval
            cpu_mem_after_retrieval, gpu_mem_after_retrieval = self.metrics_monitor.get_memory_usage()
            
            # Count tokens for context
            context_tokens = sum([self.llm_engine.count_tokens(ctx) for ctx in retrieved_contexts])
            
            # Build and generate answer
            prompt = self._build_prompt(query, retrieved_contexts)
            answer, generation_time_ms, prompt_tokens_actual, generation_tokens = self.llm_engine.generate(prompt)
            
            # For cache misses: count full tokens
            prompt_tokens = self.llm_engine.count_tokens(query)
            tokens_saved = 0
            
            # Get memory AFTER generation
            cpu_mem_after_generation, gpu_mem_after_generation = self.metrics_monitor.get_memory_usage()
            
            # MEMORY: Use average memory during operation
            cpu_mem_avg = (cpu_mem_after_retrieval + cpu_mem_after_generation) / 2
            gpu_mem_avg = (gpu_mem_after_retrieval + gpu_mem_after_generation) / 2
            cpu_mem = cpu_mem_avg
            gpu_mem = gpu_mem_avg
            
            # Cache the results for future hits
            self.cag_layer.prompt_cache.set(query, answer)
            
            if self.retriever.embedding_model:
                try:
                    # Use the embedding we might have created for semantic cache
                    query_embedding = self.retriever.embedding_model.encode([query])[0]
                    self.cag_layer.semantic_cache.set(query_embedding, answer)
                except:
                    pass
        
        total_latency_ms = (time.perf_counter() - query_start) * 1000
        
        # Calculate total tokens
        total_tokens = prompt_tokens + context_tokens + generation_tokens
        
        # Record metrics
        metrics = CAGQueryMetrics(
            query_id=self.query_counter,
            query=query,
            answer=answer,
            total_latency_ms=total_latency_ms,
            cache_lookup_latency_ms=cache_lookup_time_ms,
            retrieval_latency_ms=retrieval_time_ms,
            generation_latency_ms=generation_time_ms,
            cache_hit=cache_hit,
            cache_type=cache_type,
            prompt_cache_checked=True,
            semantic_cache_checked=(not cache_hit),  # Only checked if not prompt hit
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
            generation_tokens=generation_tokens,
            total_tokens=total_tokens,
            tokens_saved=tokens_saved,
            cpu_memory_mb=cpu_mem,
            gpu_memory_mb=gpu_mem,
            num_retrieved_docs=num_retrieved_docs,
            retrieved_contexts=retrieved_contexts
        )
        
        self.metrics_monitor.record_query(metrics)
        self.query_counter += 1
        
        return {
            "query": query,
            "answer": answer,
            "cache_hit": cache_hit,
            "cache_type": cache_type,
            "total_latency_ms": total_latency_ms,
            "retrieval_latency_ms": retrieval_time_ms,
            "generation_latency_ms": generation_time_ms,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "context_tokens": context_tokens,
            "generation_tokens": generation_tokens,
            "tokens_saved": tokens_saved,
            "cpu_memory_mb": cpu_mem,
            "gpu_memory_mb": gpu_mem,
            "num_retrieved_docs": num_retrieved_docs,
            "retrieved_contexts": retrieved_contexts
        }
    
    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        """Build prompt for LLM from query and context documents"""
        context_str = "\n\n".join([
            f"Context {i+1}: {ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context_str}

Question: {query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def save_metrics(self, filepath: str):
        """Save all metrics to file"""
        self.metrics_monitor.save_metrics(filepath)
    
    def get_statistics(self) -> Dict:
        """Return execution statistics"""
        return self.metrics_monitor.get_summary_statistics()
    
    def print_statistics(self):
        """Print system statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("CAG-ENHANCED RAG SYSTEM STATISTICS")
        print("="*70)
        
        if not stats:
            print("No statistics available yet")
            return
        
        print(f"\nTotal Queries: {stats['total_queries']}")
        print(f"Cache Hit Rate: {stats['cache_performance']['overall_hit_rate']:.2%}")
        print(f"Prompt Cache Hits: {stats['cache_performance']['prompt_cache_hits']}")
        print(f"Semantic Cache Hits: {stats['cache_performance']['semantic_cache_hits']}")
        
        print("\n--- LATENCY METRICS ---")
        lat = stats['latency']['all_queries']
        print(f"Mean: {lat['mean']:.2f}ms")
        print(f"p50: {lat['p50']:.2f}ms")
        print(f"p95: {lat['p95']:.2f}ms")
        
        print("\n--- TOKEN METRICS ---")
        tok = stats['tokens']['total']
        print(f"Total Tokens: {tok['sum']:,}")
        print(f"Mean per query: {tok['mean']:.2f}")
        print(f"Tokens Saved: {stats['tokens']['saved']['total']:,}")
        
        print("\n--- MEMORY METRICS ---")
        print(f"CPU RAM (peak): {stats['memory']['cpu']['peak_mb']:.2f}MB")
        print(f"GPU VRAM (peak): {stats['memory']['gpu']['peak_mb']:.2f}MB")
        
        print("\n" + "="*70)

def main():
    """Example usage of CAG-Enhanced RAG system with separate Qdrant Cloud"""
    
    print("\n" + "="*70)
    print("CAG-ENHANCED RAG SYSTEM SETUP (SEPARATE QDRANT CLOUD)")
    print("="*70)
    
    # Ask user about collection management
    print("\nCollection Management:")
    print("  1. Use existing collection (if available)")
    print("  2. Always recreate collection (start fresh)")
    
    collection_choice = input("\nEnter your choice (1 or 2): ").strip()
    recreate = collection_choice == "2"
    
    # Create configuration
    config = CAGConfig(
        llm_model="llama-3.3-70b-versatile",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_groq_api=True,
        collection_name_cag="cag_rag_docs",
        use_bm25=True,
        use_dense=True,
        top_k=3,
        chunk_size=256,
        prompt_cache_size=1000,
        semantic_cache_size=1000,
        similarity_threshold=0.85,
        ttl_hours=24,
        kv_cache_enabled=True,
        recreate_collection=recreate
    )
    
    # Save configuration
    config.save("cag_rag_config.json")
    
    print(f"\nUsing Qdrant Cloud (CAG):")
    print(f"  URL: {config.qdrant_url_cag}")
    print(f"  Collection: {config.collection_name_cag}")
    
    # Initialize system
    system = CAGEnhancedRAG(config)
    
    # ----------------------------
    # Load documents from JSONL
    # ----------------------------
    data_path = "./data/wikiqa_all/all_data.jsonl"
    documents = []
    metadatas = []
    
    print("\n" + "="*70)
    print("LOADING DOCUMENTS FROM DATASET")
    print("="*70)
    
    try:
        with open(data_path, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"Skipping line {idx} due to JSON error: {e}")
                    continue
                
                # Extract text from various possible fields
                text = obj.get("context") or obj.get("answer") or obj.get("text") or ""
                if not text:
                    continue
                
                documents.append(text)
                metadatas.append({
                    "question": obj.get("question"),
                    "source": "wikiqa",
                    "orig_id": obj.get("id", idx)
                })
        
        print(f"\n✓ Loaded {len(documents)} documents from {data_path}")
        
    except FileNotFoundError:
        print(f"\n⚠ Error: {data_path} not found.")
        print("Using sample documents as fallback...\n")
        
        # Fallback sample documents
        documents = [
            "Paris is the capital and most populous city of France. It is located in the north-central part of the country.",
            "George Orwell wrote the dystopian novel '1984', published in 1949.",
            "World War II ended in 1945, with Germany surrendering in May and Japan in September.",
            "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined.",
            "Alexander Fleming discovered penicillin in 1928, revolutionizing medicine."
        ]
        metadatas = [{"source": "sample", "doc_id": i} for i in range(len(documents))]
    
    # ----------------------------
    # CRITICAL FIX: Actually add documents to the CAG system!
    # ----------------------------
    if documents:
        print("\n" + "="*70)
        print("INGESTING DOCUMENTS INTO CAG-ENHANCED RAG SYSTEM")
        print("="*70)
        
        # This will:
        # 1. Add documents to BM25 index
        # 2. Generate embeddings and store in Qdrant Cloud (CAG instance)
        # 3. Populate both prompt cache and semantic cache
        system.add_documents(documents)
        
        print(f"\n✓ Successfully ingested {len(documents)} documents into:")
        print(f"  - BM25 index")
        print(f"  - Qdrant Cloud (CAG instance)")
        print(f"  - Prompt cache")
        print(f"  - Semantic cache")
    else:
        print("\n⚠ Warning: No documents loaded. System will have no knowledge base.")
        return
    
    # ----------------------------
    # Process test queries
    # ----------------------------
    questions = [
        "What is the capital of France?",
        "Who wrote the novel '1984'?",
        "When did World War II end?",
        "What is the largest planet in the solar system?",
        "Who discovered penicillin?",
    ]
    
    print("\n" + "="*70)
    print("PROCESSING QUERIES WITH CAG")
    print("="*70)
    
    # Process queries (first pass - all cache misses)
    for i, question in enumerate(questions, 1):
        print(f"\n[Query {i}/{len(questions)}] {question}")
        result = system.query(question)
        
        cache_status = "✓ CACHE HIT" if result['cache_hit'] else "✗ CACHE MISS"
        cache_info = f" ({result['cache_type']})" if result['cache_hit'] else ""
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"{cache_status}{cache_info}")
        print(f"Latency: {result['total_latency_ms']:.2f}ms | Tokens: {result['total_tokens']} | Saved: {result['tokens_saved']}")
        print("-" * 70)
    
    # ----------------------------
    # DEMO: Repeat queries to show cache hits
    # ----------------------------
    print("\n" + "="*70)
    print("REPEATING QUERIES TO DEMONSTRATE CACHE EFFECTIVENESS")
    print("="*70)
    
    for i, question in enumerate(questions[:3], 1):  # Repeat first 3 queries
        print(f"\n[Repeat {i}/3] {question}")
        result = system.query(question)
        
        cache_status = "✓ CACHE HIT" if result['cache_hit'] else "✗ CACHE MISS"
        cache_info = f" ({result['cache_type']})" if result['cache_hit'] else ""
        
        print(f"{cache_status}{cache_info}")
        print(f"Latency: {result['total_latency_ms']:.2f}ms (should be much faster!)")
        print("-" * 70)
    
    # Print statistics
    system.print_statistics()
    
    # Save metrics
    system.save_metrics("cag_rag_metrics.json")
    
    # Save cache statistics
    cache_stats = system.cag_layer.get_all_stats()
    with open("cag_cache_stats.json", 'w') as f:
        json.dump(cache_stats, f, indent=2)
    
    print("\n✓ CAG-Enhanced RAG evaluation complete!")
    print("✓ Metrics saved to: cag_rag_metrics.json")
    print("✓ Cache stats saved to: cag_cache_stats.json")
    print("✓ Configuration saved to: cag_rag_config.json")

if __name__ == "__main__":
    main()