import time
import json
import torch
import numpy as np
import psutil
import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Load environment variables
load_dotenv()

# Core imports
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
import uuid

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False
    print("Warning: NVML not available. GPU metrics will not be collected.")

# ============================================================================
# AUTHENTICATION
# ============================================================================

def authenticate_huggingface():
    """Authenticate with Hugging Face using token from .env"""
    try:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✓ Hugging Face authentication successful")
        else:
            print("⚠ HF_TOKEN not found in .env, attempting default login...")
            login()
    except Exception as e:
        print(f"⚠ Hugging Face authentication failed: {e}")
        print("Please set HF_TOKEN in .env file")
        print("Get your token at: https://huggingface.co/settings/tokens")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for Vanilla RAG System"""
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
    
    # Qdrant Cloud settings (loaded from .env)
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = "vanilla_rag_docs"
    
    # Groq API settings
    groq_api_key: str = ""
    
    # Collection management
    recreate_collection: bool = False
    
    def __post_init__(self):
        """Load credentials from environment if not provided"""
        if not self.qdrant_url:
            self.qdrant_url = os.getenv("QDRANT_URL", "https://your-qdrant-instance.qdrant.io")
        if not self.qdrant_api_key:
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "your-api-key-here")
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY", "")
    
    def save(self, path: str):
        """Save configuration to JSON (without credentials)"""
        config_dict = asdict(self)
        # Don't save sensitive data
        config_dict['qdrant_api_key'] = "***REDACTED***"
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
        data['groq_api_key'] = os.getenv("GROQ_API_KEY", data.get('groq_api_key', ''))
        return cls(**data)

# ============================================================================
# METRICS MONITORING
# ============================================================================

@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query_id: int
    query: str
    answer: str
    
    # Timing metrics
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    
    # Token metrics
    prompt_tokens: int
    context_tokens: int
    generation_tokens: int
    total_tokens: int
    
    # Memory metrics
    cpu_memory_mb: float
    gpu_memory_mb: float
    
    # Retrieval metrics
    num_retrieved_docs: int
    retrieved_contexts: List[str]
    
    # Quality metrics (to be filled by evaluation)
    exact_match: Optional[float] = None
    f1_score: Optional[float] = None
    
    timestamp: float = time.time()

class MetricsMonitor:
    """Monitor and track all system metrics"""
    
    def __init__(self):
        self.query_metrics: List[QueryMetrics] = []
        self.process = psutil.Process()
        
        if NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
                print("Warning: Could not get GPU handle")
        else:
            self.gpu_handle = None
        
        self.query_counter = 0
        self.start_time = time.time()
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current CPU and GPU memory usage in MB"""
        # CPU memory
        cpu_mem = self.process.memory_info().rss / (1024 ** 2)
        
        # GPU memory
        gpu_mem = 0.0
        if NVML_AVAILABLE and self.gpu_handle:
            try:
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem = gpu_info.used / (1024 ** 2)
            except:
                pass
        
        return cpu_mem, gpu_mem
    
    def record_query(self, metrics: QueryMetrics):
        """Record metrics for a query"""
        self.query_metrics.append(metrics)
        self.query_counter += 1
    
    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics across all queries"""
        if not self.query_metrics:
            return {}
        
        # Latency statistics
        total_latencies = [m.total_latency_ms for m in self.query_metrics]
        retrieval_latencies = [m.retrieval_latency_ms for m in self.query_metrics]
        generation_latencies = [m.generation_latency_ms for m in self.query_metrics]
        
        # Token statistics
        total_tokens = [m.total_tokens for m in self.query_metrics]
        prompt_tokens = [m.prompt_tokens for m in self.query_metrics]
        context_tokens = [m.context_tokens for m in self.query_metrics]
        generation_tokens = [m.generation_tokens for m in self.query_metrics]
        
        # Memory statistics
        cpu_memory = [m.cpu_memory_mb for m in self.query_metrics]
        gpu_memory = [m.gpu_memory_mb for m in self.query_metrics]
        
        return {
            "total_queries": len(self.query_metrics),
            "runtime_seconds": time.time() - self.start_time,
            
            # Latency metrics
            "latency": {
                "total": {
                    "mean": np.mean(total_latencies),
                    "median": np.median(total_latencies),
                    "p50": np.percentile(total_latencies, 50),
                    "p95": np.percentile(total_latencies, 95),
                    "p99": np.percentile(total_latencies, 99),
                    "std": np.std(total_latencies),
                    "min": np.min(total_latencies),
                    "max": np.max(total_latencies)
                },
                "retrieval": {
                    "mean": np.mean(retrieval_latencies),
                    "median": np.median(retrieval_latencies)
                },
                "generation": {
                    "mean": np.mean(generation_latencies),
                    "median": np.median(generation_latencies)
                }
            },
            
            # Token metrics
            "tokens": {
                "total": {
                    "sum": sum(total_tokens),
                    "mean": np.mean(total_tokens),
                    "median": np.median(total_tokens),
                    "std": np.std(total_tokens)
                },
                "prompt": {
                    "mean": np.mean(prompt_tokens),
                    "median": np.median(prompt_tokens)
                },
                "context": {
                    "mean": np.mean(context_tokens),
                    "median": np.median(context_tokens)
                },
                "generation": {
                    "mean": np.mean(generation_tokens),
                    "median": np.median(generation_tokens)
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
        """Save all metrics to JSON file"""
        output = {
            "config": "vanilla_rag",
            "summary": self.get_summary_statistics(),
            "individual_queries": [
                {
                    "query_id": m.query_id,
                    "query": m.query,
                    "answer": m.answer,
                    "total_latency_ms": m.total_latency_ms,
                    "retrieval_latency_ms": m.retrieval_latency_ms,
                    "generation_latency_ms": m.generation_latency_ms,
                    "total_tokens": m.total_tokens,
                    "prompt_tokens": m.prompt_tokens,
                    "context_tokens": m.context_tokens,
                    "generation_tokens": m.generation_tokens,
                    "cpu_memory_mb": m.cpu_memory_mb,
                    "gpu_memory_mb": m.gpu_memory_mb,
                    "num_retrieved_docs": m.num_retrieved_docs,
                    "timestamp": m.timestamp
                }
                for m in self.query_metrics
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nMetrics saved to: {filepath}")

# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """Hybrid retrieval with BM25 and Dense vectors"""
    
    def __init__(self, config: RAGConfig):
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
        
        # Initialize Qdrant Cloud
        if config.use_dense:
            print(f"Connecting to Qdrant Cloud...")
            base_url = config.qdrant_url.replace(":6333", "")
            self.qdrant_client = QdrantClient(
                url=base_url,
                api_key=config.qdrant_api_key,
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
        
        collection_exists = self.config.collection_name in collection_names
        
        if collection_exists:
            print(f"\n✓ Collection '{self.config.collection_name}' already exists")
            
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
                print(f"Deleting existing collection: {self.config.collection_name}")
                self.qdrant_client.delete_collection(
                    collection_name=self.config.collection_name
                )
                print("Creating new collection...")
                self._create_collection()
            else:
                print(f"Using existing collection: {self.config.collection_name}")
        else:
            print(f"Collection '{self.config.collection_name}' does not exist")
            print("Creating new collection...")
            self._create_collection()
    
    def _create_collection(self):
        """Create a new Qdrant collection"""
        self.qdrant_client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        print(f"✓ Created Qdrant collection: {self.config.collection_name}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []
        
        step = self.config.chunk_size - self.config.chunk_overlap
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + self.config.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to both BM25 and Qdrant"""
        print(f"\nAdding {len(documents)} documents to retriever...")
        
        # Chunk documents
        all_chunks = []
        all_metadatas = []
        
        for idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
            
            metadata = metadatas[idx] if metadatas else {}
            for chunk_idx in range(len(chunks)):
                chunk_metadata = metadata.copy()
                chunk_metadata['doc_id'] = idx
                chunk_metadata['chunk_id'] = chunk_idx
                all_metadatas.append(chunk_metadata)
        
        self.documents = all_chunks
        print(f"Created {len(all_chunks)} chunks")
        
        # Initialize BM25
        if self.config.use_bm25:
            print("Building BM25 index...")
            tokenized_docs = [doc.lower().split() for doc in all_chunks]
            self.bm25 = BM25Okapi(tokenized_docs)
            print("BM25 index built")
        
        # Add to Qdrant
        if self.config.use_dense and self.embedding_model:
            print("Generating embeddings and uploading to Qdrant...")
            embeddings = self.embedding_model.encode(
                all_chunks,
                show_progress_bar=True,
                batch_size=32
            )
            
            points = []
            for idx, (chunk, embedding, metadata) in enumerate(
                zip(all_chunks, embeddings, all_metadatas)
            ):
                point_id = str(uuid.uuid4())
                self.doc_ids.append(point_id)
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk,
                            **metadata
                        }
                    )
                )
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.config.collection_name,
                    points=batch
                )
            
            print(f"Uploaded {len(points)} vectors to Qdrant")
    
    def retrieve_bm25(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """BM25 sparse retrieval"""
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(self.documents[idx], scores[idx]) for idx in top_indices]
        
        return results
    
    def retrieve_dense(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Dense vector retrieval"""
        if not self.embedding_model or not self.qdrant_client:
            return []
        
        query_embedding = self.embedding_model.encode(query)
        
        search_results = self.qdrant_client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        results = [
            (hit.payload['text'], hit.score) 
            for hit in search_results
        ]
        
        return results
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[str], float]:
        """
        Hybrid retrieval combining BM25 and dense search
        Returns: (retrieved_contexts, retrieval_time_ms)
        """
        if top_k is None:
            top_k = self.config.top_k
        
        start_time = time.perf_counter()
        
        all_results = {}
        
        # BM25 retrieval
        if self.config.use_bm25:
            bm25_results = self.retrieve_bm25(query, top_k * 2)
            for text, score in bm25_results:
                all_results[text] = all_results.get(text, 0) + score * 0.5
        
        # Dense retrieval
        if self.config.use_dense:
            dense_results = self.retrieve_dense(query, top_k * 2)
            for text, score in dense_results:
                all_results[text] = all_results.get(text, 0) + score * 0.5
        
        # Sort by combined score
        sorted_results = sorted(
            all_results.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        contexts = [text for text, score in sorted_results[:top_k]]
        
        retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        
        return contexts, retrieval_time_ms


class LLMEngine:
    """LLM generation engine using Groq Cloud API"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        print(f"\nInitializing LLM Engine...")
        print(f"Model: {config.llm_model}")
        print(f"API: Groq Cloud")
        
        # Initialize Groq client
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file. Please set it.")
        
        self.client = Groq(api_key=config.groq_api_key)
        self.conversation_history = []
        
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


class VanillaRAG:
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        print("\n" + "="*70)
        print("INITIALIZING VANILLA RAG SYSTEM (NO CACHE)")
        print("="*70)
        
        # Initialize components
        self.retriever = HybridRetriever(config)
        self.llm_engine = LLMEngine(config)
        self.metrics_monitor = MetricsMonitor()
        
        print("\n" + "="*70)
        print("VANILLA RAG SYSTEM READY")
        print("="*70 + "\n")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to the retrieval system"""
        self.retriever.add_documents(documents, metadatas)
    
    def build_prompt(self, query: str, contexts: List[str]) -> str:
        """Build prompt with retrieved contexts"""
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
    
    def query(self, question: str) -> Dict:

        query_start = time.perf_counter()
        query_id = self.metrics_monitor.query_counter
        
        # Get memory BEFORE this query
        cpu_mem_before, gpu_mem_before = self.metrics_monitor.get_memory_usage()
        
        # Step 1: Retrieve contexts
        contexts, retrieval_time_ms = self.retriever.retrieve(question)
        cpu_mem_after_retrieval, gpu_mem_after_retrieval = self.metrics_monitor.get_memory_usage()
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, contexts)
        context_tokens = sum([
            self.llm_engine.count_tokens(ctx) 
            for ctx in contexts
        ])
        
        # Step 3: Generate answer
        answer, generation_time_ms, prompt_tokens, generation_tokens = \
            self.llm_engine.generate(prompt)
        cpu_mem_after_generation, gpu_mem_after_generation = self.metrics_monitor.get_memory_usage()
        
        # Calculate total metrics
        total_latency_ms = (time.perf_counter() - query_start) * 1000
        total_tokens = prompt_tokens + generation_tokens
        
        # MEMORY: Use average memory during full operation (not peak)
        cpu_mem_avg = (cpu_mem_after_retrieval + cpu_mem_after_generation) / 2
        gpu_mem_avg = (gpu_mem_after_retrieval + gpu_mem_after_generation) / 2
        cpu_mem = cpu_mem_avg
        gpu_mem = gpu_mem_avg
        
        # Create metrics object
        metrics = QueryMetrics(
            query_id=query_id,
            query=question,
            answer=answer.strip(),
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_time_ms,
            generation_latency_ms=generation_time_ms,
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
            generation_tokens=generation_tokens,
            total_tokens=total_tokens,
            cpu_memory_mb=cpu_mem,
            gpu_memory_mb=gpu_mem,
            num_retrieved_docs=len(contexts),
            retrieved_contexts=contexts
        )
        
        # Record metrics
        self.metrics_monitor.record_query(metrics)
        
        # Return result dictionary
        return {
            "query_id": query_id,
            "question": question,
            "answer": answer.strip(),
            "contexts": contexts,
            "total_latency_ms": total_latency_ms,
            "retrieval_latency_ms": retrieval_time_ms,
            "generation_latency_ms": generation_time_ms,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "context_tokens": context_tokens,
            "generation_tokens": generation_tokens,
            "cpu_memory_mb": cpu_mem,
            "gpu_memory_mb": gpu_mem
        }
    
    def get_statistics(self) -> Dict:
        """Get summary statistics"""
        return self.metrics_monitor.get_summary_statistics()
    
    def save_metrics(self, filepath: str):
        """Save all metrics to file"""
        self.metrics_monitor.save_metrics(filepath)
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("VANILLA RAG SYSTEM STATISTICS")
        print("="*70)
        
        print(f"\nTotal Queries: {stats['total_queries']}")
        print(f"Runtime: {stats['runtime_seconds']:.2f} seconds")
        print(f"Throughput: {stats['throughput']['queries_per_second']:.2f} QPS")
        
        print("\n--- LATENCY METRICS ---")
        lat = stats['latency']['total']
        print(f"Mean: {lat['mean']:.2f}ms")
        print(f"Median (p50): {lat['p50']:.2f}ms")
        print(f"p95: {lat['p95']:.2f}ms")
        print(f"p99: {lat['p99']:.2f}ms")
        print(f"Std Dev: {lat['std']:.2f}ms")
        print(f"Min: {lat['min']:.2f}ms, Max: {lat['max']:.2f}ms")
        
        print("\n--- LATENCY BREAKDOWN ---")
        print(f"Retrieval (mean): {stats['latency']['retrieval']['mean']:.2f}ms")
        print(f"Generation (mean): {stats['latency']['generation']['mean']:.2f}ms")
        
        print("\n--- TOKEN METRICS ---")
        tok = stats['tokens']['total']
        print(f"Total Tokens: {tok['sum']:,}")
        print(f"Mean per query: {tok['mean']:.2f}")
        print(f"Median per query: {tok['median']:.2f}")
        print(f"Std Dev: {tok['std']:.2f}")
        
        print("\n--- TOKEN BREAKDOWN ---")
        print(f"Prompt (mean): {stats['tokens']['prompt']['mean']:.2f}")
        print(f"Context (mean): {stats['tokens']['context']['mean']:.2f}")
        print(f"Generation (mean): {stats['tokens']['generation']['mean']:.2f}")
        
        print("\n--- MEMORY METRICS ---")
        print(f"CPU RAM (mean): {stats['memory']['cpu']['mean_mb']:.2f}MB")
        print(f"CPU RAM (peak): {stats['memory']['cpu']['peak_mb']:.2f}MB")
        print(f"GPU VRAM (mean): {stats['memory']['gpu']['mean_mb']:.2f}MB")
        print(f"GPU VRAM (peak): {stats['memory']['gpu']['peak_mb']:.2f}MB")
        
        print("\n" + "="*70)


def main():
    """Example usage of Vanilla RAG system with Groq API"""
    
    print("\n" + "="*70)
    print("VANILLA RAG SYSTEM SETUP")
    print("="*70)
    
    # Ask user about collection management
    print("\nCollection Management:")
    print("  1. Use existing collection (if available)")
    print("  2. Always recreate collection (start fresh)")
    
    collection_choice = input("\nEnter your choice (1 or 2): ").strip()
    recreate = collection_choice == "2"
    
    # Create configuration
    config = RAGConfig(
        llm_model="llama-3.3-70b-versatile",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_groq_api=True,
        collection_name="vanilla_rag_docs",
        use_bm25=True,
        use_dense=True,
        top_k=3,
        chunk_size=256,
        recreate_collection=recreate
    )
    
    # Save configuration
    config.save("vanilla_rag_config.json")
    
    # Initialize system
    system = VanillaRAG(config)
    
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
    # CRITICAL FIX: Actually add documents to the system!
    # ----------------------------
    if documents:
        print("\n" + "="*70)
        print("INGESTING DOCUMENTS INTO RAG SYSTEM")
        print("="*70)
        system.add_documents(documents, metadatas)
        print(f"\n✓ Successfully ingested {len(documents)} documents into Qdrant and BM25")
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
    print("PROCESSING QUERIES")
    print("="*70)
    
    # Process queries
    for i, question in enumerate(questions, 1):
        print(f"\n[Query {i}/{len(questions)}] {question}")
        result = system.query(question)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Latency: {result['total_latency_ms']:.2f}ms | Tokens: {result['total_tokens']}")
        print("-" * 70)
    
    # Print statistics
    system.print_statistics()
    
    # Save metrics
    system.save_metrics("vanilla_rag_metrics.json")
    
    print("\n✓ Vanilla RAG evaluation complete!")
    print("✓ Metrics saved to: vanilla_rag_metrics.json")
    print("✓ Configuration saved to: vanilla_rag_config.json")

if __name__ == "__main__":
    main()
