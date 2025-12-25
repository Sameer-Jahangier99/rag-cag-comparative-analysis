import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import both systems
from rag_v1 import VanillaRAG, RAGConfig
from cag_v1 import CAGEnhancedRAG, CAGConfig

# ============================================================================
# COMPARISON RUNNER
# ============================================================================

class SystemComparison:
    """Compare Vanilla RAG and CAG-Enhanced RAG systems"""
    
    def __init__(self):
        self.vanilla_results = None
        self.cag_results = None
        self.comparison_data = None
    
    def run_comparison(self, documents: List[str], questions: List[str],
                      include_duplicates: bool = True):
        
        print("\n" + "="*80)
        print("COMPARATIVE EVALUATION: VANILLA RAG vs CAG-ENHANCED RAG")
        print("="*80)
        
        # Add duplicates to test caching benefits
        if include_duplicates:
            original_count = len(questions)
            
            # Add 30% exact duplicates
            num_duplicates = int(len(questions) * 0.3)
            duplicates = list(np.random.choice(questions, num_duplicates, replace=False))
            
            # Add 20% semantic variations
            num_variations = int(len(questions) * 0.2)
            variations = []
            for q in np.random.choice(questions, num_variations, replace=False):
                if q.startswith("What"):
                    variations.append(q.replace("What", "Explain what"))
                elif q.startswith("Who"):
                    variations.append(q.replace("Who", "Can you tell me who"))
                elif q.startswith("When"):
                    variations.append(q.replace("When", "At what time"))
                else:
                    variations.append(f"Tell me about: {q.lower()}")
            
            # Combine all questions
            questions = questions + duplicates + variations
            
            print(f"\nQuery Mix:")
            print(f"  Original: {original_count}")
            print(f"  Exact Duplicates: {num_duplicates}")
            print(f"  Semantic Variations: {num_variations}")
            print(f"  Total: {len(questions)}")
        
        # ====================================================================
        # RUN VANILLA RAG
        # ====================================================================
        
        print("\n" + "="*80)
        print("PHASE 1: RUNNING VANILLA RAG (NO CACHE)")
        print("="*80)
        
        vanilla_config = RAGConfig(
            llm_model="llama-3.3-70b-versatile",  # Groq model
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            use_bm25=True,
            use_dense=True,
            top_k=3,
            chunk_size=256
        )
        
        vanilla_system = VanillaRAG(vanilla_config)
        vanilla_system.add_documents(documents)
        
        print(f"\nProcessing {len(questions)} queries with Vanilla RAG...")
        for i, question in enumerate(questions, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(questions)}")
            vanilla_system.query(question)
        
        self.vanilla_results = vanilla_system.get_statistics()
        vanilla_system.save_metrics("comparison_vanilla_rag.json")
        
        # ====================================================================
        # RUN CAG-ENHANCED RAG
        # ====================================================================
        
        print("\n" + "="*80)
        print("PHASE 2: RUNNING CAG-ENHANCED RAG (WITH CACHING)")
        print("="*80)
        
        cag_config = CAGConfig(
            llm_model="llama-3.3-70b-versatile",  # Groq model
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            use_bm25=True,
            use_dense=True,
            top_k=3,
            chunk_size=256,
            prompt_cache_size=1000,
            semantic_cache_size=1000,
            similarity_threshold=0.85,
            ttl_hours=24,
            kv_cache_enabled=True
        )
        
        cag_system = CAGEnhancedRAG(cag_config)
        cag_system.add_documents(documents)
        
        print(f"\nProcessing {len(questions)} queries with CAG-Enhanced RAG...")
        for i, question in enumerate(questions, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(questions)}")
            cag_system.query(question)
        
        self.cag_results = cag_system.get_statistics()
        cag_system.save_metrics("comparison_cag_rag.json")
        
        # ====================================================================
        # GENERATE COMPARISON
        # ====================================================================
        
        self._compute_comparison()
        self._print_comparison()
        self._save_comparison()
        self._generate_visualizations()
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  - comparison_vanilla_rag.json")
        print("  - comparison_cag_rag.json")
        print("  - comparison_summary.json")
        print("  - comparison_dashboard.html")
    
    def _compute_comparison(self):
        """Compute comparative metrics"""
        
        vanilla = self.vanilla_results
        cag = self.cag_results
        
        # Latency improvements
        vanilla_p50 = vanilla['latency']['total']['p50']
        cag_p50 = cag['latency']['all_queries']['p50']
        latency_improvement_p50 = ((vanilla_p50 - cag_p50) / vanilla_p50) * 100 if vanilla_p50 > 0 else 0
        
        vanilla_p95 = vanilla['latency']['total']['p95']
        cag_p95 = cag['latency']['all_queries']['p95']
        latency_improvement_p95 = ((vanilla_p95 - cag_p95) / vanilla_p95) * 100 if vanilla_p95 > 0 else 0
        
        vanilla_mean = vanilla['latency']['total']['mean']
        cag_mean = cag['latency']['all_queries']['mean']
        latency_improvement_mean = ((vanilla_mean - cag_mean) / vanilla_mean) * 100 if vanilla_mean > 0 else 0
        
        # Token savings (CAG should use FEWER tokens due to cache)
        vanilla_tokens = vanilla['tokens']['total']['sum']
        cag_tokens = cag['tokens']['total']['sum']
        token_reduction = ((vanilla_tokens - cag_tokens) / vanilla_tokens) * 100 if vanilla_tokens > 0 else 0
        
        # Memory comparison - USE MEAN not PEAK for fair comparison
        vanilla_cpu_mean = vanilla['memory']['cpu']['mean_mb']
        cag_cpu_mean = cag['memory']['cpu']['mean_mb']
        cpu_memory_improvement = ((vanilla_cpu_mean - cag_cpu_mean) / vanilla_cpu_mean) * 100 if vanilla_cpu_mean > 0 else 0
        
        vanilla_gpu_mean = vanilla['memory']['gpu']['mean_mb']
        cag_gpu_mean = cag['memory']['gpu']['mean_mb']
        gpu_memory_improvement = ((vanilla_gpu_mean - cag_gpu_mean) / vanilla_gpu_mean) * 100 if vanilla_gpu_mean > 0 else 0
        
        # Throughput comparison
        vanilla_qps = vanilla['throughput']['queries_per_second']
        cag_qps = cag['throughput']['queries_per_second']
        throughput_improvement = ((cag_qps - vanilla_qps) / vanilla_qps) * 100 if vanilla_qps > 0 else 0
        
        self.comparison_data = {
            "latency": {
                "vanilla_p50_ms": vanilla_p50,
                "cag_p50_ms": cag_p50,
                "improvement_p50_percent": latency_improvement_p50,
                "vanilla_p95_ms": vanilla_p95,
                "cag_p95_ms": cag_p95,
                "improvement_p95_percent": latency_improvement_p95,
                "vanilla_mean_ms": vanilla_mean,
                "cag_mean_ms": cag_mean,
                "improvement_mean_percent": latency_improvement_mean,
                "speedup_factor": cag['latency']['speedup_factor']
            },
            "tokens": {
                "vanilla_total": vanilla_tokens,
                "cag_total": cag_tokens,
                "tokens_saved": vanilla_tokens - cag_tokens,
                "reduction_percent": token_reduction,
                "vanilla_mean_per_query": vanilla['tokens']['total']['mean'],
                "cag_mean_per_query": cag['tokens']['total']['mean']
            },
            "memory": {
                "cpu": {
                    "vanilla_mean_mb": vanilla_cpu_mean,
                    "cag_mean_mb": cag_cpu_mean,
                    "improvement_percent": cpu_memory_improvement
                },
                "gpu": {
                    "vanilla_mean_mb": vanilla_gpu_mean,
                    "cag_mean_mb": cag_gpu_mean,
                    "improvement_percent": gpu_memory_improvement
                }
            },
            "throughput": {
                "vanilla_qps": vanilla_qps,
                "cag_qps": cag_qps,
                "improvement_percent": throughput_improvement
            },
            "cache_performance": {
                "hit_rate": cag['cache_performance']['overall_hit_rate'],
                "prompt_cache_hits": cag['cache_performance']['prompt_cache_hits'],
                "semantic_cache_hits": cag['cache_performance']['semantic_cache_hits']
            }
        }
    
    def _print_comparison(self):
        """Print formatted comparison"""
        
        comp = self.comparison_data
        
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS SUMMARY")
        print("="*80)
        
        print("\n--- LATENCY COMPARISON ---")
        print(f"Median (p50):")
        print(f"  Vanilla RAG: {comp['latency']['vanilla_p50_ms']:.2f}ms")
        print(f"  CAG RAG:     {comp['latency']['cag_p50_ms']:.2f}ms")
        print(f"  Improvement: {comp['latency']['improvement_p50_percent']:.2f}%")
        
        print(f"\np95:")
        print(f"  Vanilla RAG: {comp['latency']['vanilla_p95_ms']:.2f}ms")
        print(f"  CAG RAG:     {comp['latency']['cag_p95_ms']:.2f}ms")
        print(f"  Improvement: {comp['latency']['improvement_p95_percent']:.2f}%")
        
        print(f"\nMean:")
        print(f"  Vanilla RAG: {comp['latency']['vanilla_mean_ms']:.2f}ms")
        print(f"  CAG RAG:     {comp['latency']['cag_mean_ms']:.2f}ms")
        print(f"  Improvement: {comp['latency']['improvement_mean_percent']:.2f}%")
        print(f"  Speedup:     {comp['latency']['speedup_factor']:.2f}x")
        
        print("\n--- TOKEN USAGE COMPARISON ---")
        print(f"Total Tokens:")
        print(f"  Vanilla RAG: {comp['tokens']['vanilla_total']:,}")
        print(f"  CAG RAG:     {comp['tokens']['cag_total']:,}")
        print(f"  Saved:       {comp['tokens']['tokens_saved']:,}")
        print(f"  Reduction:   {comp['tokens']['reduction_percent']:.2f}%")
        
        print(f"\nMean Tokens per Query:")
        print(f"  Vanilla RAG: {comp['tokens']['vanilla_mean_per_query']:.2f}")
        print(f"  CAG RAG:     {comp['tokens']['cag_mean_per_query']:.2f}")
        
        print("\n--- MEMORY COMPARISON ---")
        print(f"CPU RAM (Mean):")
        print(f"  Vanilla RAG: {comp['memory']['cpu']['vanilla_mean_mb']:.2f}MB")
        print(f"  CAG RAG:     {comp['memory']['cpu']['cag_mean_mb']:.2f}MB")
        print(f"  Improvement: {comp['memory']['cpu']['improvement_percent']:.2f}%")
        
        print(f"\nGPU VRAM (Mean):")
        print(f"  Vanilla RAG: {comp['memory']['gpu']['vanilla_mean_mb']:.2f}MB")
        print(f"  CAG RAG:     {comp['memory']['gpu']['cag_mean_mb']:.2f}MB")
        print(f"  Improvement: {comp['memory']['gpu']['improvement_percent']:.2f}%")
        
        print("\n--- THROUGHPUT COMPARISON ---")
        print(f"Queries Per Second:")
        print(f"  Vanilla RAG: {comp['throughput']['vanilla_qps']:.2f}")
        print(f"  CAG RAG:     {comp['throughput']['cag_qps']:.2f}")
        print(f"  Improvement: {comp['throughput']['improvement_percent']:.2f}%")
        
        print("\n--- CACHE PERFORMANCE (CAG Only) ---")
        print(f"Overall Hit Rate: {comp['cache_performance']['hit_rate']:.2%}")
        print(f"Prompt Cache Hits: {comp['cache_performance']['prompt_cache_hits']}")
        print(f"Semantic Cache Hits: {comp['cache_performance']['semantic_cache_hits']}")
        
        print("\n" + "="*80)
    
    def _save_comparison(self):
            """Save comparison data to JSON"""
            
            # Helper function to convert NumPy types to Python types
            def numpy_converter(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Type {type(obj)} not serializable")

            try:
                with open("comparison_summary.json", 'w') as f:
                    # Use the 'default' parameter to handle numpy types
                    json.dump(self.comparison_data, f, indent=2, default=numpy_converter)
                print("\n✓ Comparison summary saved to: comparison_summary.json")
            except Exception as e:
                print(f"\n⚠ Error saving comparison summary: {e}")
    
    def _generate_visualizations(self):
        """Generate comparative visualizations"""
        
        comp = self.comparison_data
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Latency Comparison (Lower is Better)',
                'Token Usage Comparison',
                'Memory Usage Comparison',
                'Throughput Comparison (Higher is Better)',
                'Latency Distribution',
                'Cache Hit Rate'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Latency comparison
        fig.add_trace(
            go.Bar(
                name='Vanilla RAG',
                x=['p50', 'p95', 'Mean'],
                y=[comp['latency']['vanilla_p50_ms'],
                   comp['latency']['vanilla_p95_ms'],
                   comp['latency']['vanilla_mean_ms']],
                marker_color='lightcoral'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                name='CAG RAG',
                x=['p50', 'p95', 'Mean'],
                y=[comp['latency']['cag_p50_ms'],
                   comp['latency']['cag_p95_ms'],
                   comp['latency']['cag_mean_ms']],
                marker_color='lightgreen'
            ),
            row=1, col=1
        )
        
        # 2. Token usage comparison
        fig.add_trace(
            go.Bar(
                x=['Vanilla RAG', 'CAG RAG'],
                y=[comp['tokens']['vanilla_total'],
                   comp['tokens']['cag_total']],
                marker_color=['lightcoral', 'lightgreen'],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Memory usage comparison - USE MEAN
        fig.add_trace(
            go.Bar(
                name='CPU',
                x=['Vanilla RAG', 'CAG RAG'],
                y=[comp['memory']['cpu']['vanilla_mean_mb'],
                   comp['memory']['cpu']['cag_mean_mb']],
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                name='GPU',
                x=['Vanilla RAG', 'CAG RAG'],
                y=[comp['memory']['gpu']['vanilla_mean_mb'],
                   comp['memory']['gpu']['cag_mean_mb']],
                marker_color='lightpink'
            ),
            row=2, col=1
        )
        
        # 4. Throughput comparison
        fig.add_trace(
            go.Bar(
                x=['Vanilla RAG', 'CAG RAG'],
                y=[comp['throughput']['vanilla_qps'],
                   comp['throughput']['cag_qps']],
                marker_color=['lightcoral', 'lightgreen'],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. Latency improvement percentages
        fig.add_trace(
            go.Bar(
                x=['p50', 'p95', 'Mean'],
                y=[comp['latency']['improvement_p50_percent'],
                   comp['latency']['improvement_p95_percent'],
                   comp['latency']['improvement_mean_percent']],
                marker_color='lightgreen',
                showlegend=False,
                text=[f"{x:.1f}%" for x in [
                    comp['latency']['improvement_p50_percent'],
                    comp['latency']['improvement_p95_percent'],
                    comp['latency']['improvement_mean_percent']
                ]],
                textposition='outside'
            ),
            row=3, col=1
        )
        
        # 6. Cache hit rate gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=comp['cache_performance']['hit_rate'] * 100,
                title={'text': "Cache Hit Rate (%)"},
                delta={'reference': 60, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Vanilla RAG vs CAG-Enhanced RAG: Comprehensive Comparison",
            title_font_size=20
        )
        
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Total Tokens", row=1, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
        fig.update_yaxes(title_text="QPS", row=2, col=2)
        fig.update_yaxes(title_text="Improvement (%)", row=3, col=1)
        
        # Save
        fig.write_html("comparison_dashboard.html")
        print("✓ Visualization saved to: comparison_dashboard.html")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comparative evaluation using combined JSONL dataset"""
    
    print("\n" + "="*80)
    print("COMPARATIVE EVALUATION SETUP")
    print("="*80)
    
    # Check for Groq API key
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in environment variables!")
        print("Please add GROQ_API_KEY=your_api_key to your .env file")
        return
    
    # ----------------------------
    # Load documents from JSONL
    # ----------------------------
    data_path = "./data/wikiqa_all/all_data.jsonl"
    documents = []
    metadatas = []
    
    print("\n" + "="*80)
    print("LOADING DOCUMENTS FROM DATASET")
    print("="*80)
    
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
            "Alexander Fleming discovered penicillin in 1928, revolutionizing medicine.",
            "The Great Wall of China is one of the most famous landmarks in the world.",
            "Albert Einstein developed the theory of relativity in the early 20th century.",
            "The Amazon rainforest is the largest tropical rainforest in the world.",
            "Shakespeare wrote many famous plays including Hamlet, Macbeth, and Romeo and Juliet.",
            "The human body has 206 bones in total."
        ]
        metadatas = [{"source": "sample", "doc_id": i} for i in range(len(documents))]
    
    # Validate we have documents
    if not documents:
        print("\n❌ ERROR: No documents loaded. Cannot proceed with comparison.")
        return
    
    print(f"\n✓ Ready to ingest {len(documents)} documents into both systems")
    
    # ----------------------------
    # Prepare test queries
    # ----------------------------
    questions = [
        "What is the capital of France?",
        "Who wrote the novel '1984'?",
        "When did World War II end?",
        "What is the largest planet in the solar system?",
        "Who discovered penicillin?",
        "Where is the Great Wall located?",
        "Who developed the theory of relativity?",
        "What is the largest rainforest?",
        "What plays did Shakespeare write?",
        "How many bones are in the human body?"
    ]
    
    print(f"\n✓ Prepared {len(questions)} test queries")
    
    # ----------------------------
    # Run comparison with proper document ingestion
    # ----------------------------
    print("\n" + "="*80)
    print("STARTING COMPARATIVE EVALUATION")
    print("="*80)
    
    comparison = SystemComparison()
    comparison.run_comparison(
        documents=documents,
        questions=questions,
        include_duplicates=True  # Add duplicates to test cache effectiveness
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\n✓ All results saved successfully")
    print("\nGenerated files:")
    print("  📊 comparison_dashboard.html - Interactive visualizations")
    print("  📄 comparison_summary.json - Detailed comparison metrics")
    print("  📄 comparison_vanilla_rag.json - Vanilla RAG detailed metrics")
    print("  📄 comparison_cag_rag.json - CAG RAG detailed metrics")
    print("  📄 comparison_vanilla_rag.json - Vanilla RAG detailed metrics")
    print("  📄 comparison_cag_rag.json - CAG RAG detailed metrics")
    print("\nOpen comparison_dashboard.html in your browser to view results!")
if __name__ == "__main__":    
    main()