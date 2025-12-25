# 3.3.5 Development Environment (CORRECTED)

## Programming Language
**Python 3.10.12**
- Rationale: Extensive ML/AI library ecosystem and compatibility with required libraries

## Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.1.0 | Deep learning framework |
| Sentence-Transformers | 2.2.2 | Embeddings generation |
| Qdrant-client | 1.7.0 | Vector database interface (Qdrant Cloud) |
| rank-bm25 | 0.2.2 | Sparse retrieval (BM25 algorithm) |
| NumPy | 1.24.3 | Numerical computing |
| Pandas | 2.0.3 | Data manipulation and analysis |
| scikit-learn | 1.3.0 | Metrics & preprocessing (cosine similarity) |

## API & Environment Management

| Library | Version | Purpose |
|---------|---------|---------|
| python-dotenv | 0.21.0 | Environment variable management (.env file loading) |
| groq | 0.4.0+ | Groq Cloud API client (primary LLM inference interface) |

**Note:** While Transformers (4.35.0) is not directly imported, it is used as a transitive dependency by Sentence-Transformers for underlying model operations.

## Monitoring & Profiling

| Library | Version | Purpose |
|---------|---------|---------|
| psutil | 5.9.5 | CPU/memory monitoring |
| pynvml | 11.5.0 | GPU metrics (NVIDIA) |

**Note:** tqdm (4.66.1) is used internally by Sentence-Transformers for progress bars when `show_progress_bar=True` is enabled, but is not directly imported.

## Visualization

| Library | Version | Purpose |
|---------|---------|---------|
| Plotly | 5.17.0 | Interactive visualizations (comparison dashboard) |

**Note:** Matplotlib and Seaborn are not used in the current implementation. They can be removed unless planned for future use.

---

## Implementation Notes

### LLM Interface
- **Primary API:** Groq Cloud API (via `groq` library)
- **Model:** llama-3.3-70b-versatile (accessed via Groq API)
- **Rationale:** Fast inference, no local model loading required

### Vector Database
- **Service:** Qdrant Cloud (managed vector database)
- **Client:** qdrant-client library
- **Usage:** Separate collections for Vanilla RAG and CAG-Enhanced RAG systems

### Standard Library Modules Used
The implementation also utilizes Python standard library modules:
- `json`, `time`, `os`, `pathlib`, `typing`, `dataclasses`, `collections`, `hashlib`, `uuid`

---

## Version Compatibility
All specified versions are compatible with Python 3.10.12 and have been tested in the implementation.

