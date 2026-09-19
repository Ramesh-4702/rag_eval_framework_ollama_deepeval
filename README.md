# Local RAG Evaluation Framework (Ollama & DeepEval)

A 100% local, privacy-first Retrieval-Augmented Generation (RAG) pipeline and automated continuous evaluation framework. Built using **LangChain, ChromaDB, Ollama, DeepEval, and Pytest**, this framework allows you to build, benchmark, and regression-test RAG pipelines entirely offline without relying on external paid LLM APIs or exposing sensitive data to cloud providers.

---

## Key Features

- **Privacy-First & Offline**: Runs embedding models (`nomic-embed-text`), generation models (`llama3.2`), and judge models (`qwen2.5:7b`) locally via Ollama.
- **Automated Synthetic Test Generation**: Synthesizes golden QA test cases directly from local knowledge base documents (`docs/knowledge_base.txt`).
- **Custom Local LLM Judge**: Features a robust `LocalOllamaJudge` with GGUF JSON mode enforcement and regex post-processing to eliminate JSON formatting errors on lightweight open-source models.
- **Comprehensive RAG Benchmarking**: Evaluates pipeline performance across 4 core RAG pillars:
  - **Faithfulness**: Detects hallucinations and verifies factual grounding.
  - **Answer Relevancy**: Assesses output directness and query alignment.
  - **Contextual Precision**: Measures noise level in retrieved document chunks.
  - **Contextual Recall**: Checks whether retriever captured all context needed for a complete answer.
- **CI/CD Test Runner**: Integrates directly with Pytest for automated regression testing.
- **Optional Observability**: Supports optional cloud dashboard synchronization with **Confident AI**.

---

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Vector Database** | ChromaDB / LangChain | Manages document embeddings and similarity retrieval. |
| **Embeddings** | Ollama (`nomic-embed-text`) | Generates dense vector representations locally. |
| **Generation Model** | Ollama (`llama3.2`) | Generates user answers based on retrieved context. |
| **Evaluator Judge** | Ollama (`qwen2.5:7b`) | Evaluates pipeline outputs against ground-truth contexts. |
| **Evaluation Suite** | DeepEval + Pytest | Runs assertions and calculates metric scores. |

---

## Repository Structure

```text
rag-eval-framework-ollama-deepeval/
├── data/
│   └── synthetic_goldens.json  # Synthesized test cases (QA + context)
├── docs/
│   └── knowledge_base.txt      # Source text used for retrieval
├── src/
│   ├── evaluator.py            # LocalOllamaJudge implementation (JSON mode)
│   └── rag.py                  # RAG retrieval & answer generation pipeline
├── tests/
│   └── test_rag.py             # DeepEval metric test suite
├── generate_goldens.py         # Synthetic dataset generator script
├── pytest.ini                  # Pytest configuration (warning suppression)
├── .gitignore                  # Git ignore file
└── requirements.txt            # Project dependencies
```


## Setup & Installation

### 1. Prerequisites

Install [Ollama](https://ollama.com), start the service, and pull the required models:

```bash
ollama pull llama3.2
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 2. Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
# Clone repository
git clone https://github.com/Ramesh-4702/rag_eval_framework_ollama_deepeval.gitt
cd rag-eval-framework-ollama-deepeval

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows (PowerShell)
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Synthesize Golden Test Dataset

Generate synthetic question-answer-context tuples from your local knowledge base:

```bash
python generate_goldens.py
```

### 2. Run Automated Evaluation Suite

Execute the evaluation tests and view detailed live metric scores and reasoning in your terminal:

```bash
deepeval test run tests/test_rag.py -- -s -rA
```
