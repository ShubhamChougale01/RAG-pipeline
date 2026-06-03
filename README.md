# 🤖 RAG Pipeline

> An intelligent LangGraph pipeline that routes queries between a live weather agent and a resume-based RAG system, with full LangSmith observability.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflows-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-FF6B35?style=flat-square&logo=langchain&logoColor=white)](https://smith.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Groq](https://img.shields.io/badge/Groq-gemma2--9b--it-F55036?style=flat-square)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Overview

**RAG Pipeline** demonstrates production-grade AI pipeline patterns in a single coherent project. It combines intelligent query routing, vector search over PDF documents, real-time external API integration, and end-to-end LLM tracing — all orchestrated through a LangGraph master agent with a Streamlit front-end.

This is not a toy demo. Every component reflects patterns used in production LLM systems: stateful graph-based routing, embedding-based retrieval with a persistent vector store, structured observability, and isolated unit-testable agents.

| Capability | Technology |
|---|---|
| Agentic routing workflow | LangGraph |
| LLM inference | Groq (gemma2-9b-it) |
| Vector search | Qdrant + HuggingFace all-MiniLM-L6-v2 |
| Document ingestion | PyPDF2 + LangChain text splitters |
| Weather data | OpenWeatherMap REST API |
| Tracing & observability | LangSmith |
| Interactive UI | Streamlit |
| Testing | Python unittest |

---

## Key Features

- **LangGraph Master Agent** — classifies query intent and routes to the appropriate downstream agent
- **Resume RAG Agent** — chunks, embeds, and retrieves from a PDF resume stored in Qdrant; generates answers via Groq
- **Weather Agent** — fetches real-time conditions from OpenWeatherMap and formats a natural-language response
- **LangSmith Observability** — every LLM call is traced, latency and token usage visible in the LangSmith dashboard
- **Streamlit UI** — clean, interactive interface for submitting queries and viewing responses
- **Full Unit Test Suite** — isolated tests for routing logic, RAG retrieval, and weather integration

---

## Architecture

```
User Query (Streamlit)
        |
        v
+-------------------+
|  LangGraph Master |  <-- classifies intent: "weather" or "resume"
|      Agent        |
+-------------------+
        |
   +---------+----------+
   |                    |
   v                    v
+-------------+   +------------------+
| WeatherAgent|   |    RAG Agent     |
|             |   |                  |
| OpenWeather |   | Qdrant retrieval |
|  Map API    |   |  --> Groq LLM    |
+-------------+   +------------------+
        |                    |
        +--------+-----------+
                 |
                 v
         Formatted Response
                 |
                 v
     LangSmith Trace (async)
                 |
                 v
      Streamlit UI (display)
```

---

## Feature Deep-Dive

### LangGraph Master Agent

**What it does**
The master agent is a stateful LangGraph `StateGraph` that accepts a user query, invokes an intent-classification step, and conditionally edges to either the `WeatherAgent` node or the `RAGAgent` node. State is passed between nodes as a typed dict, keeping the workflow inspectable and extensible.

**Why it matters**
Hard-coded `if/else` routing breaks down as pipelines grow. LangGraph encodes routing logic as a directed graph, making it trivial to add new agents, insert validation steps, or introduce parallel branches — without restructuring existing code.

---

### RAG Agent (Qdrant + Groq)

**What it does**
At indexing time, `populate_collection.py` reads the PDF resume via PyPDF2, splits it into overlapping chunks, encodes each chunk with `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers), and upserts the vectors into a local Qdrant collection. At query time, the RAG agent embeds the question, performs a cosine-similarity search against Qdrant, retrieves the top-k chunks, and passes them as context to the Groq-hosted `gemma2-9b-it` model via a LangChain `RetrievalQA` chain.

**Why it matters**
Embedding a document once and querying it repeatedly is the canonical pattern for "chat with your data" products. This implementation is fully local (Qdrant runs on-prem) and swappable — the vector store, embedding model, and LLM are each behind a LangChain abstraction.

---

### Weather Agent

**What it does**
A thin wrapper around the OpenWeatherMap Current Weather API. The agent extracts a city name from the user query, calls the REST endpoint, and formats temperature, humidity, and conditions into a concise natural-language answer.

**Why it matters**
Real-world pipelines almost always involve external APIs alongside retrieval-augmented generation. Keeping the weather agent as a separate, testable module demonstrates clean separation of concerns in multi-tool LLM systems.

---

### LangSmith Observability

**What it does**
With `LANGCHAIN_TRACING_V2=true` set, every LangChain/LangGraph invocation automatically emits a structured trace to LangSmith. Traces capture inputs, outputs, latency, token counts, and the full execution graph for each run.

**Why it matters**
Debugging LLM pipelines without tracing is guesswork. LangSmith provides a production-grade observability layer that lets you inspect exactly which retrieval chunks were passed to the model, how long each node took, and where failures occurred.

---

### Streamlit UI

**What it does**
`streamlit_app.py` provides a single-page interface with a text input for queries and a response panel. It calls the LangGraph master agent directly and renders the response in real time.

**Why it matters**
A UI accelerates iteration. Rather than scripting test queries, you can explore routing behavior, compare responses, and demo the pipeline without any API client setup.

---

## Quick Setup

### Prerequisites

- Python 3.10+
- [Qdrant](https://qdrant.tech/documentation/quick-start/) installed locally
- API keys for Groq, OpenWeatherMap, and LangSmith

### 1. Clone and install

```bash
git clone https://github.com/ShubhamChougale01/RAG-pipeline.git
cd RAG-pipeline
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
QDRANT_URL=http://localhost:6333
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-pipeline
TOKENIZERS_PARALLELISM=false
```

### 3. Start Qdrant

```bash
qdrant --config-path ./qdrant_config.yaml
```

### 4. Index the PDF

Place your resume PDF in the project (update the path in `populate_collection.py` if needed), then run:

```bash
python populate_collection.py
```

This is a one-time step. Vectors are persisted by Qdrant and survive restarts.

### 5. Launch the app

```bash
streamlit run src/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Or run individual test modules:

```bash
python -m unittest tests.test_agent
python -m unittest tests.test_rag
python -m unittest tests.test_weather
```

| Test file | Covers |
|---|---|
| `tests/test_agent.py` | LangGraph routing logic and intent classification |
| `tests/test_rag.py` | Qdrant retrieval and RAG chain response quality |
| `tests/test_weather.py` | OpenWeatherMap API integration and response parsing |

---

## Folder Structure

```
RAG-pipeline/
├── src/
│   ├── master_agent.py      # LangGraph StateGraph — intent routing workflow
│   ├── rag_chain.py         # RAG pipeline: Qdrant retrieval + Groq LLM chain
│   ├── weather.py           # OpenWeatherMap API integration
│   ├── streamlit_app.py     # Streamlit UI entry point
│   └── __init__.py
├── tests/
│   ├── test_agent.py        # Unit tests for LangGraph routing
│   ├── test_rag.py          # Unit tests for RAG retrieval
│   └── test_weather.py      # Unit tests for weather agent
├── populate_collection.py   # One-time script: parse PDF and index into Qdrant
├── qdrant_config.yaml       # Qdrant local storage and collection config
├── requirements.txt         # Python dependencies
└── .env                     # API keys — never committed to version control
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key for Groq LLM inference |
| `OPENWEATHERMAP_API_KEY` | Yes | API key for current weather data |
| `QDRANT_URL` | Yes | URL of running Qdrant instance (e.g. `http://localhost:6333`) |
| `LANGSMITH_API_KEY` | Yes | API key for LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Yes | Set to `true` to enable LangSmith trace emission |
| `LANGCHAIN_PROJECT` | No | LangSmith project name for grouping traces (default: `default`) |
| `TOKENIZERS_PARALLELISM` | No | Set to `false` to suppress HuggingFace tokenizer warnings |

---

## FAQ

**Q: Can I swap the PDF for a different document?**
Replace the PDF file referenced in `populate_collection.py`, delete the existing Qdrant collection (or use a new collection name), and re-run `python populate_collection.py`. No other changes are needed.

**Q: Can I use a different LLM instead of Groq?**
Yes. `rag_chain.py` uses a standard LangChain `ChatGroq` instance. Replace it with any LangChain-compatible chat model (OpenAI, Anthropic, Ollama, etc.) by swapping the LLM object — the rest of the chain is provider-agnostic.

**Q: Can I use a cloud-hosted Qdrant instead of local?**
Yes. Update `QDRANT_URL` in your `.env` to point to your Qdrant Cloud cluster URL and add your Qdrant API key. Update the `QdrantClient` initialization in `rag_chain.py` and `populate_collection.py` accordingly.

**Q: How do I view traces in LangSmith?**
Log in at [smith.langchain.com](https://smith.langchain.com), navigate to your project (set via `LANGCHAIN_PROJECT`), and you will see a run for every query submitted through the app.

**Q: What happens if the master agent misroutes a query?**
The routing prompt in `master_agent.py` can be refined. You can also add a fallback node that returns a clarifying response when confidence is low, or extend the graph with an explicit "unknown intent" edge.

**Q: Do I need a GPU to run this?**
No. Embeddings use `all-MiniLM-L6-v2` via HuggingFace Sentence Transformers, which runs efficiently on CPU. LLM inference is fully offloaded to Groq's API.

---

## Tech Stack Summary

| Layer | Technology | Role |
|---|---|---|
| Orchestration | LangGraph | Stateful agentic routing graph |
| LLM | Groq (gemma2-9b-it) | Answer generation |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Document and query encoding |
| Vector Store | Qdrant | Persistent similarity search |
| Document Parsing | PyPDF2 | PDF text extraction |
| Observability | LangSmith | LLM trace collection and analysis |
| UI | Streamlit | Interactive query interface |
| External Data | OpenWeatherMap API | Real-time weather retrieval |
| Language | Python 3.10+ | Runtime |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Built to demonstrate production-grade LangGraph pipeline patterns — routing, retrieval, external APIs, and observability in one place.*
