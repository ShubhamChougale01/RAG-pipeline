# AI Pipeline with LangGraph, LangChain, Qdrant & Streamlit

Here is the assignment project demonstrates how to implement intelligent AI pipelines using **LangGraph**, **LangChain**, **LangSmith**, **Qdrant**, and **Streamlit**. It supports routing user queries intelligently between a real-time weather agent and a RAG (Retrieval-Augmented Generation) system that answers questions based on a PDF (resume) document.

---

## Project Structure

```
RAG-pipeline/
├── src/
│   ├── master_agent.py       # LangGraph workflow for routing queries
│   ├── rag_chain.py          # RAG pipeline for resume using Qdrant and LangChain
│   ├── weather.py            # Fetches weather data using OpenWeatherMap API
│   └── __init__.py           
├── tests/
│   ├── test_agent.py         # Unit tests for master_agent.py
│   ├── test_rag.py           # Unit tests for rag_chain.py
│   ├── test_weather.py       # Unit tests for weather.py
│   └── __init__.py           
├── docs/
│   └── Shubham_re.pdf        # Resume PDF used for RAG
├── .env                      # Environment variables (API keys)
├── .gitignore                # Excludes sensitive files (e.g., .env, venv, PDFs)
├── qdrant_config.yaml        # Qdrant storage configuration
├── venv/                     # Virtual environment
├── requirements.txt          # Python dependencies
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ShubhamChougale01/RAG-pipeline.git
cd RAG-pipeline
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory and configure it as follows:

```env
GROQ_API_KEY=your_groq_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
QDRANT_URL=http://localhost:6333
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-pipeline
TOKENIZERS_PARALLELISM=false
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Qdrant Locally

Make sure you have Qdrant installed, then run:

```bash
qdrant --config-path ./qdrant_config.yaml
```

### 5. Launch the Streamlit UI

```bash
streamlit run src/streamlit_app.py
```

---

## Running Tests

Run the following unit tests to verify the system:

```bash
python -m unittest tests.test_agent
python -m unittest tests.test_rag
python -m unittest tests.test_weather
```

---

## Implementation Details

### LangGraph Pipeline

* A **master agent** built using LangGraph decides whether a query should be routed to the weather API or the RAG pipeline.
* **Weather Agent**: Uses OpenWeatherMap API to fetch current weather details.
* **RAG Agent**:

  * Embeds a PDF resume using HuggingFace models.
  * Stores the embeddings in **Qdrant**.
  * Uses **LangChain**'s `RetrievalQA` to answer resume-related queries.

### Components Used

* **LangChain**: Interfaces with LLM (ChatGroq using `gemma2-9b-it`) and manages chains.
* **Qdrant**: Efficient vector database to store and retrieve embedded documents.
* **LangSmith**: Tracks and evaluates the performance of LLM responses.
* **Streamlit**: Simple and interactive UI for user input and response display.

---

## LangSmith Evaluation

LangSmith tracing is enabled via environment variables to monitor LLM reasoning, performance, and accuracy. Screenshots demonstrating this are available in the `IMG/` directory.

---

## Test Coverage

* `test_agent.py`: Validates LangGraph routing logic for queries.
* `test_rag.py`: Validates resume RAG retrieval accuracy.
* `test_weather.py`: Validates weather API integration and response formatting.

