AI Pipeline Implementation 
In this project we have created AI pipelines using LangChain, LangGraph, LangSmith, Qdrant, and Streamlitt to fetch real-time weather data and answer questions from a PDF using RAG.

Project Structure

RAG-pipeline/

├── src/

│   ├── master_agent.py       # LangGraph workflow for routing queries

│   ├── rag_chain.py          # RAG pipeline for resume using Qdrant and LangChain

│   ├── weather.py            # Fetches weather data using OpenWeatherMap API

│   ├── __init__.py           

├── tests/

│   ├── test_agent.py         # Unit tests for master_agent.py

│   ├── test_rag.py           # Unit tests for rag_chain.py

│   ├── test_weather.py       # Unit tests for weather.py

│   ├── __init__.py           

├── docs/

│   ├── Shubham_re.pdf        # Resume PDF 

├── .env                      # Environment variables (API keys)

├── .gitignore                # Excludes sensitive files (e.g., .env, venv, PDFs)

├── qdrant_config.yaml        # Qdrant storage configuration

├── venv/                     # Virtual environment

├── requirements.txt          # Python dependencies

Setup Instructions

Clone the Repository:
git clone https://github.com/ShubhamChougale01/RAG-pipeline.git
cd RAG-pipeline


Set Up Environment:
Update .env with:
GROQ_API_KEY=your_groq_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
QDRANT_URL=http://localhost:6333
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-pipeline
TOKENIZERS_PARALLELISM=false

Install Dependencies:
pip install -r requirements.txt


Run Qdrant Locally:
qdrant --storage-path ~/qdrant_storage - to start the server


Run streamlit UI:
streamlit run src/streamlit_app.py

Run Tests:
python3 -m unittest tests.test_agent
python3 -m unittest tests.test_rag
python3 -m unittest tests.test_weather



Implementation Details

LangGraph Pipeline: 
A decision node routes queries to either the weather API or RAG pipeline.
Weather node fetches data using OpenWeatherMap API.
RAG node processes the resume PDF content, stores embeddings in Qdrant, and retrieves answers.


LangChain: Uses ChatGroq with gemma2-9b-it for LLM processing.
Qdrant: Stores PDF embeddings for efficient retrieval.
LangSmith: Evaluates LLM responses (check langsmith_screenshot.png).
Streamlit: Provides a chat interface for ui.
Tests: Unit tests cover API handling, LLM processing, and retrieval logic.

LangSmith Evaluation
LangSmith traces are enabled to monitor and evaluate LLM performance.


I have added some Screenshots in IMG folder - 
