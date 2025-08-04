AI Pipeline Assignment
This project implements an AI pipeline using LangChain, LangGraph, LangSmith, Qdrant, and Streamlit to fetch real-time weather data and answer questions from a PDF using RAG.
Setup Instructions

Clone the Repository:
git clone https://github.com/yourusername/ai-pipeline-assignment.git
cd ai-pipeline-assignment


Set Up Environment:

Copy .env.example to .env and add your API keys:cp .env.example .env

Update .env with:GROQ_API_KEY=your_groq_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
QDRANT_URL=http://localhost:6333
LANGSMITH_API_KEY=your_langsmith_api_key




Install Dependencies:
pip install -r requirements.txt


Run Qdrant Locally:
docker run -p 6333:6333 qdrant/qdrant


Run Streamlit UI:
streamlit run src/streamlit_app.py


Run Tests:
pytest tests/



Implementation Details

LangGraph Pipeline: 
A decision node routes queries to either the weather API or RAG pipeline.
Weather node fetches data using OpenWeatherMap API.
RAG node processes PDF content, stores embeddings in Qdrant, and retrieves answers.


LangChain: Uses ChatGroq with gemma2-9b-it for LLM processing.
Qdrant: Stores PDF embeddings for efficient retrieval.
LangSmith: Evaluates LLM responses (see langsmith_screenshot.png).
Streamlit: Provides a chat interface for user interaction.
Tests: Unit tests cover API handling, LLM processing, and retrieval logic.

LangSmith Evaluation

LangSmith traces are enabled to monitor and evaluate LLM performance.
See langsmith_screenshot.png for sample evaluation logs.

Loom Video

A Loom video explaining the code and LangSmith results is available at: [Insert Loom URL].



----------
qd - curl -L -O https://github.com/qdrant/qdrant/releases/download/v1.13.4/qdrant-aarch64-apple-darwin.tar.gz

-- python3 -m unittest tests/test_rag.py
-- to check collection - curl http://localhost:6333/collections
