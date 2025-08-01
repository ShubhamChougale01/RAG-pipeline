import pytest
from src.rag import RAGPipeline

def test_rag_pipeline():
    rag = RAGPipeline()
    response = rag.query("What is the main objective of the paper?")
    assert isinstance(response, str)
    assert "generalization" in response.lower()