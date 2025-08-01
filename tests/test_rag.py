import pytest
from src.rag import RAGPipeline

def test_rag_pipeline():
    rag = RAGPipeline()
    response = rag.query("What is the main topic of the document?")
    assert isinstance(response, str)
    assert len(response) > 0