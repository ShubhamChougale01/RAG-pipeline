import pytest
from src.agent import create_workflow

def test_workflow():
    workflow = create_workflow()
    result = workflow.invoke({"query": "What is the main objective of the paper?", "response": "", "route": ""})
    assert isinstance(result["response"], str)
    assert "generalization" in result["response"].lower()

def test_workflow_weather():
    workflow = create_workflow()
    result = workflow.invoke({"query": "Weather in India", "response": "", "route": ""})
    assert isinstance(result["response"], str)
    assert "weather in India" in result["response"].lower()