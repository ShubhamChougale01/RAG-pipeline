import pytest
from src.agent import create_workflow

def test_weather_query():
    workflow = create_workflow()
    result = workflow.invoke({"query": "Weather in India", "response": "", "route": ""})
    assert isinstance(result["response"], str)
    assert "weather in India" in result["response"].lower()