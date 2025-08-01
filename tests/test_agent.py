import pytest
from src.agent import create_workflow
from langchain_core.messages import HumanMessage

def test_agent_workflow_weather():
    workflow = create_workflow()
    response = workflow.invoke({"messages": [HumanMessage(content="Weather in India")]})
    assert "Weather in India" in response["messages"][-1].content
    assert "Temperature" in response["messages"][-1].content

def test_workflow():
    workflow = create_workflow()
    result = workflow.invoke({"query": "What is the main objective of the paper?", "response": "", "route": ""})
    assert isinstance(result["response"], str)
    assert "generalization" in result["response"].lower()