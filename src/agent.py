from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.weather import get_weather
from src.rag import RAGPipeline
from typing import Dict, Any
import os

class AgentState(MessagesState):
    response: str

def decision_node(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1].content.lower()
    if "weather" in last_message:
        return {"response": "weather"}
    return {"response": "rag"}

def weather_node(state: AgentState) -> Dict[str, Any]:
    city = state["messages"][-1].content.split("weather in")[-1].strip()
    weather_data = get_weather(city)
    llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))
    prompt = ChatPromptTemplate.from_template(
        "Summarize this weather data: {data}"
    )
    chain = prompt | llm
    response = chain.invoke({"data": weather_data})
    return {"messages": [response]}

def rag_node(state: AgentState) -> Dict[str, Any]:
    query = state["messages"][-1].content
    rag = RAGPipeline()
    response = rag.query(query)
    return {"messages": [{"role": "assistant", "content": response}]}

def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("decision", decision_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("rag", rag_node)
    workflow.add_edge(START, "decision")
    workflow.add_conditional_edges(
        "decision",
        lambda state: state["response"],
        {"weather": "weather", "rag": "rag"}
    )
    workflow.add_edge("weather", END)
    workflow.add_edge("rag", END)
    return workflow.compile()