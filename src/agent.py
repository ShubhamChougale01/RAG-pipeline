from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag import RAGPipeline
import requests
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

class AgentState(TypedDict):
    query: str
    response: str
    route: str

def create_workflow():
    llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))
    rag_pipeline = RAGPipeline()

    def route_query(state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_template(
            "Determine if the query is about weather or something else. "
            "If it's about weather, return 'weather'. Otherwise, return 'rag'. "
            "Query: {query}"
        )
        chain = prompt | llm | StrOutputParser()
        decision = chain.invoke({"query": state["query"]})
        return {"query": state["query"], "response": state["response"], "route": decision.strip().lower()}

    def fetch_weather(state: AgentState) -> AgentState:
        try:
            api_key = os.getenv("OPENWEATHERMAP_API_KEY")
            city = state["query"].lower().replace("weather in", "").strip()
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            weather = f"Weather in {city.capitalize()}: {data['weather'][0]['description']}, Temperature: {data['main']['temp']}°C"
            return {"query": state["query"], "response": weather, "route": state["route"]}
        except Exception as e:
            return {"query": state["query"], "response": f"Error fetching weather: {str(e)}", "route": state["route"]}

    def query_rag(state: AgentState) -> AgentState:
        try:
            response = rag_pipeline.query(state["query"])
            return {"query": state["query"], "response": response, "route": state["route"]}
        except Exception as e:
            return {"query": state["query"], "response": f"Error querying RAG: {str(e)}", "route": state["route"]}

    workflow = StateGraph(AgentState)
    workflow.add_node("route_query", route_query)
    workflow.add_node("fetch_weather", fetch_weather)
    workflow.add_node("query_rag", query_rag)
    workflow.set_entry_point("route_query")
    workflow.add_conditional_edges(
        "route_query",
        lambda state: state["route"],
        {"weather": "fetch_weather", "rag": "query_rag"}
    )
    workflow.add_edge("fetch_weather", END)
    workflow.add_edge("query_rag", END)
    return workflow.compile()