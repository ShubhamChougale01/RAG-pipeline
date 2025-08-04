import os
import json
from groq import Groq
from dotenv import load_dotenv
from rag_chain import query_resume
from weather import get_weather
from langsmith import Client
from langchain.callbacks.tracers.langchain import LangChainTracer
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

tracer = LangChainTracer()
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langsmith_client = Client()

class AgentState(TypedDict):
    query: str
    tool: Literal["resume_rag_tool", "weather_agent_tool", "none"]
    result: str

def tool_router(state: AgentState) -> AgentState:
    query = state["query"]
    tool = "none"
    with LangChainTracer(project_name="ai-pipeline", client=langsmith_client) as tracer:
    try:
        prompt = f"""
        You are master agent. Your job is to analyze a user's query and decide which tool is most appropriate to answer it. You have two tools:

        1. 'resume_rag_tool': Use this for any question related to a person's professional history, skills, experience, or projects from their resume.
        2. 'weather_agent_tool': Use this for queries about weather information.

        For all other queries, return 'none'.

        Respond with a single JSON object containing a 'tool' key ('resume_rag_tool', 'weather_agent_tool', or 'none').

        User: {query}
        Response:
        """
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that decides which tool to use. You must always respond with a single JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        tool_choice = json.loads(response.choices[0].message.content)
        tool = tool_choice.get("tool", "none")
    except Exception as e:
        print(f"Failed to parse LLM response: {e}")
        tool = "none"

    return {"query": state["query"], "tool": tool, "result": ""}

def run_resume_rag(state: AgentState) -> AgentState:
    with LangChainTracer(project_name="ai-pipeline", client=langsmith_client) as tracer:
        try:
            result = query_resume(state["query"])
        except Exception as e:
            result = f"RAG Pipeline failed: {str(e)}"
        return {**state, "result": result}

def run_weather_agent(state: AgentState) -> AgentState:
    with LangChainTracer(project_name="ai-pipeline", client=langsmith_client) as tracer:
        try:
            result = get_weather(state["query"])
        except Exception as e:
            result = f"Location not found: {str(e)}"
        return {**state, "result": result}

def handle_unknown(state: AgentState) -> AgentState:
    with LangChainTracer(project_name="ai-pipeline", client=langsmith_client) as tracer:
        return {**state, "result": "We don't have access to this information."}

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", tool_router)
    builder.add_node("resume_rag_tool", run_resume_rag)
    builder.add_node("weather_agent_tool", run_weather_agent)
    builder.add_node("fallback", handle_unknown)

    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        lambda state: state["tool"],
        {
            "resume_rag_tool": "resume_rag_tool",
            "weather_agent_tool": "weather_agent_tool",
            "none": "fallback"
        }
    )
    builder.add_edge("resume_rag_tool", END)
    builder.add_edge("weather_agent_tool", END)
    builder.add_edge("fallback", END)

    return builder.compile()

def run_agent(query: str):
    graph = build_graph()
    with LangChainTracer(project_name="ai-pipeline", client=langsmith_client) as tracer:
        result = graph.invoke({"query": query, "tool": "", "result": ""})
    print(f"\n--- Final Answer ---\n{result['result']}\n")
    return result["result"]