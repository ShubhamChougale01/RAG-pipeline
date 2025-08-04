import os
import json
from groq import Groq
from rag_chain import query_resume
from weather import get_weather
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_tool_choice(query: str) -> str:
    prompt = f"""
    You are a master agent. Your job is to analyze a user's query and decide which tool is most appropriate to answer it. You have two tools:

    1. 'resume_rag_tool': Use this for any question related to a person's professional history, skills, experience, or projects from their resume. (For privacy reasons, I won't provide personal contact information like LinkedIn, Portfolio or phone number here.)
    2. 'weather_agent_tool': Use this for queries about weather information, such as current conditions or forecasts.

    For all other queries (e.g., general knowledge, unrelated topics), return 'none'.

    Respond with a single JSON object containing a 'tool' key with the name of the tool to be used ('resume_rag_tool', 'weather_agent_tool', or 'none'). DO NOT include any other text or explanation.

    User: {query}
    Response:
    """
    try:
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
        tool = tool_choice.get("tool")
        if tool not in ["resume_rag_tool", "weather_agent_tool", "none"]:
            return "none" 
        return tool
    except Exception as e:
        print(f"Router Error: {e}")
        return "none" 


def run_master_agent(query: str) -> str:
    print(f"\n--- Query: {query} ---")
    tool = get_tool_choice(query)
    print(f"→ Master Agent chose: '{tool}'")

    if tool == "resume_rag_tool":
        result = query_resume(query)
    elif tool == "weather_agent_tool":
        result = get_weather(query)
    else:
        result = "We don't have access to this information."

    print(f"\nFinal Answer:\n{result}")
    return result