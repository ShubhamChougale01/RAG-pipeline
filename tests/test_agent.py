import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from agent import create_workflow 

def test_weather_query():
    app = create_workflow()
    result = app.invoke({"query": "What's the weather in Mumbai?", "response": "", "route": ""})
    print(result)

def test_rag_query():
    app = create_workflow()
    result = app.invoke({"query": "Explain the Transformer architecture.", "response": "", "route": ""})
    print(result)

if __name__ == "__main__":
    test_weather_query()
    test_rag_query()
