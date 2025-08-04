import unittest
from src.weather import get_weather

def test_weather_query():
    workflow = get_weather()
    result = workflow.invoke({"query": "Weather in India", "response": "", "route": ""})
    assert isinstance(result["response"], str)
    assert "weather in India" in result["response"].lower()

if __name__ == '__main__':
    unittest.main()