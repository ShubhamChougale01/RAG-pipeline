import os
import unittest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from master_agent import run_agent, tool_router

class TestLangGraphAgent(unittest.TestCase):
    def test_resume_query(self):
        query = "What projects has Shubham worked on?"
        routed = tool_router({"query": query, "tool": "", "result": ""})
        tool = routed["tool"]
        print(f"[Resume Test] Selected Tool: {tool}")
        self.assertEqual(tool, "resume_rag_tool")

        response = run_agent(query)
        print(f"[Resume Test] Response: {response}")
        self.assertTrue(len(response) > 0)

    def test_weather_query(self):
        query = "What's the weather in Mumbai?"
        routed = tool_router({"query": query, "tool": "", "result": ""})
        tool = routed["tool"]
        print(f"[Weather Test] Selected Tool: {tool}")
        self.assertEqual(tool, "weather_agent_tool")

        response = run_agent(query)
        print(f"[Weather Test] Response: {response}")
        self.assertTrue(len(response) > 0)

    def test_unknown_query(self):
        query = "Who is the president of India?"
        routed = tool_router({"query": query, "tool": "", "result": ""})
        tool = routed["tool"]
        print(f"[Unknown Test] Selected Tool: {tool}")
        self.assertEqual(tool, "none")

        response = run_agent(query)
        print(f"[Unknown Test] Response: {response}")
        self.assertEqual(response, "We don't have access to this information.")

if __name__ == "__main__":
    unittest.main()
