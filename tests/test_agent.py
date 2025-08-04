import unittest
from src.master_agent import get_tool_choice, run_master_agent

class TestMasterAgent(unittest.TestCase):
    def test_resume_query(self):
        query = "What is Shubham's experience?"
        tool = get_tool_choice(query)
        print(f"Selected Tool for '{query}': {tool}")
        self.assertEqual(tool, "resume_rag_tool", f"Expected 'resume_rag_tool' for '{query}', got '{tool}'")
        response = run_master_agent(query)
        print(f"Assistant Response: {response}")
        self.assertTrue(len(response) > 0, "Response should not be empty")

    def test_weather_query(self):
        query = "What's the weather in Tokyo?"
        tool = get_tool_choice(query)
        print(f"Selected Tool for '{query}': {tool}")
        self.assertEqual(tool, "weather_agent_tool", f"Expected 'weather_agent_tool' for '{query}', got '{tool}'")
        response = run_master_agent(query)
        print(f"Assistant Response: {response}")
        self.assertTrue(len(response) > 0, "Response should not be empty")

    def test_unrelated_query(self):
        query = "Who is the president of the USA?"
        tool = get_tool_choice(query)
        print(f"Selected Tool for '{query}': {tool}")
        self.assertEqual(tool, "none", f"Expected 'none' for '{query}', got '{tool}'")
        response = run_master_agent(query)
        print(f"Assistant Response: {response}")
        self.assertEqual(response, "We don't have access to this information.", 
                         f"Expected fallback message for '{query}', got '{response}'")

if __name__ == '__main__':
    unittest.main()