import unittest
from src.rag_chain import query_resume

class TestRAG(unittest.TestCase):
    def test_query_resume(self):
        query = "What is Shubham's experience?"
        response = query_resume(query)
        print("RAG Response:", response)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()