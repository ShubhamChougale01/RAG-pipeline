import streamlit as st
from agent import create_workflow
from dotenv import load_dotenv
import warnings

# Suppress Pydantic deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_huggingface")

# Load environment variables
load_dotenv()

st.title("AI Pipeline Demo")

workflow = create_workflow()

query = st.text_input("Enter your query (e.g., 'Weather in Tokyo' or 'What is cross-view goal specification?')")
if st.button("Submit"):
    if query:
        result = workflow.invoke({"query": query, "response": "", "route": ""})
        st.write("**Response**:")
        st.write(result["response"])
    else:
        st.write("Please enter a query.")