import streamlit as st
from master_agent import run_master_agent, get_tool_choice
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

st.set_page_config(page_title="AI Pipeline Demo", page_icon="🤖", layout="centered")
st.title("AI Pipeline Demo")
st.markdown("""
AI Pipeline Demo
- Ask about Shubham's resume or the weather in any city.   
""")

query = st.text_input("Enter your query:", placeholder="e.g., 'Weather in India' or 'What is Shubham's experience?'")

if st.button("Submit", key="submit_button"):
    if query:
        with st.spinner("Processing your query..."):
            tool = get_tool_choice(query)
            response = run_master_agent(query)
        st.subheader("Result")
        st.write(f"**Selected Tool**: {tool}")
        st.write(f"**Response**: {response}")
    else:
        st.error("Please enter a query.")
st.markdown("---")