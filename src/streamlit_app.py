import streamlit as st
from src.agent import create_workflow
from langchain_core.messages import HumanMessage

st.title("AI Pipeline Demo")
st.write("Ask about the weather or query the PDF document!")

user_query = st.text_input("Enter your question:", value="")
if st.button("Get Response"):
    with st.spinner("Generating response..."):
        try:
            workflow = create_workflow()
            response = workflow.invoke({"messages": [HumanMessage(content=user_query)]})
            st.success("Response:")
            st.write(response["messages"][-1].content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")