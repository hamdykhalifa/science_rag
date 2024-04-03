import streamlit as st
import requests

# The URL of your FastAPI backend endpoint
BACKEND_URL = "http://localhost:8000/ask-chatbot"

st.title("Chatbot Interface")

# User input
question = st.text_input("Type your question here:")

if st.button("Ask"):
    if question:
        # Send the question to the backend API
        response = requests.post(BACKEND_URL, json={"question": question})
        
        if response.status_code == 200:
            # Display the answer
            answer = response.json().get("response")
            st.write(answer)
        else:
            st.error("An error occurred while contacting the backend API.")
    else:
        st.error("Please type a question before clicking ask.")
