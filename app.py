import streamlit as st
import requests

st.set_page_config(page_title="Custom LLM Chatbot", page_icon="🤖")

st.title("🚀 Custom LLM Chatbot")
st.markdown("Enter a prompt and get AI-generated responses!")

prompt = st.text_area("Enter your prompt:")

if st.button("Generate Response"):
    if prompt:
        api_url = "http://your-ngrok-url/generate"  # Replace with actual ngrok URL
        response = requests.post(api_url, json={"prompt": prompt})
        if response.status_code == 200:
            st.write("### AI Response:")
            st.write(response.json()["response"])
        else:
            st.error("Error: Could not fetch response")
