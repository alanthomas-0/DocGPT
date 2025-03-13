import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face model details
MODEL_NAME = "your-huggingface-username/your-finetuned-model"  # Replace with your model

@st.cache_resource
def load_model():
    """Load fine-tuned LoRA model (No Unsloth needed)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

st.title("Fine-Tuned LLM Web App")
st.write("Enter a prompt, and the model will generate a response.")

# User input
prompt = st.text_area("Enter your prompt:")

temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=1.0, step=0.1)
max_length = st.slider("Max Length", min_value=50, max_value=500, value=200, step=10)

generate_button = st.button("Generate")

if generate_button and prompt:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move to GPU if available
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, temperature=temperature)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.subheader("Model Response:")
    st.write(response)
