python -m pip install torch
python -m pip install unsloth
import streamlit as st
import torch
from transformers import AutoTokenizer
import unsloth

# Hugging Face model details
MODEL_NAME = "Al-To1234/DocGPT"  # Replace with your actual model path

@st.cache_resource
def load_model():
    """Load the fine-tuned LoRA model using Unsloth."""
    model, tokenizer = unsloth.load_model(
        MODEL_NAME,
        max_seq_length=4096,  # Adjust based on your model's training
        dtype=torch.float16,   # Efficient inference
        load_in_4bit=True,     # Use 4-bit quantization (optional)
    )
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

st.title("DocGPT")
st.write("Ask anything")
# User input
prompt = st.text_area("Enter your prompt:")

temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=1.0, step=0.1)
max_length = st.slider("Max Length", min_value=50, max_value=500, value=200, step=10)

generate_button = st.button("Generate")

if generate_button and prompt:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move to GPU if available
    with torch.no_grad():
        outputs = model
