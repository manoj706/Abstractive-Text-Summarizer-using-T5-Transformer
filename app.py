import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model():
    model_path = "t5_model"   # folder from your notebook
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device

tokenizer, model, device = load_model()

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

st.title("📌 Abstractive Text Summarizer using T5")
st.write("Enter long text and get an AI-generated summary.")

input_text = st.text_area("✍️ Enter text to summarize:", height=300)

max_len = st.slider("📏 Summary Length:", 30, 200, 100)

if st.button("🚀 Summarize"):
    if len(input_text.strip()) < 10:
        st.warning("Please enter a longer text.")
    else:
        with st.spinner("Generating summary... ⏳"):
            # Prepare input for model
            text = "summarize: " + input_text
            inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)

            # Generate
            summary_ids = model.generate(
                inputs,
                max_length=max_len,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.success("✨ Summary Generated:")
        st.write(summary)
