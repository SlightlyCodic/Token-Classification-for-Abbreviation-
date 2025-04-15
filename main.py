import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model and tokenizer
model_name = "slightlycodic/TC-ABB-BERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
label_list = ['O', 'B-AC', 'B-LF', 'I-LF']  # update if different

# Title and UI
st.set_page_config(page_title="Abbreviation & Long-Form Detection", layout="wide")
st.title("🧠 Abbreviation & Long-Form Detector (NER)")
st.markdown("Detects abbreviations (AC) and their long forms (LF) in medical or technical text using a BERT model.")

# User input
text_input = st.text_area("Enter a sentence:", "MRR, mortality rate ratio; TBI, traumatic brain injury.")

# Predict button
if st.button("Detect Entities"):
    if text_input.strip():
        words = text_input.strip().split()

        # Tokenize
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )
        inputs = {k: v for k, v in encoding.items() if k != "offset_mapping"}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        word_ids = encoding.word_ids()
        predicted_labels = predictions[0].tolist()

        seen = set()
        results = []
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            label = label_list[predicted_labels[idx]]
            word = words[word_id]
            results.append((word, label))
            seen.add(word_id)

        # Display results
        for word, label in results:
            st.markdown(f"<span style='background-color: #f0f0f0; padding: 5px; border-radius: 5px;'>{word}</span> → <span style='color: #007acc; font-weight: bold'>{label}</span>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a sentence.")
