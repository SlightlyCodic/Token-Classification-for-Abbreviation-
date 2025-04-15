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
        # Color map for labels
        label_colors = {
            "B-AC": "#f39c12",   # orange
            "B-LF": "#27ae60",   # green
            "I-LF": "#2ecc71",   # light green
            "O": "#bdc3c7"       # gray
        }

        # Build styled sentence
        styled_sentence = ""
        for word, label in results:
            bg_color = label_colors.get(label, "#ffffff")
            styled_sentence += f"<span style='background-color: {bg_color}; padding: 4px 8px; border-radius: 6px; margin-right: 4px; color: black; font-weight: 500;'>{word}</span>"

        # Display the full sentence
        st.markdown(styled_sentence, unsafe_allow_html=True)
