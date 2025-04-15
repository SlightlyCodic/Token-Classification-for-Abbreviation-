import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model and tokenizer
model_name = "slightlycodic/TC-ABB-BERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Label list used by the model
label_list = ['O', 'B-AC', 'B-LF', 'I-LF']

# Set up the Streamlit app
st.set_page_config(page_title="Abbreviation Detector", layout="wide")
st.title("🧠 Abbreviation & Long-Form Detector")
st.markdown("Detect abbreviations (AC) and their long forms (LF) in biomedical or technical sentences using a fine-tuned BERT model.")

# Text input
text_input = st.text_area("Enter a sentence:", "The patient was diagnosed with COPD, which stands for chronic obstructive pulmonary disease.")

# Color map
label_colors = {
    "B-AC": "#f39c12",   # orange
    "B-LF": "#27ae60",   # green
    "I-LF": "#2ecc71",   # light green
    "O": None            # plain
}

# On button click
if st.button("🔍 Detect Entities"):
    if not text_input.strip():
        st.warning("Please enter a sentence.")
    else:
        # Preprocess
        words = text_input.strip().split()

        # Tokenize with alignment
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )
        inputs = {k: v for k, v in encoding.items() if k != "offset_mapping"}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        word_ids = encoding.word_ids()
        predicted_labels = predictions[0].tolist()

        results = []
        seen = set()
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            label = label_list[predicted_labels[idx]]
            word = words[word_id]
            results.append((word, label))
            seen.add(word_id)

        # Build HTML output
        styled_sentence = ""
        for word, label in results:
            if label == "O":
                styled_sentence += f"<span style='margin-right: 6px;'>{word}</span>"
            else:
                color = label_colors[label]
                styled_sentence += f"""
                <span style='margin: 4px; display: inline-block; vertical-align: middle;'>
                    <span style='background-color: {color}; border-radius: 8px; padding: 6px 10px;
                                display: inline-flex; align-items: center; gap: 6px;'>
                        <span style='font-weight: 500;'>{word}</span>
                        <span style='background-color: rgba(0,0,0,0.1); border-radius: 4px; padding: 2px 6px;
                                     font-size: 11px; font-weight: bold;'>{label}</span>
                    </span>
                </span>
                """

        # Show output
        st.markdown("### 🧾 Tagged Sentence")
        st.markdown(styled_sentence, unsafe_allow_html=True)
