import streamlit as st
import requests
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import seaborn as sn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def default_paraphrase_models():
    return {
        "QLORA": r"C:/Users/LENOVO/Desktop/Detect AI/Paraphrase/Qlora",
        "Pegasus":  r"C:/Users/LENOVO/Desktop/Detect AI/Paraphrase/pegasus_paraphrase",
        "T5 Paraphraser": r"C:/Users/LENOVO/Desktop/Detect AI/Paraphrase/t5_paraphrase"
    }

def default_detection_models():
    return {
        "BART Detector": {
          "hf_name": "facebook/bart-base",
          "ckpt": r"C:/Users/LENOVO/Desktop/Detect AI/Final_Databases/bart_finetuned_downsampled.pt"
        },
        "BERT Detector": {
          "hf_name": "bert-base-uncased",
          "ckpt": r"C:/Users/LENOVO/Desktop/Detect AI/Final_Databases/bert_finetuned_downsampled.pt"
        },
        "RoBERTA Detector": {
          "hf_name": "roberta-base",
          "ckpt": r"C:/Users/LENOVO/Desktop/Detect AI/Final_Databases/roberta_finetuned_downsampled.pt"
        },
        "LoRA Detector": {
          "hf_name": "bert-base-uncased",
          "ckpt": r"C:/Users/LENOVO/Desktop/Detect AI/LAURA/lora_bert_model"
        },
    }


embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

@st.cache_resource
def load_paraphrase_model(name):
    model_paths = default_paraphrase_models()
    st.write(f"Loading model for {name}")
    tok = T5Tokenizer.from_pretrained(model_paths[name])
    mod = T5ForConditionalGeneration.from_pretrained(model_paths[name]).to(DEVICE)
    return tok, mod



@st.cache_resource
def load_detection_model(name):
    cfg = default_detection_models()[name]
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_name"], num_labels=2
    )
    state = torch.load(cfg["ckpt"], map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def detect_ai(text, tokenizer, model, max_len):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    label_idx = int(probs.argmax())
    label_map = {0: "Human", 1: "AI"}
    return label_map[label_idx], float(probs[label_idx])




def paraphrase_text(text, tokenizer, model, max_len, num_return):
    input_ids = tokenizer.encode(
        "paraphrase: " + text,
        return_tensors="pt",
        max_length=max_len,
        truncation=True
    ).to(DEVICE)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_len,
        num_beams=10,
        num_return_sequences=num_return,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=1.5
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def detect_ai(text, tokenizer, model, max_len):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    label_idx = int(probs.argmax())
    raw_label = model.config.id2label[label_idx]
    if raw_label == "LABEL_0":
        label = "Human"
    else:
        label = "AI"
    
    confidence = probs[label_idx] * 100
    
    return label, confidence





def make_map(sentence, decoding_params, tokenizer, model):
    paraphrases = forward(sentence, decoding_params, tokenizer, model)
    all_sents = [sentence] + paraphrases
    embeddings = embedder.encode(all_sents, convert_to_numpy=True)
    return embeddings, all_sents


def plot_similarity(labels, features, rotation=90):
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd"
    )
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    st.pyplot()


def main():
    st.title("AI Text Toolkit")
    st.sidebar.header("Mode Selection")
    mode = st.sidebar.radio("Choose an action:", ["Paraphrase", "Detect AI"])

    if mode == "Paraphrase":
        st.header("Paraphrase Generator")
        text = st.text_area("Enter text to paraphrase:")
        para_models = default_paraphrase_models()
        model_choice = st.sidebar.selectbox("Tokenizer/Model", list(para_models.keys()))
        max_len = st.sidebar.slider("Max Output Length", 10, 512, 256)
        num_return = st.sidebar.slider("Number of paraphrases", 1, 10, 5)

        tokenizer, model = load_paraphrase_model(model_choice)
        if st.button("Generate Paraphrases"):
            if not text.strip():
                st.error("Please enter some text.")
            else:
                with st.spinner("Generating..."):
                    results = paraphrase_text(text, tokenizer, model, max_len, num_return)
                    for i, res in enumerate(results, 1):
                        st.write(f"{i}. {res}")

    else:
        st.header("AI-generated Text Detection")
        text = st.text_area("Enter text to classify:")
        det_models = list(default_detection_models().keys())
        model_choice = st.sidebar.selectbox("Detection Model", det_models)
        max_len = st.sidebar.slider("Max Words / Sequence Length", 10, 1024, 512)

        tokenizer, model = load_detection_model(model_choice)
        if st.button("Detect AI"):
            if not text.strip():
                st.error("Please enter some text.")
            else:
                with st.spinner("Detecting..."):
                    label, conf = detect_ai(text, tokenizer, model, max_len)
                    st.write(f"**Prediction:** {label}")
                    st.write(f"**Confidence:** {conf:.2f}%")

if __name__ == "__main__":
    main()
