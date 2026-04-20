import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# torch MUST be imported before transformers pipeline
import torch
import torch.nn  # force full torch initialization

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Load FinBERT manually (bypasses pipeline torch issue)
# Cached — loads only once per session
# ─────────────────────────────────────────────
@st.cache_resource
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()   # inference mode
    return tokenizer, model


# ─────────────────────────────────────────────
# Predict sentiment for a single headline
# ─────────────────────────────────────────────
def predict_single(headline, tokenizer, model):
    labels_map = {0: "positive", 1: "negative", 2: "neutral"}

    inputs = tokenizer(
        headline[:512],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs    = model(**inputs)
        probs      = torch.softmax(outputs.logits, dim=1)
        pred_idx   = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    label = labels_map.get(pred_idx, "neutral")
    return label, round(confidence, 3)


# ─────────────────────────────────────────────
# Analyze full news list using FinBERT
# ─────────────────────────────────────────────
def analyze_sentiment(news_list):
    if not news_list:
        return "Neutral", 0.0, []

    tokenizer, model = load_finbert()

    results  = []
    detailed = []

    for headline in news_list:
        try:
            label, confidence = predict_single(headline, tokenizer, model)
            results.append(label)
            detailed.append({
                "headline"   : headline,
                "label"      : label,
                "confidence" : confidence
            })
        except Exception:
            results.append("neutral")
            detailed.append({
                "headline"   : headline,
                "label"      : "neutral",
                "confidence" : 0.0
            })

    total = len(results)
    pos   = results.count("positive")
    neg   = results.count("negative")

    sentiment_score = (pos - neg) / total

    if sentiment_score > 0.1:
        overall = "Positive"
    elif sentiment_score < -0.1:
        overall = "Negative"
    else:
        overall = "Neutral"

    return overall, round(sentiment_score, 3), detailed