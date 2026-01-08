import numpy as np
import re
import joblib
from scipy.sparse import hstack
import streamlit as st


# for caching the loaded models so it needs to be trained only once
@st.cache_resource
def load_models():
    clf_rf = joblib.load("rf_classifier.pkl")
    reg_rf = joblib.load("rf_regressor.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return clf_rf, reg_rf, vectorizer, scaler, le

clf_rf, reg_rf, vectorizer, scaler, le = load_models()


#feature engineering

KEYWORDS_HARD = ["graph", "tree", "dp", "segment", "dijkstra", "modulo", "query"]
KEYWORDS_EASY = ["print", "sum", "average", "even", "odd", "brute"]
KEYWORDS_OPTIM = ["minimize", "maximize", "optimal"]

# extracts the constraint from input description. 
def extract_max_constraint(text):
    text = str(text)
    powers = re.findall(r'10\^(\d+)|10\*\*(\d+)', text)
    large_nums = re.findall(r'\b[1-9](0{3,})\b', text)

    found = []
    for p in powers:
        val = p[0] if p[0] else p[1]
        if val:
            found.append(int(val))
    for num in large_nums:
        found.append(len(num) - 1)

    return max(found) if found else 0

def count_keywords(text, keywords):
    text = text.lower()
    return sum(text.count(k) for k in keywords)


# models for prediction
def predict_difficulty(description, input_text, output_text):
    full_text = f" {description} {input_text} {output_text}"

    # Recreate engineered features
    f_constraint = extract_max_constraint(input_text)
    f_hard = count_keywords(full_text, KEYWORDS_HARD)
    f_easy = count_keywords(full_text, KEYWORDS_EASY)
    f_optim = count_keywords(full_text, KEYWORDS_OPTIM)
    f_words = len(full_text.split())
    f_symbols = len(re.findall(r"[+\-*/%=<>]", full_text))
    f_density = f_symbols / max(1, f_words)

    # Vectorize text
    X_text_new = vectorizer.transform([full_text])

    # Scale engineered features
    X_eng_new = np.array([[f_words, f_symbols, f_density,
                           f_constraint, f_hard, f_easy, f_optim]])
    X_eng_new = scaler.transform(X_eng_new)

    # Combine features
    X_input = hstack([X_text_new, X_eng_new])

    # Predict
    class_idx = clf_rf.predict(X_input)[0]
    predicted_class = le.inverse_transform([class_idx])[0]
    predicted_score = reg_rf.predict(X_input)[0]

    return predicted_class, predicted_score
