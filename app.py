import streamlit as st
from model_backend import predict_difficulty

st.set_page_config(page_title="Problem Difficulty Predictor", layout="centered")

st.title("🧠 Problem Difficulty Predictor")

desc = st.text_area("Problem Description")
inp = st.text_area("Input Description")
out = st.text_area("Output Description")

if st.button("Predict"):
    if desc.strip() == "" and inp.strip() == "" and out.strip() == "":
        st.warning("Please enter some problem text before predicting.")
    else:
        pred_class, pred_score = predict_difficulty(desc, inp, out)
        st.success(f"Predicted Difficulty Class: {pred_class}")
        st.info(f"Predicted Difficulty Score: {pred_score:.2f}")
