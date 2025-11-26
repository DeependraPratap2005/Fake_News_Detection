# ---------------------------------------------------------
# app.py — Final Glass + Light UI Fake News Detection App
# Developed by: Deependra Pratap Singh
# ---------------------------------------------------------

import streamlit as st
import pickle
import os
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests, json

# ---------------------------
# Configuration (UPDATED FOR DEPLOYMENT)
# ---------------------------
MODEL_PATH = "news_model.pkl"
VECTORIZER_PATH = "tfidf.pkl"

DEFAULT_FAKE = "Fake.csv"
DEFAULT_TRUE = "True.csv"

st.set_page_config(
    page_title="Fake News Detection — Glass UI",
    layout="wide",
    page_icon="📰"
)

# ---------------------------
# GLASS + LIGHT UI CSS
# ---------------------------
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #ffffff 0%, #e8f4ff 40%, #d9ecff 100%);
    background-size: cover;
    background-attachment: fixed;
}

.block-container {
    background: rgba(255, 255, 255, 0.6) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 20px;
    backdrop-filter: blur(12px) saturate(180%);
    box-shadow: 0 15px 30px rgba(0,0,0,0.08);
    padding: 35px !important;
}

.big-title {
    color: #003b5c;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 1px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.18);
    margin-bottom: 8px;
}

.sidebar-name {
    position: fixed;
    bottom: 12px;
    left: 15px;
    font-size: 14px;
    color: #002d47;
    font-weight: 800;
    background: rgba(255,255,255,0.7);
    padding:6px 10px;
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.12);
}

.footer { display: none !important; }

.stButton>button {
    background: linear-gradient(90deg,#009dff,#4bb8ff);
    border: none;
    color: white;
    font-weight: 800;
    padding: 10px 22px;
    border-radius: 10px;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,150,255,0.28);
}

.stRadio > div { gap: 12px !important; }
.stRadio label {
    background: rgba(255,255,255,0.8);
    color: #003b5c;
    padding: 12px;
    font-weight: 700;
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.15);
    transition: 0.25s;
    text-align: center;
}
.stRadio input:checked + label {
    background: linear-gradient(90deg, #009dff, #4bb8ff);
    color: white !important;
    font-weight: 800;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ("Home", "Predict Single", "Bulk Predict", "About"))

st.sidebar.markdown(
    "<div class='sidebar-name'>🔹 Developed by <b>Deependra Pratap Singh</b></div>",
    unsafe_allow_html=True
)

# ---------------------------
# Load Model + Vectorizer  (UPDATED)
# ---------------------------
model, vectorizer = None, None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except:
    model = None

try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except:
    vectorizer = None

# ---------------------------
# Cleaning Function
# ---------------------------
def clean_text(text):
    if pd.isna(text): 
        return ""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+","",t)
    t = re.sub(r"[^a-zA-Z\s]"," ",t)
    t = re.sub(r"\s+"," ",t).strip()
    return t

# ---------------------------
# Translation Function
# ---------------------------
def translate_to_english(text):
    try:
        url = (
            "https://translate.googleapis.com/translate_a/single"
            "?client=gtx&sl=auto&tl=en&dt=t&q=" + text
        )
        response = requests.get(url)
        result = json.loads(response.text)
        return result[0][0][0]
    except:
        return text

# ----------------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------------
if page == "Home":
    st.markdown("<div class='big-title'>📰 Fake News Detection</div>", unsafe_allow_html=True)
    st.write("""
        Detect Fake News using Machine Learning with TF-IDF + Logistic Regression/SVM.
        Supports **Hindi + 100 languages**.
    """)

    st.info("Model status:")
    if model and vectorizer:
        st.success("Model & Vectorizer Loaded Successfully ✔")
    else:
        st.warning("Model missing — Train or upload first.")

# ----------------------------------------------------------
# PAGE 2: Predict Single
# ----------------------------------------------------------
elif page == "Predict Single":
    st.markdown("<div class='big-title'>🔎 Predict Single News</div>", unsafe_allow_html=True)

    text = st.text_area("Enter News Text (Any Language):", height=220)

    if st.button("Predict"):
        if not text:
            st.warning("Please enter news text.")
        elif not model or not vectorizer:
            st.error("Model or vectorizer missing.")
        else:
            translated = translate_to_english(text)
            cleaned = clean_text(translated)
            vect = vectorizer.transform([cleaned])
            pred = model.predict(vect)[0]

            st.subheader("🔍 Processed Text")
            st.write("**Translated to English:**", translated)
            st.write("**Cleaned Text:**", cleaned)

            st.subheader("📢 Prediction Result")
            if pred == 1:
                st.success("✔ REAL NEWS")
            else:
                st.error("✖ FAKE NEWS")

# ----------------------------------------------------------
# PAGE 3: Bulk Predict
# ----------------------------------------------------------
elif page == "Bulk Predict":
    st.markdown("<div class='big-title'>📂 Bulk Prediction</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV:", type=["csv"])
    if uploaded:

        df = pd.read_csv(uploaded, encoding="latin1", on_bad_lines="skip")

        text_cols = df.select_dtypes(include='object').columns.tolist()

        if not text_cols:
            st.error("No text column found.")
        else:
            col = st.selectbox("Select text column:", text_cols)

            if st.button("Run Prediction"):
                if not model or not vectorizer:
                    st.error("Model or vectorizer missing.")
                else:
                    df['translated'] = df[col].apply(translate_to_english)
                    df['clean'] = df['translated'].apply(clean_text)
                    vect = vectorizer.transform(df['clean'])
                    df['prediction'] = model.predict(vect)
                    df['label'] = df['prediction'].map({0:'FAKE',1:'REAL'})

                    st.success("Prediction Completed ✔")
                    st.dataframe(df.head(50))

                    st.subheader("📊 Fake vs Real Count")
                    fig1, ax1 = plt.subplots(figsize=(5,4))
                    sns.countplot(x=df['label'], ax=ax1)
                    st.pyplot(fig1)

                    st.subheader("✍ Word Count Distribution")
                    df['word_count'] = df['clean'].apply(lambda x: len(x.split()))
                    fig2, ax2 = plt.subplots(figsize=(6,4))
                    sns.histplot(df['word_count'], bins=30, ax=ax2)
                    st.pyplot(fig2)

# ----------------------------------------------------------
# PAGE 4: ABOUT
# ----------------------------------------------------------
elif page == "About":
    st.markdown("<div class='big-title'>About This Project</div>", unsafe_allow_html=True)
    st.write("""
        This Fake News Detection app uses:
        - TF-IDF Vectorization  
        - Logistic Regression / SVM / Naive Bayes  
        - Auto-Translation (Any Language → English)  
        - Streamlit UI Dashboard  
    """)

# ----------------------------------------------------------
# END
# ----------------------------------------------------------
