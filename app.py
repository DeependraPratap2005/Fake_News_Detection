# ---------------------------------------------------------
# app.py — Safe Streamlit App for Fake News Detection
# ---------------------------------------------------------

import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import requests, json

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "news_model.pkl"
VECTORIZER_PATH = "tfidf.pkl"

st.set_page_config(page_title="Fake News Detection", layout="wide", page_icon="📰")

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.big-title { color: #003b5c; font-size: 42px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ("Home", "Predict Single", "Bulk Predict", "About"))

# ---------------------------
# Load model + vectorizer
# ---------------------------
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
# Helper functions
# ---------------------------
def clean_text(text):
    if pd.isna(text): return ""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def translate_to_english(text):
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={text}"
        resp = requests.get(url, timeout=5)
        return json.loads(resp.text)[0][0][0]
    except:
        return text

# ----------------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------------
if page == "Home":
    st.markdown("<div class='big-title'>📰 Fake News Detection</div>", unsafe_allow_html=True)
    st.write("Detect Fake News using ML with TF-IDF + Logistic Regression/SVM. Supports multiple languages.")
    if model and vectorizer:
        st.success("Model & Vectorizer loaded ✔")
    else:
        st.warning("Model or Vectorizer missing!")

# ----------------------------------------------------------
# PAGE 2: Predict Single
# ----------------------------------------------------------
elif page == "Predict Single":
    st.markdown("<div class='big-title'>🔎 Predict Single News</div>", unsafe_allow_html=True)
    text = st.text_area("Enter news text:", height=200)
    if st.button("Predict"):
        if not text:
            st.warning("Enter text first!")
        elif not model or not vectorizer:
            st.error("Model or vectorizer missing!")
        else:
            translated = translate_to_english(text)
            cleaned = clean_text(translated)
            vect = vectorizer.transform([cleaned])
            pred = model.predict(vect)[0]
            st.subheader("Prediction Result")
            st.success("✔ REAL NEWS" if pred==1 else "✖ FAKE NEWS")

# ----------------------------------------------------------
# PAGE 3: Bulk Predict
# ----------------------------------------------------------
elif page == "Bulk Predict":
    st.markdown("<div class='big-title'>📂 Bulk Prediction</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, encoding="latin1", on_bad_lines="skip")
        text_cols = df.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            st.error("No text column found!")
        else:
            col = st.selectbox("Select text column", text_cols)
            translate_option = st.checkbox("Translate to English (slow)", value=False)
            run_button = st.button("Run Prediction")
            if run_button:
                if not model or not vectorizer:
                    st.error("Model or vectorizer missing!")
                else:
                    # Translation + cleaning
                    df["translated"] = df[col].astype(str).apply(lambda x: translate_to_english(x) if translate_option else x)
                    df["clean"] = df["translated"].apply(clean_text)

                    # Predict
                    vect = vectorizer.transform(df["clean"])
                    df["prediction"] = model.predict(vect)
                    df["label"] = df["prediction"].map({0:"FAKE", 1:"REAL"})

                    st.success("Prediction done ✔")
                    st.dataframe(df)  # Show all rows safely

                    # Dynamic graphs
                    st.subheader("Graphs")
                    plot_col = st.selectbox("Select column for graph", [c for c in df.columns if c not in ["clean","translated","prediction","label"]])
                    if plot_col:
                        fig, ax = plt.subplots(figsize=(6,4))
                        if df[plot_col].dtype=="object":
                            sns.countplot(x=plot_col, hue="label", data=df, ax=ax)
                        else:
                            sns.histplot(data=df, x=plot_col, hue="label", bins=30, kde=True, ax=ax)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

# ----------------------------------------------------------
# PAGE 4: ABOUT
# ----------------------------------------------------------
elif page == "About":
    st.markdown("<div class='big-title'>About</div>", unsafe_allow_html=True)
    st.write("""
    Fake News Detection App:
    - TF-IDF Vectorization
    - Logistic Regression / SVM
    - Auto-Translation
    - Streamlit UI Dashboard
    """)
