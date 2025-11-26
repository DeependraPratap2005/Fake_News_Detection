# ---------------------------------------------------------
# app.py — Final Glass + Light UI Fake News Detection App
# Developed by: Deependra Pratap Singh
# ---------------------------------------------------------

import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests, json

# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = "news_model.pkl"
VECTORIZER_PATH = "tfidf.pkl"

st.set_page_config(
    page_title="Fake News Detection — Glass UI",
    layout="wide",
    page_icon="📰"
)

# ---------------------------
# CSS
# ---------------------------
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #ffffff 0%, #e8f4ff 40%, #d9ecff 100%); background-size: cover; background-attachment: fixed; }
.block-container { background: rgba(255, 255, 255, 0.6) !important; border: 1px solid rgba(0,0,0,0.1) !important; border-radius: 20px; backdrop-filter: blur(12px) saturate(180%); box-shadow: 0 15px 30px rgba(0,0,0,0.08); padding: 35px !important; }
.big-title { color: #003b5c; font-size: 42px; font-weight: 800; letter-spacing: 1px; text-shadow: 0 2px 8px rgba(0,0,0,0.18); margin-bottom: 8px; }
.sidebar-name { position: fixed; bottom: 12px; left: 15px; font-size: 14px; color: #002d47; font-weight: 800; background: rgba(255,255,255,0.7); padding:6px 10px; border-radius: 10px; border: 1px solid rgba(0,0,0,0.12); }
.footer { display: none !important; }
.stButton>button { background: linear-gradient(90deg,#009dff,#4bb8ff); border: none; color: white; font-weight: 800; padding: 10px 22px; border-radius: 10px; }
.stButton>button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,150,255,0.28); }
.stRadio > div { gap: 12px !important; }
.stRadio label { background: rgba(255,255,255,0.8); color: #003b5c; padding: 12px; font-weight: 700; border-radius: 12px; border: 1px solid rgba(0,0,0,0.15); transition: 0.25s; text-align: center; }
.stRadio input:checked + label { background: linear-gradient(90deg, #009dff, #4bb8ff); color: white !important; font-weight: 800; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ("Home", "Predict Single", "Bulk Predict", "About"))
st.sidebar.markdown("<div class='sidebar-name'>🔹 Developed by <b>Deependra Pratap Singh</b></div>", unsafe_allow_html=True)

# ---------------------------
# Load Model + Vectorizer
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
# Text Cleaning
# ---------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------------------------
# Translate to English
# ---------------------------
def translate_to_english(text):
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={text}"
        response = requests.get(url, timeout=5)
        result = json.loads(response.text)
        return result[0][0][0]
    except:
        return text

# ----------------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------------
if page == "Home":
    st.markdown("<div class='big-title'>📰 Fake News Detection</div>", unsafe_allow_html=True)
    st.write("Detect Fake News using Machine Learning with TF-IDF + Logistic Regression/SVM. Supports **Hindi + 100 languages**.")
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
            st.success("✔ REAL NEWS" if pred == 1 else "✖ FAKE NEWS")

# ----------------------------------------------------------
# PAGE 3: Bulk Predict with Separate Graphs
# ----------------------------------------------------------
elif page == "Bulk Predict":
    st.markdown("<div class='big-title'>📂 Bulk Prediction</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV:", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded, encoding="latin1", on_bad_lines="skip")
        text_cols = df.select_dtypes(include="object").columns.tolist()

        if not text_cols:
            st.error("No text column found.")
        else:
            col = st.selectbox("Select text column:", text_cols)
            use_translation = st.checkbox("Enable Translation (Slow for large files)", value=False)

            if st.button("Run Prediction"):
                if not model or not vectorizer:
                    st.error("Model or vectorizer missing.")
                else:
                    # Translation (optional)
                    if use_translation:
                        st.warning("Translation ON — This may take time for large files.")
                        df["translated"] = df[col].astype(str).apply(translate_to_english)
                    else:
                        df["translated"] = df[col].astype(str)

                    # Clean text
                    df["clean"] = df["translated"].apply(clean_text)

                    # Prediction
                    vect = vectorizer.transform(df["clean"])
                    df["prediction"] = model.predict(vect)
                    df["label"] = df["prediction"].map({0: "FAKE", 1: "REAL"})

                    st.success("Prediction Completed ✔")
                    st.dataframe(df)  # Show full dataframe

                    # --------------------- SEPARATE GRAPHS ---------------------
                    st.subheader("📊 Separate Analysis Graphs")

                    # helper to limit to top N categories for plotting
                    def top_n_df(series, n=10):
                        top = series.value_counts().nlargest(n).index
                        return series.where(series.isin(top))

                    # 1) Label Distribution
                    st.markdown("### 🔹 1. Fake vs Real Distribution")
                    try:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.countplot(x="label", data=df, ax=ax)
                        ax.set_title("Fake vs Real News Count")
                        ax.set_xlabel("")
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception:
                        st.info("Could not render Label distribution.")

                    # 2) Category vs Label (top 10 categories)
                    if "Category" in df.columns:
                        st.markdown("### 🔹 2. Category-wise Fake vs Real (Top categories)")
                        try:
                            plot_col = "Category"
                            tmp = df.copy()
                            tmp[plot_col] = top_n_df(tmp[plot_col], n=10)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.countplot(x=plot_col, hue="label", data=tmp, order=tmp[plot_col].value_counts().nlargest(10).index, ax=ax)
                            plt.xticks(rotation=45)
                            ax.set_title("Category vs Label (Top 10)")
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception:
                            st.info("Could not render Category vs Label.")

                    # 3) Website Source vs Label (top 10)
                    if "Web" in df.columns:
                        st.markdown("### 🔹 3. Website Source-wise Fake vs Real (Top sources)")
                        try:
                            plot_col = "Web"
                            tmp = df.copy()
                            tmp[plot_col] = top_n_df(tmp[plot_col], n=10)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.countplot(x=plot_col, hue="label", data=tmp, order=tmp[plot_col].value_counts().nlargest(10).index, ax=ax)
                            plt.xticks(rotation=45)
                            ax.set_title("Web Source vs Label (Top 10)")
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception:
                            st.info("Could not render Web vs Label.")

                    # 4) Category Distribution (Top 10)
                    if "Category" in df.columns:
                        st.markdown("### 🔹 4. Category Distribution (Top categories)")
                        try:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            df_cat = df["Category"].value_counts().nlargest(10)
                            sns.barplot(x=df_cat.values, y=df_cat.index, ax=ax)
                            ax.set_title("Top Categories")
                            ax.set_xlabel("Count")
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception:
                            st.info("Could not render Category distribution.")

                    # 5) Website Distribution (Top 10)
                    if "Web" in df.columns:
                        st.markdown("### 🔹 5. Website Source Distribution (Top sources)")
                        try:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            df_web = df["Web"].value_counts().nlargest(10)
                            sns.barplot(x=df_web.values, y=df_web.index, ax=ax)
                            ax.set_title("Top Web Sources")
                            ax.set_xlabel("Count")
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception:
                            st.info("Could not render Web distribution.")

                    # 6) Time Trend: Fake vs Real Over Time
                    if "Date" in df.columns:
                        st.markdown("### 🔹 6. Fake vs Real Over Time (Monthly)")
                        try:
                            # parse dates safely
                            df["Date_clean"] = pd.to_datetime(df["Date"], errors="coerce")
                            # drop null dates
                            df_time = df.dropna(subset=["Date_clean"]).copy()
                            if not df_time.empty:
                                # group by month
                                df_time["month"] = df_time["Date_clean"].dt.to_period("M").dt.to_timestamp()
                                trend = df_time.groupby(["month", "label"]).size().unstack(fill_value=0)
                                # plot
                                fig, ax = plt.subplots(figsize=(10, 4))
                                trend.plot(ax=ax)
                                ax.set_title("Fake vs Real Trend Over Time (by month)")
                                ax.set_xlabel("Month")
                                ax.set_ylabel("Count")
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info("No valid Date values to plot time trend.")
                        except Exception:
                            st.info("Date column format invalid for time graph.")

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
        - Separate Analysis Graphs (Category, Source, Time, Label)  
        - Streamlit UI Dashboard  
    """)
