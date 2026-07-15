import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords")

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- UI ----------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)
st.image("assets/banner.png", use_container_width=True)

st.markdown("""
### Artificial Intelligence & Machine Learning

This web application predicts whether a news article is **Fake** or **Real** using Natural Language Processing (NLP) and Machine Learning.

---
""")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Dataset", "44,898")

with col2:
    st.metric("🎯 Accuracy", "99.45%")   # Replace with your actual accuracy

with col3:
    st.metric("🤖 Algorithm", "Logistic Regression")

with col4:
    st.metric("📝 Features", "TF-IDF")
st.sidebar.title("📰 Fake News Detector")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Project Details")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Feature Extraction:** TF-IDF")
st.sidebar.write("**Language:** Python")
st.sidebar.write("**Framework:** Streamlit")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Dataset Statistics")
st.sidebar.write("Total Articles : **44,898**")
st.sidebar.write("Fake News : **23,481**")
st.sidebar.write("Real News : **21,417**")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully ✅")

st.subheader("📝 Enter News Article")

news = st.text_area(
    "",
    height=250,
    placeholder="Paste a complete news article here..."
)

if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)
        confidence = model.predict_proba(vector)

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction[0] == 0:

            confidence_score = confidence[0][0]

            st.error("🚨 Fake News")
            st.progress(float(confidence_score))
            st.write(f"Confidence : **{confidence_score*100:.2f}%**")

        else:

            confidence_score = confidence[0][1]

            st.success("✅ Real News")
            st.progress(float(confidence_score))
            st.write(f"Confidence : **{confidence_score*100:.2f}%**")

st.markdown("---")
st.caption(
    "Fake News Detection System | Final Year Project | Developed using Python, Streamlit, Scikit-learn and NLP"
)