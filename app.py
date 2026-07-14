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

st.title("📰 Fake News Detection using Machine Learning")
st.markdown("### Final Year Project")
st.write("This application predicts whether a news article is **Fake** or **Real** using NLP and Machine Learning.")

st.sidebar.title("Project Details")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Feature Extraction:** TF-IDF")
st.sidebar.write("**Language:** Python")
st.sidebar.write("**Framework:** Streamlit")

news = st.text_area("📝 Enter News Article", height=250)

if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")
    else:

        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)

        confidence = model.predict_proba(vector)

        if prediction[0] == 0:
            st.error("🚨 Fake News")
            st.write(f"**Confidence:** {confidence[0][0]*100:.2f}%")
        else:
            st.success("✅ Real News")
            st.write(f"**Confidence:** {confidence[0][1]*100:.2f}%")