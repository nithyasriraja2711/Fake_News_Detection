import streamlit as st
import pickle
import requests
from config import NEWS_API_KEY
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
# Initialize prediction history
if "history" not in st.session_state:
    st.session_state.history = []

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- UI ----------------
def verify_live_news(query):

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}"
        f"&language=en"
        f"&sortBy=publishedAt"
        f"&pageSize=5"
        f"&apiKey={NEWS_API_KEY}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    return data.get("articles", [])

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
    st.metric("🎯 Accuracy", "98.46%")   # Replace with your actual accuracy

with col3:
    st.metric("🤖 Algorithm", "Logistic Regression")

with col4:
    st.metric("📝 Features", "TF-IDF")
# 👇 ADD THE ABOUT PROJECT SECTION HERE
with st.expander("ℹ️ About This Project"):

    st.markdown("""
### Fake News Detection System

This application uses **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **Fake** or **Real**.

### Technologies Used
- 🐍 Python
- 📚 Scikit-learn
- 🌐 Streamlit
- 📝 NLP
- 🔤 TF-IDF Vectorizer
- 🤖 Logistic Regression

### Dataset
- Total Articles: **44,898**
- Fake News: **23,481**
- Real News: **21,417**

### Model Performance
- Accuracy: **98.46%**
""")

# 👇 Your existing input section
st.subheader("📝 Enter News Article")

news = st.text_area(
    "📝 Enter News Article",
    value=st.session_state.get("news", ""),
    height=250,
    placeholder="Paste a complete news article here...",
    key="news_input"
)
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

if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])

        with st.spinner("🔍 Analyzing the news article..."):
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

        from datetime import datetime

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),"Prediction": "🚨 Fake" if prediction[0] == 0 else "✅ Real",
            "Confidence": f"{confidence_score*100:.2f}%"
})

st.markdown("---")

if st.button("🌐 Verify with Live News"):

    with st.spinner("Searching trusted news sources..."):

        articles = verify_live_news(news[:100])

    if articles:

        st.subheader("📰 Similar News Found")

        for article in articles:

            st.markdown(f"### {article['title']}")
            st.write(article["source"]["name"])
            st.write(article["publishedAt"])
            st.write(article["url"])

    else:

        st.warning("No similar news found.")
        
st.markdown("---")
st.subheader("📜 Prediction History")

if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.info("No predictions yet.")

st.markdown("---")
st.caption(
    "Fake News Detection System | Final Year Project | Developed using Python, Streamlit, Scikit-learn and NLP"
)