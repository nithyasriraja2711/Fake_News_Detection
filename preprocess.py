import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only first time)
nltk.download('stopwords')

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge datasets
news = pd.concat([fake, true], axis=0)
news = news.sample(frac=1, random_state=42).reset_index(drop=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

news["text"] = news["text"].apply(clean_text)

print(news[["text", "label"]].head())