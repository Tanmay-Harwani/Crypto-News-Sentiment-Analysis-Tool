import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import streamlit as st

def load_model():
    """Load the FinBERT tokenizer and model."""
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict_sentiment(text_list, tokenizer, model):
    """Predict the sentiment (Positive, Neutral, Negative) for a list of texts."""
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    results = nlp(text_list)
    sentiments = [result['label'] for result in results]
    return sentiments

def plot_sentiment_distribution(sentiments):
    """Plot the sentiment distribution as a pie chart."""
    sentiment_series = pd.Series(sentiments)
    counts = sentiment_series.value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    pastel_colors = ["#AEC6CF", "#FFB347", "#77DD77"]

    counts.plot.pie(
        ax=ax,
        colors=pastel_colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05] * len(counts),
        textprops={'fontsize': 14},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    ax.set_title("Sentiment Distribution", fontsize=18, fontweight='bold')
    ax.set_ylabel("")
    plt.tight_layout()
    return fig

def fetch_crypto_headlines_multi_api(limit=30):
    """Fetch latest cryptocurrency news headlines from GNews and NewsData APIs."""
    headlines = []
    api_key_gnews = st.secrets["api_key_gnews"]
    api_key_newsdata = st.secrets["api_key_newsdata"]

    # Try GNews first
    try:
        gnews_url = f"https://gnews.io/api/v4/search?q=cryptocurrency&lang=en&country=us&token={api_key_gnews}"
        response = requests.get(gnews_url)
        data = response.json()

        articles = data.get('articles', [])
        for article in articles:
            title = article.get('title', '')
            if title and len(title) > 30:
                headlines.append(title)
            if len(headlines) >= limit:
                break
    except Exception as e:
        print("GNews API failed:", e)

    # If not enough headlines, try NewsData.io
    if len(headlines) < limit:
        try:
            newsdata_url = f"https://newsdata.io/api/1/news?apikey={api_key_newsdata}&q=cryptocurrency&language=en&country=us"
            response = requests.get(newsdata_url)
            data = response.json()

            articles = data.get('results', [])
            for article in articles:
                title = article.get('title', '')
                if title and len(title) > 30:
                    headlines.append(title)
                if len(headlines) >= limit:
                    break
        except Exception as e:
            print("NewsData API failed:", e)

    return headlines[:limit]

def plot_wordcloud(texts):
    """Generate a WordCloud from a list of texts."""
    combined_text = " ".join(texts)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='tab10',
        max_words=100
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig

def calculate_sentiment_score(sentiments):
    """Calculate the average sentiment score: Positive = 1, Neutral = 0, Negative = -1."""
    score_mapping = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }
    scores = [score_mapping.get(s, 0) for s in sentiments]
    if len(scores) == 0:
        return 0
    return sum(scores) / len(scores)
