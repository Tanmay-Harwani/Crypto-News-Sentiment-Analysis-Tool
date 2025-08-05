import streamlit as st
import pandas as pd
from sentiment_analysis import (
    load_model,
    predict_sentiment,
    plot_sentiment_distribution,
    fetch_crypto_headlines_multi_api,
    plot_wordcloud,
    calculate_sentiment_score
)

tokenizer, model = load_model()

st.set_page_config(page_title="Crypto News Sentiment Analysis Tool")
st.title("📰 Crypto News Sentiment Analysis Tool (USA Only)")
st.write("Upload a CSV of crypto news articles or fetch the latest US-based cryptocurrency news to analyze their sentiment (Positive, Neutral, Negative).")

st.markdown("### 📂 Upload Your Own Crypto News CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'text' not in data.columns:
        st.error("The CSV must contain a 'text' column.")
    else:
        st.success("File successfully uploaded!")
        st.dataframe(data.head())

        if st.button("Analyze Uploaded News Sentiment"):
            with st.spinner("Analyzing..."):
                articles = data['text'].tolist()
                sentiments = predict_sentiment(articles, tokenizer, model)
                data['Sentiment'] = sentiments

            st.success("Analysis Complete!")
            st.subheader("Sentiment Results")
            st.dataframe(data[['text', 'Sentiment']])
            fig = plot_sentiment_distribution(sentiments)
            st.pyplot(fig)
            fig_wc = plot_wordcloud(data['text'].tolist())
            st.pyplot(fig_wc)

            sentiment_score = calculate_sentiment_score(sentiments)
            st.subheader(" 📈 Crypto Market Mood Index")
            st.line_chart([0, sentiment_score])

            csv = data.to_csv(index=False)
            st.download_button("Download Results as CSV", csv, "crypto_sentiment_uploaded.csv", "text/csv")

st.markdown("### 🔄 Fetch Latest Crypto News (Multi-API)")

if st.button("Fetch & Analyze Latest Crypto News"):
    with st.spinner("Fetching and analyzing..."):
        headlines = fetch_crypto_headlines_multi_api(limit=30)
        sentiments = predict_sentiment(headlines, tokenizer, model)
        auto_df = pd.DataFrame({'text': headlines, 'Sentiment': sentiments})
        sentiment_score = calculate_sentiment_score(sentiments)

    st.success("Analysis Complete!")
    st.dataframe(auto_df)

    fig = plot_sentiment_distribution(sentiments)
    st.pyplot(fig)

    fig_wc = plot_wordcloud(auto_df['text'].tolist())
    st.pyplot(fig_wc)

    st.subheader("📈 Crypto Market Mood Index")
    st.line_chart([0, sentiment_score])

    st.markdown("---")

    csv = auto_df.to_csv(index=False)
    st.download_button("Download Results as CSV", csv, "crypto_sentiment_results.csv", "text/csv")
