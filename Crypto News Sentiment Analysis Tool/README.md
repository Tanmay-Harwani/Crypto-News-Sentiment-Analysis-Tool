# 📰 Crypto News Sentiment Analysis Tool (USA)

Analyze the sentiment of the latest cryptocurrency news articles with an interactive Streamlit web app!  
Built using a pre-trained FinBERT model, this tool lets you explore the mood of the crypto market in real-time.

---

## 🚀 Overview

This project fetches or accepts user-uploaded crypto news articles and predicts their sentiment (Positive, Neutral, Negative) using a fine-tuned FinBERT model.  
The app also provides visualizations like a Sentiment Pie Chart, WordCloud of trending terms, and a Crypto Market Mood Index.

✅ Upload your own news CSV  
✅ Or fetch the latest U.S. cryptocurrency headlines automatically  
✅ Get powerful insights instantly!

---

## 🗂 Project Structure

- `sentiment_analysis.py` — Functions for loading FinBERT, sentiment prediction, news scraping, WordCloud generation, and sentiment scoring.
- `app.py` — Streamlit app to interactively upload files, fetch news, and visualize results.
- `requirements.txt` — List of required Python libraries.
- `data/` — (Optional) Folder for any sample news data (for testing).
- `.streamlit/` — Streamlit app configuration (optional).

---

## 🛠 How to Run Locally

1. **Install dependencies** (preferably inside a virtual environment):

   ```bash
   pip install -r requirements.txt 
   ```

2. **Run the Streamlit App:**

      ```bash
      streamlit run app.py
      ```

# 🔑 API Keys Requirement
This project fetches live crypto news from GNews and NewsData.io APIs.

Before running the app, you must:

Create a free account on gnews.io and newsdata.io

Obtain your API keys and then insert those API Keys inside the sentiment_analysis.py file

---
## **Use the app**:
   - 📂 **Upload your own CSV file**: Make sure the CSV has a `text` column containing news headlines.
   - 🔄 **Or click "Fetch & Analyze Latest Crypto News"** to automatically fetch the latest US-based crypto news headlines.

---
## 📄 CSV File Requirements
When uploading your own CSV file:
- The CSV must contain a column named text.

- Each row under text should contain a news headline or a short article snippet.

- Extra columns (like date, source) are allowed but will be ignored.

- Ensure that the file is saved with UTF-8 encoding to avoid special character errors.

# Example:


text
- Bitcoin surges 10% after ETF approval.
- Ethereum gas fees remain high despite upgrades.
- Dogecoin rallies as Musk tweets again.
---
## Output Visualizations

    - 📄 Table with Sentiment Predictions
    - 🥧 Sentiment Distribution Pie Chart
    - ☁️ WordCloud of trending crypto terms
    - 📈 Crypto Market Mood Index (Sentiment Score Line Chart)
    - 💾 Option to Download Analyzed Results as CSV

---
## Technologies Used

    - Python 3.10+
    - Streamlit
    - Huggingface Transformers(FinBERT model)
    - Pandas
    - Matplotlib
    - WordCloud
    - Requests / Web Scraping
