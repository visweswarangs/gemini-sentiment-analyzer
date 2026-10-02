# 🎭 Gemini Sentiment & Tone Analyzer

A complete, beginner-friendly AI project built with **Streamlit** and the **Google Gemini API**. It analyzes text input in real-time and visualizes sentiment classification, polarity score, emotional breakdown, and key influential phrases.

---

## 🌟 Features

- **Real-time AI Sentiment Analysis**: Categorizes text into Positive, Negative, Neutral, or Mixed.
- **Sentiment Polarity Scale (0–100)**: Quantitative scoring of text sentiment.
- **Tone & Emotion Breakdown**: Identifies primary/secondary tone and emotional distribution (e.g., Joy, Frustration, Anticipation).
- **Influential Key Phrases**: Extracts the specific keywords that drove the AI's decision.
- **Visual Metric Cards & Progress Bars**: Clean, responsive Streamlit dashboard.
- **Interactive Quick-Load Samples**: Test customer reviews, complaints, and formal emails with 1 click.
- **Secure API Key Handling**: Input your key safely via the sidebar (never hardcoded).

---

## 🚀 How to Run the Application

### 1. Prerequisites
- Python 3.10 or higher installed.

### 2. Navigate to Project Directory
Open your terminal (PowerShell or Command Prompt):
```powershell
cd C:\Projects\sentiment_analyzer
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```powershell
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 🔑 Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API Key**.
4. Paste the key into the sidebar of the web application.

---

## 🛠️ Project Structure

- `app.py`: Main application code containing UI layout, CSS styling, and Gemini API calls.
- `requirements.txt`: List of required Python packages (`streamlit`, `google-genai`, `pydantic`, `python-dotenv`).
- `.env.example`: Optional environment variable template if you prefer setting `GEMINI_API_KEY` in a `.env` file.
- `README.md`: Project documentation and user guide.
