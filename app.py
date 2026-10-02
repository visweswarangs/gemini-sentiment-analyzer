"""
Gemini Sentiment & Tone Analyzer
================================
A beginner-friendly Streamlit web application powered by the Google Gemini API.
Analyzes sentiment, tone, emotional intensity, and key phrases in user text.
"""

import os
import json
import time
import streamlit as st
from pydantic import BaseModel, Field
from typing import List

# Load .env file (supports python-dotenv or lightweight fallback)
env_file_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file_path, override=True)
    except ImportError:
        try:
            with open(env_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip('"').strip("'")
        except Exception:
            pass

# ==============================================================================
# 1. Page Configuration & Custom CSS Styling
# ==============================================================================
st.set_page_config(
    page_title="Gemini Sentiment & Tone Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern visual cards and badges
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
        margin-bottom: 1rem;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #1e212b;
            border-color: #2e3342;
        }
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .metric-subtitle {
        font-size: 0.85rem;
        color: #888;
    }
    /* Tag Pills */
    .tag-pill {
        display: inline-block;
        background-color: #eef2ff;
        color: #4338ca;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
        border: 1px solid #c7d2fe;
    }
    @media (prefers-color-scheme: dark) {
        .tag-pill {
            background-color: #27273a;
            color: #a5b4fc;
            border-color: #3e3f5b;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. Structured Output Schema (Pydantic)
# ==============================================================================
class EmotionItem(BaseModel):
    emotion: str = Field(description="Name of the emotion (e.g., Joy, Frustration, Hope, Anxiety)")
    percentage: int = Field(description="Estimated intensity percentage from 0 to 100")

class SentimentAnalysisResult(BaseModel):
    sentiment: str = Field(description="Overall sentiment: 'Positive', 'Negative', 'Neutral', or 'Mixed'")
    sentiment_score: int = Field(description="Score from 0 (Extremely Negative) to 100 (Extremely Positive), 50 is Neutral")
    primary_tone: str = Field(description="Primary tone, e.g. 'Formal & Professional', 'Sarcastic', 'Enthusiastic', 'Empathetic'")
    secondary_tone: str = Field(description="Secondary or underlying tone")
    intensity: str = Field(description="'Low', 'Moderate', or 'High'")
    emotional_breakdown: List[EmotionItem] = Field(description="List of top 2 to 4 emotions identified with percentages")
    key_phrases: List[str] = Field(description="3 to 5 notable phrases or words that influenced this judgment")
    summary_explanation: str = Field(description="A concise 2-3 sentence explanation of the sentiment and tone findings")


# ==============================================================================
# 3. Helper: Call Gemini API
# ==============================================================================
def analyze_text_with_gemini(api_key: str, text: str, model_name: str = "gemini-2.0-flash") -> tuple[SentimentAnalysisResult, str]:
    """
    Sends text to Gemini API and asks for structured JSON output conforming to SentimentAnalysisResult.
    Includes automatic retry and fallback to ensure resilience during high-demand periods (503).
    """
    system_prompt = (
        "You are an expert psycholinguist and sentiment analysis engine. "
        "Analyze the provided user text and evaluate its overall sentiment, primary tone, "
        "emotional breakdown, emotional intensity, key influential phrases, and a summary explanation."
    )

    # Order of backup models to try if the requested model is busy (503) or not found (404)
    fallback_candidates = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.8-flash"]
    models_to_try = [model_name] + [m for m in fallback_candidates if m != model_name]

    last_error = None

    for candidate_model in models_to_try:
        # Try up to 2 times for temporary spikes
        for attempt in range(2):
            try:
                # 1. Modern google-genai SDK
                try:
                    from google import genai
                    from google.genai import types

                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=candidate_model,
                        contents=text,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=SentimentAnalysisResult,
                            temperature=0.2,
                        ),
                    )
                    return SentimentAnalysisResult.model_validate_json(response.text), candidate_model

                except ImportError:
                    # 2. Legacy google-generativeai SDK fallback
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=api_key)
                    model = legacy_genai.GenerativeModel(
                        model_name=candidate_model,
                        system_instruction=system_prompt,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    prompt = (
                        f"{text}\n\n"
                        f"Return valid JSON matching this schema: "
                        f"{json.dumps(SentimentAnalysisResult.model_json_schema())}"
                    )
                    response = model.generate_content(prompt)
                    data = json.loads(response.text)
                    return SentimentAnalysisResult.model_validate(data), candidate_model

            except Exception as e:
                err_str = str(e)
                last_error = e
                # Check for high server load (503) or rate limit (429)
                if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str or "429" in err_str:
                    if attempt == 0:
                        time.sleep(2)  # Wait 2 seconds and retry same model
                        continue
                    else:
                        break  # Try next candidate model
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    # Model not available, immediately switch to next model
                    break
                else:
                    # Fatal error like invalid API key
                    raise e

    if last_error:
        raise last_error
    raise RuntimeError("Failed to get analysis from Gemini API.")


# ==============================================================================
# 4. Sidebar: Setup & Presets
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Retrieve API key from environment or secrets if available
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    if not env_api_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                env_api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            # Streamlit raises StreamlitSecretNotFoundError if secrets.toml doesn't exist
            pass

    user_api_key = st.text_input(
        "Gemini API Key",
        value=env_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Get a free API key from Google AI Studio (aistudio.google.com)",
    )

    st.markdown("[🔑 Get a Free Gemini API Key](https://aistudio.google.com/app/apikey)", unsafe_allow_html=True)
    st.divider()

    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.8-flash"],
        index=0,
        help="Gemini Flash models are fast, intelligent, and cost-effective.",
    )

    st.divider()
    st.subheader("💡 Try Sample Texts")
    sample_options = {
        "— Select a sample —": "",
        "Positive Feedback": "I just received my order and I am blown away! The build quality exceeded every expectation, the packaging was gorgeous, and shipping was two days early. You gained a customer for life!",
        "Frustrated Customer": "I've been on hold for over 45 minutes and your representative hung up on me twice. The app crashed right during checkout and charged my card twice. Fix this immediately or I am reporting this to consumer protection.",
        "Mixed Review": "The design of this smartphone is gorgeous and the display is top-tier. However, the battery drains in barely 5 hours and the camera gets sluggish in low light. Good start, but hard to recommend at this price point.",
        "Formal Business Notice": "Please find attached the revised quarterly schedule. All teams are requested to review Section 3 by Thursday afternoon and provide necessary sign-offs prior to final distribution."
    }
    selected_sample = st.selectbox("Quick load example:", list(sample_options.keys()))

    st.divider()
    st.caption("Built with Streamlit & Google Gemini API")


# ==============================================================================
# 5. Main User Interface
# ==============================================================================
st.title("✨ Gemini Sentiment & Tone Analyzer")
st.write(
    "Enter any text below (customer review, email, social post, or feedback) to analyze its "
    "sentiment, emotional tone, and key phrases in real-time."
)

# Text input area (pre-filled if a sample is picked)
default_text = sample_options[selected_sample] if selected_sample != "— Select a sample —" else ""
input_text = st.text_area(
    "Text to Analyze",
    value=default_text,
    placeholder="Type or paste text here (e.g. 'I was really excited about the product, but the delivery took forever...')...",
    height=160,
)

analyze_button = st.button("🚀 Analyze Sentiment & Tone", type="primary", use_container_width=True)

# ==============================================================================
# 6. Analysis Execution & Visual Metric Cards
# ==============================================================================
if analyze_button:
    if not user_api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar to proceed.")
    elif not input_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🤖 Analyzing sentiment and tone with Gemini..."):
            try:
                result, used_model = analyze_text_with_gemini(
                    api_key=user_api_key,
                    text=input_text.strip(),
                    model_name=selected_model,
                )

                if used_model != selected_model:
                    st.info(f"ℹ️ Note: **{selected_model}** was experiencing high server demand. Automatically processed via **{used_model}**.")

                st.success("✅ Analysis Complete!")
                st.subheader("📊 Key Metrics")

                # Configure Sentiment Colors & Badges
                sentiment_meta = {
                    "Positive": {"icon": "🟢", "color": "#16a34a"},
                    "Negative": {"icon": "🔴", "color": "#dc2626"},
                    "Neutral":  {"icon": "⚪", "color": "#64748b"},
                    "Mixed":    {"icon": "🟣", "color": "#9333ea"},
                }
                meta = sentiment_meta.get(result.sentiment, {"icon": "🔍", "color": "#3b82f6"})

                # 4 Visual Metric Cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Overall Sentiment</div>
                        <div class="metric-value" style="color: {meta['color']};">
                            {meta['icon']} {result.sentiment}
                        </div>
                        <div class="metric-subtitle">Classification</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    score = result.sentiment_score
                    score_color = "#16a34a" if score >= 60 else ("#dc2626" if score <= 40 else "#64748b")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Sentiment Score</div>
                        <div class="metric-value" style="color: {score_color};">
                            {score}/100
                        </div>
                        <div class="metric-subtitle">0=Negative • 100=Positive</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Primary Tone</div>
                        <div class="metric-value" style="font-size: 1.25rem; color: #3b82f6;">
                            🎭 {result.primary_tone}
                        </div>
                        <div class="metric-subtitle">Secondary: {result.secondary_tone}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    intensity_colors = {"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"}
                    int_color = intensity_colors.get(result.intensity, "#6366f1")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Emotional Intensity</div>
                        <div class="metric-value" style="color: {int_color};">
                            ⚡ {result.intensity}
                        </div>
                        <div class="metric-subtitle">Strength of feeling</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Sentiment Score Bar
                st.write("**Sentiment Polarity Scale**")
                st.progress(result.sentiment_score / 100)

                st.divider()

                # Breakdown Section (2 Columns)
                left_col, right_col = st.columns([1, 1])

                with left_col:
                    st.subheader("🎭 Emotional Breakdown")
                    if result.emotional_breakdown:
                        for item in result.emotional_breakdown:
                            col_label, col_val = st.columns([3, 1])
                            col_label.write(f"**{item.emotion}**")
                            col_val.write(f"`{item.percentage}%`")
                            st.progress(min(max(item.percentage, 0), 100) / 100)
                    else:
                        st.info("No distinct secondary emotions detected.")

                with right_col:
                    st.subheader("🔍 Influential Key Phrases")
                    if result.key_phrases:
                        pills_html = "".join([f'<span class="tag-pill">"{phrase}"</span>' for phrase in result.key_phrases])
                        st.markdown(f"<div style='margin-bottom: 1.5rem;'>{pills_html}</div>", unsafe_allow_html=True)
                    else:
                        st.write("None identified.")

                    st.subheader("💡 AI Summary & Explanation")
                    st.info(result.summary_explanation)

                # Developer / Debug Expander
                with st.expander("🛠️ View Raw JSON Response (API Output)"):
                    st.json(result.model_dump())

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.info("Tip: Double-check your API key and make sure you have internet access.")