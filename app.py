import gradio as gr
import joblib
import numpy as np

# Load models (runs once at startup)
tfidf = joblib.load("tfidf_vectorizer.pkl")
lr = joblib.load("logistic_regression_model.pkl")

def classify_news(text):
    if not text.strip():
        return {"Error": "Please enter some text"}
    
    text_vectorized = tfidf.transform([text])
    probabilities = lr.predict_proba(text_vectorized)[0]
    
    # Use model's actual class order instead of hardcoded mapping
    results = {label: float(round(prob, 4)) for label, prob in zip(lr.classes_, probabilities)}
    return results

demo = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(
        label="Enter a news article headline or description",
        lines=3,
        placeholder="e.g., NASA launches new satellite to monitor climate change in the Arctic."
    ),
    outputs=gr.Label(label="Category Prediction"),
    title="NewsWise AI - News Article Classifier",
    description="Classifies news articles into World, Sports, Business, or Sci/Tech categories using TF-IDF + Logistic Regression. Trained on 120,000 AG News articles with 92.18% test accuracy.",
    examples=[
        ["LeBron James scores 40 points as the Lakers defeat the Celtics in overtime thriller."],
        ["Federal Reserve holds interest rates steady amid mixed inflation signals on Wall Street."],
        ["NASA launches new satellite to study climate change patterns in the Arctic region."],
        ["United Nations Security Council votes on new sanctions following recent missile tests."],
        ["Apple unveils new AI chip designed for on-device machine learning to compete with Nvidia."]
    ],
    theme="default"
)

demo.launch()
