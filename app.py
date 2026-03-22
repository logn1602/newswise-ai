import gradio as gr
import joblib
import numpy as np

# Load models (runs once at startup)
tfidf = joblib.load("tfidf_vectorizer.pkl")
lr = joblib.load("logistic_regression_model.pkl")

# Category icons
CATEGORY_ICONS = {
    "Business": "💼",
    "Sci/Tech": "🔬",
    "Sports": "⚽",
    "World": "🌍"
}

def classify_news(text):
    if not text or not text.strip():
        return "", "", ""
    
    text_vectorized = tfidf.transform([text])
    probabilities = lr.predict_proba(text_vectorized)[0]
    
    # Build results
    results = {label: float(prob) for label, prob in zip(lr.classes_, probabilities)}
    top_category = max(results, key=results.get)
    top_confidence = results[top_category]
    icon = CATEGORY_ICONS.get(top_category, "📰")
    
    # Build prediction card HTML
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        margin: 10px 0;
    ">
        <div style="font-size: 48px; margin-bottom: 10px;">{icon}</div>
        <div style="
            font-size: 32px;
            font-weight: 700;
            color: #e94560;
            margin-bottom: 8px;
            letter-spacing: 1px;
        ">{top_category}</div>
        <div style="
            font-size: 18px;
            color: #a8a8b3;
            margin-bottom: 20px;
        ">Confidence: {top_confidence:.1%}</div>
        <div style="
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            height: 12px;
            width: 100%;
            overflow: hidden;
            margin-bottom: 5px;
        ">
            <div style="
                background: linear-gradient(90deg, #e94560, #0f3460);
                height: 100%;
                width: {top_confidence*100}%;
                border-radius: 12px;
                transition: width 0.5s ease;
            "></div>
        </div>
    </div>
    """
    
    # Build breakdown HTML
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    breakdown_html = """
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        margin: 10px 0;
    ">
        <div style="
            font-size: 16px;
            font-weight: 600;
            color: #e94560;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 2px;
        ">Full Breakdown</div>
    """
    
    for label, prob in sorted_results:
        icon_small = CATEGORY_ICONS.get(label, "📰")
        bar_color = "#e94560" if label == top_category else "#533483"
        breakdown_html += f"""
        <div style="margin-bottom: 12px;">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 4px;
            ">
                <span style="color: #e0e0e0; font-size: 14px; font-weight: 500;">
                    {icon_small} {label}
                </span>
                <span style="color: #a8a8b3; font-size: 14px; font-weight: 600;">
                    {prob:.1%}
                </span>
            </div>
            <div style="
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                height: 8px;
                overflow: hidden;
            ">
                <div style="
                    background: {bar_color};
                    height: 100%;
                    width: {prob*100}%;
                    border-radius: 8px;
                "></div>
            </div>
        </div>
        """
    
    breakdown_html += "</div>"
    
    # Label output for API compatibility
    label_results = {f"{CATEGORY_ICONS[label]} {label}": prob for label, prob in results.items()}
    
    return card_html, breakdown_html, label_results

# Custom CSS
custom_css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    footer {visibility: hidden}
"""

# Header HTML
header_html = """
<div style="
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 20px;
    padding: 40px 30px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
">
    <div style="font-size: 56px; margin-bottom: 12px;">📰</div>
    <h1 style="
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(90deg, #e94560, #533483, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
        letter-spacing: 1px;
    ">NewsWise AI</h1>
    <p style="
        color: #a8a8b3;
        font-size: 16px;
        margin: 0 0 16px 0;
        line-height: 1.6;
    ">Intelligent News Article Classifier powered by Machine Learning</p>
    <div style="
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
    ">
        <span style="
            background: rgba(233,69,96,0.15);
            color: #e94560;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        ">🎯 92.18% Accuracy</span>
        <span style="
            background: rgba(83,52,131,0.2);
            color: #a78bfa;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        ">📊 120K Articles Trained</span>
        <span style="
            background: rgba(15,52,96,0.3);
            color: #60a5fa;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        ">⚡ TF-IDF + Logistic Regression</span>
    </div>
</div>
"""

# About section HTML
about_html = """
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 10px;
">
    <h3 style="
        color: #e94560;
        margin: 0 0 16px 0;
        font-size: 18px;
        letter-spacing: 1px;
    ">📖 About This Model</h3>
    <div style="color: #c4c4cc; font-size: 14px; line-height: 1.8;">
        <p style="margin: 0 0 12px 0;">
            <strong style="color: #e0e0e0;">Dataset:</strong> AG News — 120,000 training articles and 7,600 test articles across 4 categories
        </p>
        <p style="margin: 0 0 12px 0;">
            <strong style="color: #e0e0e0;">Model:</strong> TF-IDF Vectorizer (50,000 features, unigram + bigram) + Logistic Regression
        </p>
        <p style="margin: 0 0 12px 0;">
            <strong style="color: #e0e0e0;">Accuracy:</strong> 92.18% on held-out test set
        </p>
        <p style="margin: 0 0 12px 0;">
            <strong style="color: #e0e0e0;">Best Category:</strong> Sports (98% recall) — distinctive vocabulary makes classification easy
        </p>
        <p style="margin: 0 0 12px 0;">
            <strong style="color: #e0e0e0;">Hardest Category:</strong> Business (89% recall) — overlapping vocabulary with Sci/Tech
        </p>
        <p style="margin: 0 0 0 0;">
            <strong style="color: #e0e0e0;">Built by:</strong> Shubh Dave | Northeastern University | EAI 6010
        </p>
    </div>
</div>
"""

# Build the app with Blocks for full layout control
with gr.Blocks(css=custom_css, theme=gr.themes.Base(
    primary_hue="red",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
)) as demo:
    
    # Header
    gr.HTML(header_html)
    
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📝 Enter News Text",
                placeholder="Paste a news headline or article description here...",
                lines=5,
                max_lines=10
            )
            submit_btn = gr.Button("🔍 Classify Article", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
            
            gr.Examples(
                examples=[
                    ["LeBron James scores 40 points as the Lakers defeat the Celtics in overtime thriller."],
                    ["Federal Reserve holds interest rates steady amid mixed inflation signals on Wall Street."],
                    ["NASA launches new satellite to study climate change patterns in the Arctic region."],
                    ["United Nations Security Council votes on new sanctions following recent missile tests."],
                    ["Apple unveils new AI chip designed for on-device machine learning to compete with Nvidia."]
                ],
                inputs=text_input,
                label="📌 Try These Examples"
            )
        
        # Right column - Output
        with gr.Column(scale=1):
            prediction_card = gr.HTML(label="Prediction")
            breakdown_card = gr.HTML(label="Breakdown")
            label_output = gr.Label(visible=False)
    
    # About section
    gr.HTML(about_html)
    
    # Button actions
    submit_btn.click(
        fn=classify_news,
        inputs=text_input,
        outputs=[prediction_card, breakdown_card, label_output]
    )
    clear_btn.click(
        fn=lambda: ("", "", "", {}),
        inputs=None,
        outputs=[text_input, prediction_card, breakdown_card, label_output]
    )

demo.launch()
