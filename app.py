"""
app.py — PawScan AI Streamlit Web Application
================================================
The main web app that ties together:
  - Disease detection (predict.py)
  - Health scoring (health_score.py)
  - LLM care recommendations (llm_advisor.py)

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    1. Push code to GitHub
    2. Go to share.streamlit.io
    3. Connect your repo and deploy
"""

import os
import sys
import json
import time
from datetime import datetime
from io import BytesIO

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ─── PATH SETUP ──────────────────────────────────────────────
# Add src directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import PawScanPredictor
from health_score import calculate_health_score
from llm_advisor import generate_care_plan

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="PawScan AI — Pet Health Assessment",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CONSTANTS ───────────────────────────────────────────────
MODEL_PATH = "models/pawscan_model.pth"
CLASS_NAMES_PATH = "class_names.json"
DISEASE_INFO_PATH = "data/disease_info.json"
HISTORY_FILE = "scan_history.json"

SYMPTOM_OPTIONS = [
    "None", "Itching", "Hair loss", "Redness", "Excessive scratching",
    "Visible lesions", "Appetite change", "Lethargy", "Bleeding",
    "Weight loss", "Swelling"
]

# ─── CACHED MODEL LOADER ─────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model once and cache it."""
    try:
        predictor = PawScanPredictor(MODEL_PATH, CLASS_NAMES_PATH)
        return predictor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_data
def load_disease_info():
    """Load disease information database."""
    try:
        with open(DISEASE_INFO_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_history():
    """Load scan history from JSON file."""
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_history(history):
    """Save scan history to JSON file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# ─── VISUALIZATION FUNCTIONS ─────────────────────────────────
def plot_health_gauge(score):
    """Create a health score gauge chart using Plotly."""
    if score >= 80:
        color = "#2ecc71"
    elif score >= 60:
        color = "#f1c40f"
    elif score >= 40:
        color = "#e67e22"
    else:
        color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score", 'font': {'size': 20}},
        number={'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 40], 'color': '#ffeaa7'},
                {'range': [40, 60], 'color': '#fab1a0'},
                {'range': [60, 80], 'color': '#81ecec'},
                {'range': [80, 100], 'color': '#55efc4'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font={'color': "#333", 'family': "Arial"})
    return fig

def plot_probability_bars(probabilities, display_names=None):
    """Create a horizontal bar chart showing all class probabilities."""
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]

    labels = []
    if display_names:
        labels = [display_names.get(c, c) for c in classes]
    else:
        labels = classes

    # Sort by value descending
    sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_data)

    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
    colors = colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel('Probability (%)', fontsize=11)
    ax.set_title('AI Detection Results — All Conditions', fontsize=13)
    ax.set_xlim(0, 100)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def plot_score_breakdown(breakdown):
    """Create a pie chart showing score component breakdown."""
    components = breakdown["components"]
    labels = list(components.keys())
    values = list(components.values())

    colors = ['#3498db', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90,
        textprops={'fontsize': 10}
    )
    ax.set_title('Health Score Breakdown', fontsize=13)
    plt.tight_layout()
    return fig

# ─── MAIN APP ────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <h1 style='font-size: 2.5em; margin-bottom: 5px;'>🐾 PawScan AI</h1>
        <p style='font-size: 1.2em; color: #666;'>AI-Powered Pet Health Assessment — Upload a photo, get instant insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    predictor = load_model()
    disease_info = load_disease_info()

    if predictor is None:
        st.error("⚠️ Model not found. Make sure `models/pawscan_model.pth` exists.")
        st.info("Place the trained model file in the `models/` directory.")
        return

    # ─── SIDEBAR ─────────────────────────────────────────────
    st.sidebar.markdown("### 🐾 PawScan AI")
    st.sidebar.markdown("---")

    # Pet profile
    st.sidebar.markdown("#### Pet Profile")
    pet_name = st.sidebar.text_input("Pet Name", value="", placeholder="e.g. Bruno")
    pet_species = st.sidebar.selectbox("Species", ["Dog", "Cat"])
    pet_breed = st.sidebar.text_input("Breed", value="", placeholder="e.g. Labrador")
    pet_age = st.sidebar.number_input("Age (years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    pet_weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio("Navigate", ["🔍 New Scan", "📋 Scan History", "ℹ️ About"])

    # ─── NEW SCAN PAGE ───────────────────────────────────────
    if page == "🔍 New Scan":
        st.markdown("### Upload a Photo of Your Pet")
        st.markdown("Take or upload a clear photo of your pet's skin area. The AI will analyze it for common skin conditions.")

        col_upload, col_info = st.columns([1, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png'],
                help="JPG or PNG. Best results with close-up, well-lit photos."
            )

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Photo", use_container_width=True)

        with col_info:
            st.markdown("#### Reported Symptoms")
            st.markdown("Select any symptoms you've noticed:")
            symptoms = st.multiselect(
                "Symptoms",
                SYMPTOM_OPTIONS,
                default=["None"],
                label_visibility="collapsed"
            )

            # Clean symptoms
            if "None" in symptoms and len(symptoms) > 1:
                symptoms = [s for s in symptoms if s != "None"]

            st.markdown("")

            # API key for LLM (optional — stored in session)
            with st.expander("⚙️ Advanced — LLM Care Plan (Optional)"):
                st.markdown("Enter a free Groq API key for AI-generated care recommendations.")
                st.markdown("Get one at [console.groq.com](https://console.groq.com) — free, no credit card.")
                api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
                st.markdown("Leave empty to use built-in care recommendations.")

        # Scan button
        st.markdown("---")

        if uploaded_file:
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                scan_clicked = st.button("🔍 Scan Now", use_container_width=True, type="primary")

            if scan_clicked:
                # Run inference
                with st.spinner("🤖 AI is analyzing your pet's photo..."):
                    # Small delay for UX
                    time.sleep(0.5)

                    # Convert uploaded file to PIL Image
                    image = Image.open(uploaded_file)

                    # Run prediction
                    result = predictor.predict(image)

                    # Calculate health score
                    score, breakdown = calculate_health_score(
                        prediction_result=result,
                        pet_species=pet_species,
                        pet_age=pet_age,
                        pet_weight=pet_weight,
                        symptoms=symptoms
                    )

                    # Generate care plan
                    care_plan = generate_care_plan(
                        prediction=result,
                        pet_species=pet_species,
                        pet_breed=pet_breed or "Unknown",
                        pet_age=pet_age,
                        pet_weight=pet_weight,
                        symptoms=symptoms,
                        api_key=api_key if 'api_key' in locals() else ""
                    )

                # ─── RESULTS ───────────────────────────────────
                st.markdown("---")
                st.markdown("## 📊 Scan Results")

                # Save to history
                history = load_history()
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "pet_name": pet_name or "Unnamed",
                    "species": pet_species,
                    "predicted_class": result["predicted_class"],
                    "display_name": result["display_name"],
                    "confidence": result["confidence_pct"],
                    "health_score": score,
                    "severity": result["severity"]
                }
                history.append(history_entry)
                save_history(history)

                # Layout: 3 columns
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.markdown("#### 📸 Photo")
                    st.image(image, use_container_width=True)

                with col2:
                    st.markdown("#### 💯 Health Score")
                    gauge_fig = plot_health_gauge(score)
                    st.plotly_chart(gauge_fig, use_container_width=True)

                    # Status
                    status_level = breakdown["status_level"]
                    if status_level == "good":
                        st.success(breakdown["status"])
                    elif status_level == "moderate":
                        st.warning(breakdown["status"])
                    elif status_level == "concerning":
                        st.warning(breakdown["status"])
                    else:
                        st.error(breakdown["status"])

                with col3:
                    st.markdown("#### 🎯 Detection")
                    st.metric("Condition", result["display_name"])
                    st.metric("Confidence", f"{result['confidence_pct']:.1f}%")
                    st.metric("Severity", result["severity"].title())

                # ─── PROBABILITY CHART ─────────────────────────
                st.markdown("---")
                st.markdown("### AI Detection — All Conditions")
                prob_fig = plot_probability_bars(result["all_probabilities"], predictor.display_names)
                st.pyplot(prob_fig)

                # ─── SCORE BREAKDOWN ───────────────────────────
                col_score, col_breakdown = st.columns([1, 1])

                with col_score:
                    st.markdown("### Score Breakdown")
                    pie_fig = plot_score_breakdown(breakdown)
                    st.pyplot(pie_fig)

                with col_breakdown:
                    st.markdown("### Score Components")
                    for component, points in breakdown["components"].items():
                        st.markdown(f"**{component}:** {points:.0f} / {component.split('(')[1].rstrip(')')} pts")
                    st.markdown("")
                    st.info(f"📋 **Disease Detection:** {breakdown['disease_note']}")
                    st.info(f"🩺 **Symptoms:** {breakdown['symptom_note']}")
                    st.info(f"📊 **Metadata:** {breakdown['metadata_note']}")

                # ─── CARE PLAN ─────────────────────────────────
                st.markdown("---")
                st.markdown("### 🩺 Care Plan")

                # Triage badge
                triage = care_plan.get("triage", "ROUTINE")
                triage_colors = {
                    "URGENT": "🔴",
                    "NON-URGENT": "🟡",
                    "ROUTINE": "🟢"
                }
                st.markdown(f"### {triage_colors.get(triage, '🟢')} Triage: {triage}")

                # Summary
                st.markdown(f"**Summary:** {care_plan.get('summary', '')}")

                col_steps, col_care = st.columns([1, 1])

                with col_steps:
                    st.markdown("#### ✅ Recommended Next Steps")
                    for step in care_plan.get("next_steps", []):
                        st.markdown(f"- {step}")

                with col_care:
                    st.markdown("#### 🏠 Home Care Tips")
                    for tip in care_plan.get("home_care", []):
                        st.markdown(f"- {tip}")

                st.markdown("#### ⚠️ Watch For")
                for item in care_plan.get("watch_for", []):
                    st.markdown(f"- {item}")

                # Disease info
                disease_key = result["predicted_class"]
                if disease_key in disease_info:
                    info = disease_info[disease_key]
                    with st.expander("📖 Disease Information"):
                        st.markdown(f"**Description:** {info.get('description', 'N/A')}")
                        st.markdown(f"**Treatment:** {info.get('treatment_approach', 'N/A')}")
                        st.markdown(f"**Contagious to humans:** {'Yes ⚠️' if info.get('contagious_to_humans') else 'No'}")
                        if info.get("common_symptoms"):
                            st.markdown(f"**Common symptoms:** {', '.join(info['common_symptoms'])}")

                # Disclaimer
                st.markdown("---")
                st.warning("⚠️ **Disclaimer:** " + care_plan.get("disclaimer", "This AI assessment is preliminary and not a substitute for professional veterinary diagnosis. Always consult a licensed veterinarian."))

                # Success balloon for healthy results
                if result["is_healthy"] and score >= 80:
                    st.balloons()

        else:
            st.info("👆 Upload a photo to start the scan.")

    # ─── SCAN HISTORY PAGE ────────────────────────────────────
    elif page == "📋 Scan History":
        st.markdown("### 📋 Scan History")
        st.markdown("Track your pet's health over time — each scan is saved here.")

        history = load_history()

        if not history:
            st.info("No scans yet. Run your first scan from the 'New Scan' page!")
        else:
            # Show trend chart
            if len(history) > 1:
                st.markdown("#### 📈 Health Score Trend")
                scores = [h["health_score"] for h in history]
                dates = [h["timestamp"] for h in history]

                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(
                    x=list(range(len(scores))),
                    y=scores,
                    mode='lines+markers',
                    name='Health Score',
                    line=dict(color='#2ecc71', width=3),
                    marker=dict(size=10)
                ))
                trend_fig.update_layout(
                    xaxis_title="Scan #",
                    yaxis_title="Health Score",
                    yaxis=dict(range=[0, 100]),
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(trend_fig, use_container_width=True)

            # Show history table
            st.markdown("#### 📝 All Scans")
            for i, entry in enumerate(reversed(history)):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    with col1:
                        st.markdown(f"**{entry['timestamp']}**")
                    with col2:
                        st.markdown(f"🐾 {entry['pet_name']} ({entry['species']})")
                    with col3:
                        st.markdown(f"🔍 {entry['display_name']}")
                    with col4:
                        st.markdown(f"💯 Score: {entry['health_score']}/100")
                    st.markdown("---")

            # Clear history button
            if st.button("🗑️ Clear History"):
                save_history([])
                st.rerun()

    # ─── ABOUT PAGE ───────────────────────────────────────────
    elif page == "ℹ️ About":
        st.markdown("### ℹ️ About PawScan AI")

        st.markdown("""
        **PawScan AI** is an AI-powered pet health assessment platform that detects common skin conditions in pets from a single photo.

        #### How It Works
        1. **Upload** a clear photo of your pet
        2. **AI Detection** — An EfficientNet-B0 model analyzes the image for 6 skin conditions
        3. **Health Score** — A 0-100 score is calculated from scan results + symptoms + pet metadata
        4. **Care Plan** — Get AI-generated recommendations for next steps and home care

        #### Detected Conditions
        - 🟢 **Healthy** — No issues detected
        - 🟡 **Hypersensitivity** — Allergic reactions
        - 🟠 **Dermatitis** — Skin inflammation
        - 🟠 **Fungal Infection** — Fungal skin conditions
        - 🔴 **Demodicosis** — Mange mite infection
        - 🔴 **Ringworm** — Contagious fungal infection

        #### Technology
        - **Model:** EfficientNet-B0 (transfer learning from ImageNet)
        - **Framework:** PyTorch
        - **LLM:** Groq Llama 3.3 70B for care recommendations
        - **Dataset:** 4,300+ labeled pet skin disease images
        - **Training Accuracy:** ~99% | **Validation Accuracy:** ~97%

        #### ⚠️ Important Disclaimer
        This tool is for informational purposes only and is **not a substitute for professional veterinary diagnosis**. Always consult a licensed veterinarian for health concerns about your pet.
        """)

if __name__ == "__main__":
    main()
