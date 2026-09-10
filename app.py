"""
app.py — PawScan AI Streamlit Web Application
================================================
Features:
  - Disease detection (EfficientNet-B0)
  - Health score engine (0-100)
  - LLM care recommendations (Groq Llama 3.3 70B)
  - Session state management (results persist across page navigation)
  - Clickable scan history (view past scan details)
  - Downloadable reports (HTML + Text)
"""

import os
import sys
import json
import time
import base64
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import PawScanPredictor
from health_score import calculate_health_score
from llm_advisor import generate_care_plan

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


# ─── IMAGE UTILITY ───────────────────────────────────────────
def image_to_b64(image, max_size=None, fmt="JPEG", quality=85):
    """Convert PIL Image to base64 string. Handles all image modes."""
    img = image.copy()
    if max_size:
        img.thumbnail((max_size, max_size))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_image(b64_str):
    """Convert base64 string back to PIL Image."""
    img_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_data))


# ─── SESSION STATE ───────────────────────────────────────────
def init_session_state():
    if "current_scan" not in st.session_state:
        st.session_state.current_scan = None
    if "viewing_history_scan" not in st.session_state:
        st.session_state.viewing_history_scan = None


# ─── API KEY ─────────────────────────────────────────────────
def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


# ─── CACHED LOADERS ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return PawScanPredictor(MODEL_PATH, CLASS_NAMES_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def load_disease_info():
    try:
        with open(DISEASE_INFO_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def save_history_entry(history_entry, image):
    """Save scan to history with embedded image thumbnail (base64)."""
    history = load_history()
    history_entry["image_b64"] = image_to_b64(image, max_size=300, fmt="JPEG", quality=70)
    history.append(history_entry)
    save_history(history)


# ─── VISUALIZATION ──────────────────────────────────────────
def plot_health_gauge(score):
    if score >= 80: color = "#2ecc71"
    elif score >= 60: color = "#f1c40f"
    elif score >= 40: color = "#e67e22"
    else: color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score", 'font': {'size': 20}},
        number={'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color}, 'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2, 'bordercolor': "#333",
            'steps': [
                {'range': [0, 40], 'color': '#ffeaa7'},
                {'range': [40, 60], 'color': '#fab1a0'},
                {'range': [60, 80], 'color': '#81ecec'},
                {'range': [80, 100], 'color': '#55efc4'},
            ],
            'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", font={'color': "#333", 'family': "Arial"})
    return fig


def plot_probability_bars(probabilities, display_names=None):
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]
    labels = [display_names.get(c, c) for c in classes] if display_names else classes
    sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_data)
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c'][:len(labels)]
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
    components = breakdown["components"]
    labels = list(components.keys())
    values = list(components.values())
    colors = ['#3498db', '#e74c3c', '#f39c12']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title('Health Score Breakdown', fontsize=13)
    plt.tight_layout()
    return fig


# ─── DISPLAY SCAN RESULTS ────────────────────────────────────
def display_scan_results(scan_data, disease_info, predictor, show_download=True):
    """Display full scan results. Works for both current and history scans."""
    # Handle image: could be PIL Image or base64 string
    image = scan_data.get("image")
    image_b64 = scan_data.get("image_b64")

    if image is None and image_b64:
        try:
            image = b64_to_image(image_b64)
        except:
            image = None

    if image is not None:
        # Ensure it's a fresh RGB PIL Image (avoids format attribute issues)
        if image.mode != "RGB":
            image = image.convert("RGB")

    result = scan_data["result"]
    score = scan_data["score"]
    breakdown = scan_data["breakdown"]
    care_plan = scan_data["care_plan"]
    pet_info = scan_data["pet_info"]

    st.markdown("## 📊 Scan Results")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 📸 Photo")
        if image is not None:
            st.image(image, use_container_width=True)
        else:
            st.markdown("🐾 *Photo not available*")

    with col2:
        st.markdown("#### 💯 Health Score")
        st.plotly_chart(plot_health_gauge(score), use_container_width=True)
        status_level = breakdown["status_level"]
        if status_level == "good":
            st.success(breakdown["status"])
        elif status_level in ("moderate", "concerning"):
            st.warning(breakdown["status"])
        else:
            st.error(breakdown["status"])

    with col3:
        st.markdown("#### 🎯 Detection")
        st.metric("Condition", result["display_name"])
        st.metric("Confidence", f"{result['confidence_pct']:.1f}%")
        st.metric("Severity", result["severity"].title())

    # Probability chart
    st.markdown("---")
    st.markdown("### AI Detection — All Conditions")
    st.pyplot(plot_probability_bars(result["all_probabilities"], predictor.display_names))

    # Score breakdown
    col_score, col_breakdown = st.columns([1, 1])
    with col_score:
        st.markdown("### Score Breakdown")
        st.pyplot(plot_score_breakdown(breakdown))
    with col_breakdown:
        st.markdown("### Score Components")
        for component, points in breakdown["components"].items():
            max_pts = component.split('(')[1].rstrip(')').split('%')[0].strip()
            st.markdown(f"**{component}:** {points:.0f} / {max_pts} pts")
        st.markdown("")
        st.info(f"📋 **Disease Detection:** {breakdown['disease_note']}")
        st.info(f"🩺 **Symptoms:** {breakdown['symptom_note']}")
        st.info(f"📊 **Metadata:** {breakdown['metadata_note']}")

    # Care plan
    st.markdown("---")
    st.markdown("### 🩺 Care Plan")
    triage = care_plan.get("triage", "ROUTINE")
    triage_colors = {"URGENT": "🔴", "NON-URGENT": "🟡", "ROUTINE": "🟢"}
    st.markdown(f"### {triage_colors.get(triage, '🟢')} Triage: {triage}")
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

    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** " + care_plan.get("disclaimer", "This AI assessment is preliminary and not a substitute for professional veterinary diagnosis. Always consult a licensed veterinarian."))

    # Download Report
    if show_download:
        st.markdown("---")
        st.markdown("### 📄 Download Report")

        if image is not None:
            html_report = generate_html_report(
                image=image, result=result, score=score, breakdown=breakdown,
                care_plan=care_plan, pet_name=pet_info.get("name", ""),
                pet_species=pet_info.get("species", "Dog"), pet_breed=pet_info.get("breed", ""),
                pet_age=pet_info.get("age", 3), pet_weight=pet_info.get("weight", 15),
                symptoms=pet_info.get("symptoms", []), disease_info=disease_info
            )
            pet_name_clean = (pet_info.get("name", "pet") or "pet").replace(" ", "_")
            st.download_button(
                label="📥 Download Full Report (HTML)",
                data=html_report.encode("utf-8"),
                file_name=f"pawscan_report_{pet_name_clean}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html", use_container_width=True
            )

        text_report = generate_text_report(
            result=result, score=score, breakdown=breakdown, care_plan=care_plan, pet_info=pet_info
        )
        pet_name_clean = (pet_info.get("name", "pet") or "pet").replace(" ", "_")
        st.download_button(
            label="📝 Download Summary (Text)",
            data=text_report.encode("utf-8"),
            file_name=f"pawscan_summary_{pet_name_clean}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain", use_container_width=True
        )

    if result["is_healthy"] and score >= 80:
        st.balloons()


# ─── REPORT GENERATION ───────────────────────────────────────
def generate_html_report(image, result, score, breakdown, care_plan,
                         pet_name, pet_species, pet_breed, pet_age,
                         pet_weight, symptoms, disease_info):
    """Generate a self-contained HTML report."""
    img_b64 = image_to_b64(image, fmt="PNG")
    symptoms_str = ", ".join(symptoms) if symptoms and "None" not in symptoms else "No symptoms reported"

    triage = care_plan.get("triage", "ROUTINE")
    triage_colors_map = {"URGENT": "#e74c3c", "NON-URGENT": "#f39c12", "ROUTINE": "#2ecc71"}
    triage_color = triage_colors_map.get(triage, "#2ecc71")

    if score >= 80: score_color = "#2ecc71"
    elif score >= 60: score_color = "#f1c40f"
    elif score >= 40: score_color = "#e67e22"
    else: score_color = "#e74c3c"

    prob_bars_html = ""
    sorted_probs = sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True)
    bar_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
    for i, (cls, prob) in enumerate(sorted_probs):
        display = result.get("display_name", cls) if cls == result["predicted_class"] else cls
        pct = prob * 100
        color = bar_colors[i % len(bar_colors)]
        prob_bars_html += f'<div style="margin-bottom: 8px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600;">{display}</span><span style="font-size: 13px; font-weight: bold; color: {color};">{pct:.1f}%</span></div><div style="background: #f0f0f0; border-radius: 4px; height: 20px;"><div style="background: {color}; border-radius: 4px; height: 20px; width: {pct}%;"></div></div></div>'

    next_steps_html = "".join(f"<li>{step}</li>" for step in care_plan.get("next_steps", []))
    home_care_html = "".join(f"<li>{tip}</li>" for tip in care_plan.get("home_care", []))
    watch_for_html = "".join(f"<li>{item}</li>" for item in care_plan.get("watch_for", []))

    disease_key = result["predicted_class"]
    disease_desc = disease_treatment = disease_contagious = disease_symptoms_str = ""
    if disease_key in disease_info:
        info = disease_info[disease_key]
        disease_desc = info.get("description", "N/A")
        disease_treatment = info.get("treatment_approach", "N/A")
        disease_contagious = "Yes" if info.get("contagious_to_humans") else "No"
        disease_symptoms_str = ", ".join(info.get("common_symptoms", []))

    components_html = ""
    for component, points in breakdown["components"].items():
        max_pts = component.split('(')[1].rstrip(')').split('%')[0].strip()
        components_html += f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;">{component}</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold;">{points:.0f} / {max_pts}</td></tr>'

    report_datetime = datetime.now().strftime("%Y-%m-%d at %H:%M")

    disease_section = ""
    if disease_desc:
        disease_section = f'<div class="section"><h2>Disease Information</h2><p style="font-size: 14px;"><strong>Condition:</strong> {result["display_name"]}</p><p style="font-size: 14px;"><strong>Description:</strong> {disease_desc}</p><p style="font-size: 14px;"><strong>Treatment:</strong> {disease_treatment}</p><p style="font-size: 14px;"><strong>Common symptoms:</strong> {disease_symptoms_str}</p><p style="font-size: 14px;"><strong>Contagious to humans:</strong> {disease_contagious}</p></div>'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PawScan AI — Health Report for {pet_name or "Pet"}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; color: #333; }}
.container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
.header h1 {{ margin: 0; font-size: 28px; }} .header p {{ margin: 5px 0 0 0; opacity: 0.9; font-size: 14px; }}
.section {{ padding: 25px 30px; border-bottom: 1px solid #f0f0f0; }}
.section h2 {{ font-size: 18px; color: #2c3e50; margin: 0 0 15px 0; border-left: 4px solid #667eea; padding-left: 10px; }}
.pet-info {{ display: flex; flex-wrap: wrap; gap: 15px; }} .pet-info-item {{ flex: 1; min-width: 120px; }}
.pet-info-label {{ font-size: 11px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }} .pet-info-value {{ font-size: 16px; font-weight: 600; margin-top: 2px; }}
.result-grid {{ display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }}
.result-photo {{ flex: 0 0 300px; }} .result-photo img {{ width: 100%; border-radius: 8px; }} .result-details {{ flex: 1; min-width: 250px; }}
.score-circle {{ width: 120px; height: 120px; border-radius: 50%; border: 8px solid {score_color}; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto; }}
.score-number {{ font-size: 36px; font-weight: bold; color: {score_color}; }} .score-label {{ text-align: center; font-size: 14px; color: #666; }}
.triage-badge {{ display: inline-block; padding: 6px 20px; border-radius: 20px; font-size: 14px; font-weight: bold; color: white; background: {triage_color}; margin-bottom: 10px; }}
.condition-name {{ font-size: 22px; font-weight: bold; margin: 5px 0; }} .confidence {{ font-size: 14px; color: #666; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }} th {{ text-align: left; padding: 8px; background: #f8f9fa; border-bottom: 2px solid #ddd; font-size: 13px; }}
ul {{ margin: 8px 0; padding-left: 20px; }} li {{ margin-bottom: 6px; font-size: 14px; }}
.disclaimer {{ background: #fff3cd; padding: 15px 30px; font-size: 12px; color: #856404; }}
.footer {{ padding: 20px 30px; text-align: center; font-size: 12px; color: #999; }}
</style></head><body><div class="container">
<div class="header"><h1>🐾 PawScan AI — Health Report</h1><p>Generated on {report_datetime}</p></div>
<div class="section"><h2>Pet Information</h2><div class="pet-info">
<div class="pet-info-item"><div class="pet-info-label">Name</div><div class="pet-info-value">{pet_name or "Unnamed"}</div></div>
<div class="pet-info-item"><div class="pet-info-label">Species</div><div class="pet-info-value">{pet_species}</div></div>
<div class="pet-info-item"><div class="pet-info-label">Breed</div><div class="pet-info-value">{pet_breed or "Unknown"}</div></div>
<div class="pet-info-item"><div class="pet-info-label">Age</div><div class="pet-info-value">{pet_age} years</div></div>
<div class="pet-info-item"><div class="pet-info-label">Weight</div><div class="pet-info-value">{pet_weight} kg</div></div>
<div class="pet-info-item"><div class="pet-info-label">Symptoms</div><div class="pet-info-value">{symptoms_str}</div></div>
</div></div>
<div class="section"><h2>Scan Results</h2><div class="result-grid">
<div class="result-photo"><img src="data:image/png;base64,{img_b64}" alt="Pet Photo"></div>
<div class="result-details"><div class="score-circle"><div class="score-number">{score}</div></div>
<div class="score-label">Health Score (out of 100)</div>
<div style="margin-top: 15px;"><div class="triage-badge">{triage}</div>
<div class="condition-name">{result["display_name"]}</div>
<div class="confidence">Confidence: {result["confidence_pct"]:.1f}% | Severity: {result["severity"].title()}</div></div>
</div></div></div>
<div class="section"><h2>AI Detection — All Conditions</h2>{prob_bars_html}</div>
<div class="section"><h2>Health Score Breakdown</h2>
<table><thead><tr><th>Component</th><th style="text-align: right;">Score</th></tr></thead><tbody>
{components_html}
<tr style="background: #f8f9fa; font-weight: bold;"><td style="padding: 10px;">Total Health Score</td><td style="padding: 10px; text-align: right; font-size: 18px; color: {score_color};">{score} / 100</td></tr>
</tbody></table>
<div style="margin-top: 10px; font-size: 13px; color: #666;">
<p><strong>Disease Detection:</strong> {breakdown["disease_note"]}</p>
<p><strong>Symptoms:</strong> {breakdown["symptom_note"]}</p>
<p><strong>Metadata:</strong> {breakdown["metadata_note"]}</p>
</div></div>
<div class="section"><h2>Care Plan</h2><div class="triage-badge">{triage}</div>
<p style="font-size: 15px; margin: 10px 0;">{care_plan.get("summary", "")}</p>
<h3 style="font-size: 15px; color: #2c3e50; margin: 15px 0 5px 0;">Recommended Next Steps</h3><ul>{next_steps_html}</ul>
<h3 style="font-size: 15px; color: #2c3e50; margin: 15px 0 5px 0;">Home Care Tips</h3><ul>{home_care_html}</ul>
<h3 style="font-size: 15px; color: #2c3e50; margin: 15px 0 5px 0;">Watch For</h3><ul>{watch_for_html}</ul>
</div>
{disease_section}
<div class="disclaimer"><strong>Disclaimer:</strong> {care_plan.get("disclaimer", "This AI assessment is preliminary and not a substitute for professional veterinary diagnosis. Always consult a licensed veterinarian.")}</div>
<div class="footer">PawScan AI — AI-Powered Pet Health Assessment<br>Report generated on {report_datetime} | Powered by EfficientNet-B0 + Llama 3.3 70B</div>
</div></body></html>"""


def generate_text_report(result, score, breakdown, care_plan, pet_info):
    symptoms = pet_info.get("symptoms", [])
    symptoms_str = ", ".join(symptoms) if symptoms and "None" not in symptoms else "No symptoms reported"
    triage = care_plan.get("triage", "ROUTINE")
    report = f"""PAWSCAN AI — PET HEALTH REPORT
================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

PET INFORMATION
----------------
Name: {pet_info.get("name", "Unnamed")}
Species: {pet_info.get("species", "Dog")}
Breed: {pet_info.get("breed", "Unknown")}
Age: {pet_info.get("age", "Unknown")} years
Weight: {pet_info.get("weight", "Unknown")} kg
Symptoms: {symptoms_str}

SCAN RESULTS
------------
Detected Condition: {result["display_name"]}
Confidence: {result["confidence_pct"]:.1f}%
Severity: {result["severity"].title()}
Health Score: {score}/100

ALL CONDITION PROBABILITIES
---------------------------
"""
    for cls, prob in sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True):
        report += f"{cls}: {prob*100:.1f}%\n"
    report += f"""
HEALTH SCORE BREAKDOWN
----------------------
Disease Detection (60%): {breakdown["components"]["Disease Detection (60%)"]:.0f}/60
Reported Symptoms (20%): {breakdown["components"]["Reported Symptoms (20%)"]:.0f}/20
Pet Metadata (20%): {breakdown["components"]["Pet Metadata (20%)"]:.0f}/20
Total: {score}/100

{breakdown["disease_note"]}
{breakdown["symptom_note"]}
{breakdown["metadata_note"]}

CARE PLAN
---------
Triage: {triage}
Summary: {care_plan.get("summary", "")}

Recommended Next Steps:
"""
    for step in care_plan.get("next_steps", []):
        report += f"  - {step}\n"
    report += "\nHome Care Tips:\n"
    for tip in care_plan.get("home_care", []):
        report += f"  - {tip}\n"
    report += "\nWatch For:\n"
    for item in care_plan.get("watch_for", []):
        report += f"  - {item}\n"
    report += f"""
DISCLAIMER
----------
{care_plan.get("disclaimer", "This AI assessment is preliminary and not a substitute for professional veterinary diagnosis. Always consult a licensed veterinarian.")}

---
PawScan AI — Powered by EfficientNet-B0 + Llama 3.3 70B
"""
    return report


# ─── MAIN APP ────────────────────────────────────────────────
def main():
    init_session_state()

    st.markdown("""
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <h1 style='font-size: 2.5em; margin-bottom: 5px;'>🐾 PawScan AI</h1>
        <p style='font-size: 1.2em; color: #666;'>AI-Powered Pet Health Assessment — Upload a photo, get instant insights</p>
    </div>
    """, unsafe_allow_html=True)

    predictor = load_model()
    disease_info = load_disease_info()

    if predictor is None:
        st.error("⚠️ Model not found. Make sure `models/pawscan_model.pth` exists.")
        st.info("Place the trained model file in the `models/` directory.")
        return

    # ─── SIDEBAR ─────────────────────────────────────────────
    st.sidebar.markdown("### 🐾 PawScan AI")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Pet Profile")
    pet_name = st.sidebar.text_input("Pet Name", value="", placeholder="e.g. Bruno")
    pet_species = st.sidebar.selectbox("Species", ["Dog", "Cat"])
    pet_breed = st.sidebar.text_input("Breed", value="", placeholder="e.g. Labrador")
    pet_age = st.sidebar.number_input("Age (years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    pet_weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", ["🔍 New Scan", "📋 Scan History", "ℹ️ About"])
    auto_api_key = get_api_key()

    # ═════════════════════════════════════════════════════════
    # CHECK FOR HISTORY SCAN VIEWING — runs before page routing
    # so it works regardless of which sidebar page is selected
    # ═════════════════════════════════════════════════════════
    if st.session_state.viewing_history_scan is not None:
        scan_data = st.session_state.viewing_history_scan
        st.markdown(f"#### 📋 Viewing Scan from {scan_data.get('timestamp', 'History')}")

        display_scan_results(scan_data, disease_info, predictor, show_download=True)

        st.markdown("---")
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            if st.button("🔄 Start New Scan", use_container_width=True, type="primary"):
                st.session_state.viewing_history_scan = None
                st.rerun()
            if st.button("← Back to History", use_container_width=True):
                st.session_state.viewing_history_scan = None
                st.rerun()
        return

    # ─── NEW SCAN PAGE ───────────────────────────────────────
    if page == "🔍 New Scan":

        # If we have a current scan, show it (persists across page navigation)
        if st.session_state.current_scan is not None:
            scan_data = st.session_state.current_scan
            display_scan_results(scan_data, disease_info, predictor, show_download=True)

            st.markdown("---")
            col_n1, col_n2, col_n3 = st.columns([1, 2, 1])
            with col_n2:
                if st.button("🔄 Start New Scan", use_container_width=True, type="primary"):
                    st.session_state.current_scan = None
                    st.rerun()
            return

        # Upload + Scan form
        st.markdown("### Upload a Photo of Your Pet")
        st.markdown("Take or upload a clear photo of your pet's skin area. The AI will analyze it for common skin conditions.")
        st.info(
            "📸 **Photo Tips for Best Results:**\n"
            "- Take a **close-up** of the affected skin area (not a full body shot)\n"
            "- Ensure **good lighting** (natural daylight is best)\n"
            "- The affected area should **fill most of the frame**\n"
            "- Avoid blurry or dark photos"
        )

        col_upload, col_info = st.columns([1, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Choose an image...", type=['jpg', 'jpeg', 'png'],
                help="JPG or PNG. Best results with close-up, well-lit photos of the skin area."
            )
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Photo", use_container_width=True)

        with col_info:
            st.markdown("#### Reported Symptoms")
            st.markdown("Select any symptoms you've noticed:")
            symptoms = st.multiselect("Symptoms", SYMPTOM_OPTIONS, default=["None"], label_visibility="collapsed")
            if "None" in symptoms and len(symptoms) > 1:
                symptoms = [s for s in symptoms if s != "None"]
            st.markdown("")

            if auto_api_key:
                st.success("✅ LLM Care Plan enabled (API key loaded from secrets)")
                api_key = auto_api_key
            else:
                with st.expander("⚙️ Advanced — LLM Care Plan (Optional)"):
                    st.markdown("Enter a free Groq API key for AI-generated care recommendations.")
                    st.markdown("Get one at [console.groq.com](https://console.groq.com) — free, no credit card.")
                    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
                    st.markdown("Leave empty to use built-in care recommendations.")

        st.markdown("---")

        if uploaded_file:
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                scan_clicked = st.button("🔍 Scan Now", use_container_width=True, type="primary")

            if scan_clicked:
                with st.spinner("🤖 AI is analyzing your pet's photo..."):
                    time.sleep(0.5)
                    image = Image.open(uploaded_file).convert("RGB")
                    result = predictor.predict(image)

                    score, breakdown = calculate_health_score(
                        prediction_result=result, pet_species=pet_species,
                        pet_age=pet_age, pet_weight=pet_weight, symptoms=symptoms
                    )

                    care_plan = generate_care_plan(
                        prediction=result, pet_species=pet_species,
                        pet_breed=pet_breed or "Unknown", pet_age=pet_age,
                        pet_weight=pet_weight, symptoms=symptoms,
                        api_key=api_key if 'api_key' in locals() else auto_api_key
                    )

                    # Store in session state — use base64 for image (avoids PIL format issues)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    pet_info_dict = {
                        "name": pet_name, "species": pet_species, "breed": pet_breed,
                        "age": pet_age, "weight": pet_weight, "symptoms": symptoms
                    }

                    st.session_state.current_scan = {
                        "image_b64": image_to_b64(image, max_size=800, fmt="JPEG", quality=85),
                        "result": result,
                        "score": score,
                        "breakdown": breakdown,
                        "care_plan": care_plan,
                        "pet_info": pet_info_dict,
                        "timestamp": timestamp
                    }

                    # Save to history
                    history_entry = {
                        "timestamp": timestamp,
                        "pet_name": pet_name or "Unnamed",
                        "species": pet_species,
                        "predicted_class": result["predicted_class"],
                        "display_name": result["display_name"],
                        "confidence": result["confidence_pct"],
                        "health_score": score,
                        "severity": result["severity"],
                        "symptoms": symptoms,
                        "result": result,
                        "score": score,
                        "breakdown": breakdown,
                        "care_plan": care_plan,
                        "pet_info": pet_info_dict
                    }
                    save_history_entry(history_entry, image)

                st.rerun()
        else:
            st.info("👆 Upload a photo to start the scan.")

    # ─── SCAN HISTORY PAGE ────────────────────────────────────
    elif page == "📋 Scan History":
        st.markdown("### 📋 Scan History")
        st.markdown("Track your pet's health over time — click on any scan to view full details.")

        history = load_history()

        if not history:
            st.info("No scans yet. Run your first scan from the 'New Scan' page!")
        else:
            if len(history) > 1:
                st.markdown("#### 📈 Health Score Trend")
                scores = [h["health_score"] for h in history]
                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(
                    x=list(range(len(scores))), y=scores, mode='lines+markers',
                    name='Health Score', line=dict(color='#2ecc71', width=3), marker=dict(size=10)
                ))
                trend_fig.update_layout(
                    xaxis_title="Scan #", yaxis_title="Health Score",
                    yaxis=dict(range=[0, 100]), height=300,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(trend_fig, use_container_width=True)

            st.markdown("#### 📝 All Scans (Click to View Details)")

            for i, entry in enumerate(reversed(history)):
                idx = len(history) - i
                col_thumb, col_info, col_btn = st.columns([1, 3, 1])

                with col_thumb:
                    if "image_b64" in entry:
                        try:
                            img = b64_to_image(entry["image_b64"])
                            st.image(img, width=80)
                        except:
                            st.markdown("🐾")
                    else:
                        st.markdown("🐾")

                with col_info:
                    st.markdown(f"**Scan #{idx}** — {entry['timestamp']}")
                    st.markdown(f"🐾 {entry['pet_name']} ({entry['species']}) | "
                                f"🔍 {entry['display_name']} | "
                                f"💯 Score: {entry['health_score']}/100")

                with col_btn:
                    if st.button("View", key=f"view_{i}", use_container_width=True):
                        # Load full scan data for viewing
                        st.session_state.viewing_history_scan = {
                            "image_b64": entry.get("image_b64"),
                            "result": entry.get("result", {}),
                            "score": entry.get("score", entry.get("health_score", 0)),
                            "breakdown": entry.get("breakdown", {}),
                            "care_plan": entry.get("care_plan", {}),
                            "pet_info": entry.get("pet_info", {
                                "name": entry.get("pet_name", ""),
                                "species": entry.get("species", "Dog"),
                                "breed": "", "age": 3, "weight": 15,
                                "symptoms": entry.get("symptoms", [])
                            }),
                            "timestamp": entry.get("timestamp", "")
                        }
                        st.rerun()

                st.markdown("---")

            if st.button("🗑️ Clear History"):
                save_history([])
                st.rerun()

    # ─── ABOUT PAGE ───────────────────────────────────────────
    elif page == "ℹ️ About":
        st.markdown("### ℹ️ About PawScan AI")
        st.markdown("""
        **PawScan AI** is an AI-powered pet health assessment platform that detects common skin conditions in pets from a single photo.

        #### How It Works
        1. **Upload** a close-up photo of your pet's skin area
        2. **AI Detection** — An EfficientNet-B0 model analyzes the image for 6 skin conditions
        3. **Health Score** — A 0-100 score is calculated from scan results + symptoms + pet metadata
        4. **Care Plan** — Get AI-generated recommendations for next steps and home care
        5. **Download Report** — Save the full results as an HTML or text file

        #### 📸 Photo Tips for Best Results
        - Take a **close-up** of the affected skin area (not a full body shot)
        - Ensure **good lighting** (natural daylight is best)
        - The affected area should **fill most of the frame**
        - Avoid blurry or dark photos

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
