# 🐾 PawScan AI

**AI-Powered Pet Health Assessment Platform**

Upload a photo of your pet → Get instant AI disease detection, health score, and care recommendations.

Built as a project for the Pawbud Health AI Engineering Intern interview.

---

## 🔗 Live Demo
<!-- Add your Streamlit Cloud URL here after deployment -->
**[Try PawScan AI →](https://your-app-url.streamlit.app)**

---

## 📋 What It Does

1. **Disease Detection** — An EfficientNet-B0 model analyzes pet photos for 6 skin conditions
2. **Health Score** — A 0-100 score combining AI prediction + symptoms + pet metadata
3. **Care Plan** — AI-generated recommendations (next steps, home care, what to watch for)
4. **Scan History** — Track health over time with a trend chart

## 🎯 Detected Conditions

| Condition | Severity | Contagious to Humans? |
|-----------|----------|----------------------|
| Healthy | None | No |
| Hypersensitivity | Mild | No |
| Dermatitis | Moderate | No |
| Fungal Infection | Moderate | Yes |
| Demodicosis | Severe | No |
| Ringworm | Severe | Yes |

## 🏗️ Architecture

```
┌──────────┐     ┌──────────────┐     ┌───────────────────┐
│ Pet Photo │────▶│ Preprocessing │────▶│ EfficientNet-B0   │
│ Upload    │     │ (Resize 224,  │     │ 6-class classifier │
│ (JPG/PNG) │     │  Normalize)   │     │ Transfer Learning  │
└──────────┘     └──────────────┘     └────────┬──────────┘
                                                 │
                                                 ▼
┌──────────┐                    ┌──────────────────────────┐
│ User Meta │──────────────────▶│    Health Score Engine    │
│ (weight,  │                    │  Weighted: 60% + 20% + 20%│
│  age,     │                    │  Output: 0-100 score      │
│ symptoms) │                    └────────┬─────────────────┘
└──────────┘                             │
                                         ▼
                            ┌──────────────────────────────┐
                            │   LLM Care Recommendations   │
                            │   (Groq Llama 3.3 70B)       │
                            │   - Triage summary           │
                            │   - Next steps               │
                            │   - Home care tips           │
                            └──────────────────────────────┘
                                         │
                                         ▼
                            ┌──────────────────────────────┐
                            │     Streamlit Web App        │
                            │  - Health score gauge        │
                            │  - Probability bar chart     │
                            │  - Care plan display         │
                            │  - Scan history + trend      │
                            └──────────────────────────────┘
```

## 📊 Model Performance

- **Architecture:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- **Dataset:** 4,315 labeled pet skin disease images (6 classes)
- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~97%
- **Training:** 20 epochs on Kaggle T4 GPU

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Disease Detection | PyTorch + EfficientNet-B0 (transfer learning) |
| Health Score | Custom weighted scoring algorithm |
| LLM Care Plan | Groq Llama 3.3 70B (free API) |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud (free) |
| Model Training | Kaggle Notebooks (free T4 GPU) |

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/AsthaMaurya05/PawScan-AI.git
cd PawScan-AI

# Install dependencies
pip install -r requirements.txt

# Place model file
# Download pawscan_model.pth and place it in models/

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
PawScan-AI/
├── app.py                     # Streamlit web app (main entry point)
├── src/
│   ├── predict.py             # Inference pipeline
│   ├── health_score.py        # Health scoring algorithm
│   └── llm_advisor.py         # Groq LLM integration
├── data/
│   └── disease_info.json      # Disease information database
├── models/
│   └── pawscan_model.pth      # Trained model weights
├── class_names.json           # Class label mapping
├── requirements.txt
└── README.md
```

## ⚠️ Disclaimer

This tool is for informational purposes only and is **not a substitute for professional veterinary diagnosis**. Always consult a licensed veterinarian for health concerns about your pet.

---

Built with ❤️ for pet health | PawScan AI
