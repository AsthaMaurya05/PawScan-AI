"""
llm_advisor.py — LLM-powered Care Recommendations for PawScan AI
=================================================================
Uses Groq's free Llama 3.3 70B API to generate veterinary care plans.

Get your free API key at: https://console.groq.com
No credit card needed. 14,400 requests/day free.

USAGE:
    from llm_advisor import generate_care_plan

    care_plan = generate_care_plan(
        prediction=result,
        pet_species="Dog",
        pet_breed="Labrador",
        pet_age=3,
        pet_weight=15,
        symptoms=["itching"],
        api_key="your_groq_api_key"
    )
"""

import os
import json
import requests


def _format_symptoms(symptoms):
    """Render the symptom selection for the LLM prompt.

    Handles both UI labels ("Hair loss") and snake_case keys, and treats a
    'None' selection as no symptoms.
    """
    real = [s for s in (symptoms or []) if s.strip() and s.strip().lower() != "none"]
    return ", ".join(real) if real else "No symptoms reported"


def generate_care_plan(prediction, pet_species="Dog", pet_breed="Unknown",
                       pet_age=None, pet_weight=None, symptoms=None,
                       api_key=None):
    """
    Generate a care plan using Groq's Llama API.

    Falls back to a static care plan if API is unavailable.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        return _static_care_plan(prediction, pet_species, symptoms)

    # Build the prompt
    disease = prediction.get("display_name", prediction.get("predicted_class", "Unknown"))
    confidence = prediction.get("confidence_pct", 0)
    severity = prediction.get("severity", "moderate")
    all_probs = prediction.get("all_probabilities", {})

    symptoms_str = _format_symptoms(symptoms)

    prompt = f"""You are a veterinary AI assistant for PawScan AI, a pet health assessment platform.
Based on the following AI scan results, provide a clear, structured care recommendation for the pet parent.

PET INFO:
- Species: {pet_species}
- Breed: {pet_breed}
- Age: {pet_age or 'Unknown'} years
- Weight: {pet_weight or 'Unknown'} kg
- Reported symptoms: {symptoms_str}

AI SCAN RESULT:
- Detected condition: {disease}
- Confidence: {confidence}%
- Severity: {severity}
- All class probabilities: {json.dumps(all_probs, indent=2)}

Provide your response in EXACTLY this format (no markdown, plain text):

SUMMARY: [2-3 sentence plain-English summary of what was detected and what it means]

TRIAGE: [URGENT or NON-URGENT or ROUTINE]

NEXT_STEPS:
- [First recommended action]
- [Second recommended action]
- [Third recommended action]

HOME_CARE:
- [First home care tip]
- [Second home care tip]

WATCH_FOR:
- [First symptom to watch for]
- [Second symptom to watch for]

IMPORTANT GUIDELINES:
- Always recommend consulting a veterinarian for a definitive diagnosis
- You are providing preliminary guidance only, not a medical diagnosis
- Keep language simple and non-technical — pet parents are not veterinarians
- If the condition is contagious to humans (like ringworm), mention this clearly
- Be honest about uncertainty if confidence is below 70%
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a helpful, accurate veterinary AI assistant. Provide safe, responsible pet health guidance. Always recommend veterinary consultation."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            },
            timeout=30
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return _parse_care_plan(content)
        else:
            print(f"Groq API error: {response.status_code}")
            return _static_care_plan(prediction, pet_species, symptoms)

    except Exception as e:
        print(f"LLM API failed: {e}")
        return _static_care_plan(prediction, pet_species, symptoms)


def _parse_care_plan(text):
    """Parse the LLM text response into structured dict."""
    plan = {
        "summary": "",
        "triage": "ROUTINE",
        "next_steps": [],
        "home_care": [],
        "watch_for": [],
        "disclaimer": "This AI assessment is preliminary and not a substitute for professional veterinary diagnosis. Always consult a licensed veterinarian for health concerns."
    }

    lines = text.strip().split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("SUMMARY:"):
            plan["summary"] = line.replace("SUMMARY:", "").strip()
            current_section = "summary"
        elif line.startswith("TRIAGE:"):
            triage = line.replace("TRIAGE:", "").strip().upper()
            plan["triage"] = triage if triage in ["URGENT", "NON-URGENT", "ROUTINE"] else "ROUTINE"
            current_section = "triage"
        elif line.startswith("NEXT_STEPS:"):
            current_section = "next_steps"
        elif line.startswith("HOME_CARE:"):
            current_section = "home_care"
        elif line.startswith("WATCH_FOR:"):
            current_section = "watch_for"
        elif line.startswith("-"):
            item = line.lstrip("- ").strip()
            if current_section == "next_steps":
                plan["next_steps"].append(item)
            elif current_section == "home_care":
                plan["home_care"].append(item)
            elif current_section == "watch_for":
                plan["watch_for"].append(item)

    return plan


def _static_care_plan(prediction, species, symptoms):
    """Fallback care plan when LLM API is unavailable."""
    disease = prediction.get("predicted_class", "Unknown")
    severity = prediction.get("severity", "moderate")
    display = prediction.get("display_name", disease)

    static_plans = {
        "Healthy": {
            "summary": f"Great news! No skin abnormalities were detected in your {species.lower()}'s scan. The skin appears healthy and normal.",
            "triage": "ROUTINE",
            "next_steps": [
                "Continue regular grooming and skin checks",
                "Maintain a balanced diet and regular exercise",
                "Schedule routine veterinary check-ups every 6-12 months"
            ],
            "home_care": [
                "Brush your pet's coat regularly to keep it healthy",
                "Monitor for any changes in skin, appetite, or behavior"
            ],
            "watch_for": [
                "Any new spots, lesions, or hair loss",
                "Excessive scratching or licking"
            ]
        },
        "Hypersensitivity": {
            "summary": f"Your {species.lower()} may be experiencing an allergic skin reaction. This is usually not serious but can cause discomfort through itching and irritation.",
            "triage": "NON-URGENT",
            "next_steps": [
                "Schedule a non-urgent vet appointment for proper diagnosis",
                "Try to identify potential allergens (food, environment, fleas)",
                "Consider an antihistamine after consulting your vet"
            ],
            "home_care": [
                "Keep your pet's environment clean and dust-free",
                "Use hypoallergenic pet shampoo for baths",
                "Ensure flea prevention treatment is up to date"
            ],
            "watch_for": [
                "Increased itching or scratching",
                "Development of hives or swelling",
                "Ear inflammation or head shaking"
            ]
        },
        "Dermatitis": {
            "summary": f"Your {species.lower()} shows signs of skin inflammation (dermatitis). This causes redness, itching, and discomfort but is usually treatable.",
            "triage": "NON-URGENT",
            "next_steps": [
                "Schedule a vet appointment within 1-2 weeks",
                "Avoid applying any human skin products",
                "Take a clear photo of the affected area to show the vet"
            ],
            "home_care": [
                "Gently clean the affected area with warm water",
                "Prevent your pet from scratching or licking the area (use a cone if needed)",
                "Use a medicated pet shampoo if recommended by your vet"
            ],
            "watch_for": [
                "Spreading of the affected area",
                "Increased redness or swelling",
                "Any signs of infection (pus, foul odor)"
            ]
        },
        "Fungal_infections": {
            "summary": f"A fungal skin infection was detected on your {species.lower()}. This can cause hair loss and scaling, and some fungal infections can spread to humans.",
            "triage": "NON-URGENT",
            "next_steps": [
                "Schedule a vet appointment within 1 week for antifungal treatment",
                "Wash your hands after handling your pet",
                "Clean your pet's bedding and living area thoroughly"
            ],
            "home_care": [
                "Keep the affected area clean and dry",
                "Avoid sharing grooming tools between pets",
                "Wash pet bedding in hot water"
            ],
            "watch_for": [
                "Spreading of lesions or hair loss",
                "Development of circular bald patches",
                "Any skin changes on yourself or family members"
            ]
        },
        "demodicosis": {
            "summary": f"Signs of demodicosis (mange mites) were detected on your {species.lower()}. This condition is caused by mites living in hair follicles and needs veterinary treatment.",
            "triage": "URGENT",
            "next_steps": [
                "Schedule a vet appointment within 2-3 days",
                "Request a skin scraping test for confirmation",
                "Inform your vet about all symptoms you've noticed"
            ],
            "home_care": [
                "Do not attempt over-the-counter treatments without vet guidance",
                "Keep your pet comfortable and prevent excessive scratching",
                "Note: This type of mange is NOT contagious to humans"
            ],
            "watch_for": [
                "Increasing hair loss or skin thickening",
                "Development of secondary skin infections",
                "Changes in appetite or energy levels"
            ]
        },
        "ringworm": {
            "summary": f"Ringworm was detected on your {species.lower()}. This is a contagious fungal infection that can spread to other pets AND humans. Prompt treatment is important.",
            "triage": "URGENT",
            "next_steps": [
                "Schedule a vet appointment within 2-3 days",
                "Isolate your pet from other animals immediately",
                "Inform family members about the contagious nature"
            ],
            "home_care": [
                "Wash your hands thoroughly after handling your pet",
                "Disinfect all pet bedding, toys, and grooming tools",
                "Vacuum and clean areas where your pet has been"
            ],
            "watch_for": [
                "New circular bald patches appearing on your pet",
                "Red, scaly, or itchy patches on your own skin",
                "Signs of infection in other household pets"
            ]
        }
    }

    plan = static_plans.get(disease, {
        "summary": f"A potential skin condition ({display}) was detected on your {species.lower()}. Please consult a veterinarian for proper diagnosis and treatment.",
        "triage": "NON-URGENT" if severity != "severe" else "URGENT",
        "next_steps": [
            "Schedule a veterinary appointment for proper diagnosis",
            "Take clear photos of the affected area",
            "Monitor your pet for any changes"
        ],
        "home_care": [
            "Keep the affected area clean",
            "Prevent excessive scratching or licking"
        ],
        "watch_for": [
            "Changes in the affected area",
            "New symptoms or worsening condition"
        ]
    })

    plan["disclaimer"] = "This AI assessment is preliminary and not a substitute for professional veterinary diagnosis. Always consult a licensed veterinarian for health concerns."
    return plan
