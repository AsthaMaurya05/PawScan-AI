"""
health_score.py — Pet Health Score Engine for PawScan AI
=========================================================
Calculates a 0-100 health score combining:
  1. Disease detection result (60% weight)
  2. User-reported symptoms (20% weight)
  3. Pet metadata — weight, age (20% weight)

USAGE:
    from health_score import calculate_health_score

    score, breakdown = calculate_health_score(
        prediction_result=predictor.predict("photo.jpg"),
        pet_species="Dog",
        pet_age=3,
        pet_weight=15,
        symptoms=["itching", "hair_loss"]
    )
    print(f"Health Score: {score}/100")
"""

SEVERITY_POINTS = {
    "none": 60,       # Healthy prediction
    "mild": 40,       # Hypersensitivity
    "moderate": 25,   # Dermatitis, Fungal
    "severe": 10      # Demodicosis, Ringworm
}

# Confidence adjustment: if model is unsure, reduce the score a bit
def _confidence_adjustment(confidence_pct):
    """Higher confidence = more trustworthy result."""
    if confidence_pct >= 85:
        return 0       # Full trust
    elif confidence_pct >= 70:
        return -3      # Slightly less trust
    elif confidence_pct >= 50:
        return -7      # Low confidence — might be wrong
    else:
        return -12     # Very low confidence

# ─── SYMPTOM SCORING ──────────────────────────────────────────
SYMPTOM_SEVERITY = {
    "none": 0,
    "itching": 1,
    "hair_loss": 1,
    "redness": 2,
    "scratching": 1,
    "lesions": 2,
    "appetite_change": 2,
    "lethargy": 3,
    "bleeding": 3,
    "weight_loss": 2,
    "swelling": 2
}

def _score_symptoms(symptoms):
    """Score user-reported symptoms. Max 20 points."""
    if not symptoms or "none" in symptoms:
        return 20, "No symptoms reported"

    total_severity = sum(SYMPTOM_SEVERITY.get(s, 1) for s in symptoms)
    num_symptoms = len(symptoms)

    if total_severity == 0:
        return 20, "No symptoms reported"
    elif total_severity <= 2:
        return 15, f"Mild symptoms: {', '.join(symptoms)}"
    elif total_severity <= 5:
        return 10, f"Moderate symptoms: {', '.join(symptoms)}"
    else:
        return 5, f"Severe symptoms: {', '.join(symptoms)}"

# ─── METADATA SCORING ─────────────────────────────────────────
# Approximate healthy weight ranges by species (in kg)
HEALTHY_WEIGHT = {
    "Dog": (10, 30),
    "Cat": (3, 5.5),
}

def _score_metadata(species, age, weight):
    """Score pet metadata. Max 20 points."""
    score = 20
    notes = []

    # Weight check
    species = species or "Dog"
    min_w, max_w = HEALTHY_WEIGHT.get(species, (10, 30))

    if weight:
        if weight < min_w * 0.7:
            score -= 6
            notes.append("Significantly underweight")
        elif weight < min_w * 0.85:
            score -= 3
            notes.append("Slightly underweight")
        elif weight > max_w * 1.3:
            score -= 6
            notes.append("Significantly overweight")
        elif weight > max_w * 1.15:
            score -= 3
            notes.append("Slightly overweight")

    # Age check — senior pets get a small penalty
    if age:
        if species == "Dog" and age > 8:
            score -= 2
            notes.append("Senior pet — monitor health closely")
        elif species == "Cat" and age > 10:
            score -= 2
            notes.append("Senior pet — monitor health closely")

    if not notes:
        notes.append("Weight and age within normal range")

    return score, "; ".join(notes)

# ─── MAIN SCORING FUNCTION ────────────────────────────────────
def calculate_health_score(prediction_result, pet_species="Dog",
                           pet_age=None, pet_weight=None,
                           symptoms=None):
    """
    Calculate overall pet health score (0-100).

    Args:
        prediction_result: dict from PawScanPredictor.predict()
        pet_species: "Dog" or "Cat"
        pet_age: age in years
        pet_weight: weight in kg
        symptoms: list of symptom strings

    Returns:
        (score, breakdown_dict)
    """
    # 1. Disease detection score (60% weight)
    severity = prediction_result.get("severity", "moderate")
    disease_score = SEVERITY_POINTS.get(severity, 25)

    confidence_adj = _confidence_adjustment(prediction_result.get("confidence_pct", 50))
    disease_score = max(0, disease_score + confidence_adj)

    disease_note = f"Detected: {prediction_result.get('display_name', 'Unknown')} ({prediction_result.get('confidence_pct', 0):.1f}% confidence)"

    # 2. Symptom score (20% weight)
    symptom_score, symptom_note = _score_symptoms(symptoms or [])

    # 3. Metadata score (20% weight)
    metadata_score, metadata_note = _score_metadata(pet_species, pet_age, pet_weight)

    # Total
    total_score = disease_score + symptom_score + metadata_score
    total_score = max(0, min(100, total_score))

    # Interpretation
    if total_score >= 80:
        status = "Your pet looks healthy! Keep up the good care."
        status_level = "good"
    elif total_score >= 60:
        status = "Minor concerns detected. Monitor closely."
        status_level = "moderate"
    elif total_score >= 40:
        status = "Health issues detected. Consider a vet consultation."
        status_level = "concerning"
    else:
        status = "Urgent vet consultation recommended."
        status_level = "urgent"

    breakdown = {
        "total_score": total_score,
        "status": status,
        "status_level": status_level,
        "disease_score": disease_score,
        "disease_note": disease_note,
        "symptom_score": symptom_score,
        "symptom_note": symptom_note,
        "metadata_score": metadata_score,
        "metadata_note": metadata_note,
        "components": {
            "Disease Detection (60%)": disease_score,
            "Reported Symptoms (20%)": symptom_score,
            "Pet Metadata (20%)": metadata_score
        }
    }

    return total_score, breakdown
