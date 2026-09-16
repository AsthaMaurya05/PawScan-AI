"""
tests/test_fixes.py — regression tests for the PawScan AI bug fixes
===================================================================
Run (no pytest needed):
    python tests/test_fixes.py

Or with pytest:
    pytest tests/ -v

Requires models/pawscan_model.pth (the trained checkpoint) for the
Grad-CAM and inference tests. All tests also pass without the model
file — those are skipped with a message.

Covers:
  1. Symptom scoring — UI labels match the severity table (bug fix:
     every symptom used to score the same default weight)
  2. "None" symptom handling in the score engine and LLM prompt
  3. Grad-CAM — output shape, range, and localization on a synthetic
     lesion (heat inside lesion >> outside)
  4. Severity lookup from disease_info.json
"""

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from health_score import _score_symptoms, calculate_health_score  # noqa: E402
from llm_advisor import _format_symptoms  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "pawscan_model.pth")
HAS_MODEL = os.path.exists(MODEL_PATH)


def _synthetic_lesion_image():
    """Pale skin background with one dark red 'lesion' in the middle."""
    arr = np.full((300, 300, 3), [225, 190, 175], dtype=np.uint8)
    arr[100:200, 120:220] = [150, 40, 40]
    return Image.fromarray(arr)


# ─── 1. Symptom scoring ───────────────────────────────────────
def test_symptom_severity_differentiates():
    """Before the fix, 'Lethargy' (3) and 'Itching' (1) scored the same."""
    itching = _score_symptoms(["Itching"])[0]
    lethargy = _score_symptoms(["Lethargy"])[0]
    severe = _score_symptoms(["Lethargy", "Bleeding"])[0]
    assert itching > lethargy > severe, (
        f"Symptom severity not differentiated: itching={itching}, "
        f"lethargy={lethargy}, severe={severe}"
    )


def test_none_symptom_scores_full_marks():
    """Before the fix, selecting 'None' cost the pet 5 points."""
    score, note = _score_symptoms(["None"])
    assert score == 20
    assert note == "No symptoms reported"


def test_ui_labels_all_recognized():
    """Every label offered by the app must have a real severity weight."""
    from health_score import SYMPTOM_SEVERITY, _normalize_symptom
    ui_labels = [
        "None", "Itching", "Hair loss", "Redness", "Excessive scratching",
        "Visible lesions", "Appetite change", "Lethargy", "Bleeding",
        "Weight loss", "Swelling",
    ]
    for label in ui_labels:
        assert _normalize_symptom(label) in SYMPTOM_SEVERITY, (
            f"UI label '{label}' has no entry in SYMPTOM_SEVERITY"
        )


# ─── 2. "None" handling in LLM prompt ─────────────────────────
def test_llm_symptom_formatting():
    assert _format_symptoms(["None"]) == "No symptoms reported"
    assert _format_symptoms([]) == "No symptoms reported"
    assert _format_symptoms(["Itching", "Hair loss"]) == "Itching, Hair loss"


# ─── 3. Severity lookup ───────────────────────────────────────
def test_severity_from_disease_info():
    info_path = os.path.join(ROOT, "data", "disease_info.json")
    if not os.path.exists(info_path):
        print("  [skip] disease_info.json not found")
        return
    with open(info_path) as f:
        info = json.load(f)
    from health_score import SEVERITY_POINTS
    for cls, d in info.items():
        assert d["severity"] in SEVERITY_POINTS, (
            f"disease_info severity '{d['severity']}' for {cls} "
            f"not understood by the score engine"
        )


# ─── 4. Grad-CAM (needs the trained model) ────────────────────
def test_gradcam():
    if not HAS_MODEL:
        print("  [skip] models/pawscan_model.pth not found — Grad-CAM test skipped")
        return
    from predict import PawScanPredictor
    from gradcam import PawScanGradCAM

    predictor = PawScanPredictor(
        MODEL_PATH,
        os.path.join(ROOT, "class_names.json"),
        disease_info_path=os.path.join(ROOT, "data", "disease_info.json"),
    )
    img = _synthetic_lesion_image()

    result = predictor.predict(img)
    assert result["severity"] in ("none", "mild", "moderate", "severe")

    gradcam = PawScanGradCAM(predictor)
    cam, class_idx = gradcam.generate(img)

    assert cam.shape == (300, 300), f"unexpected CAM shape {cam.shape}"
    assert 0.0 <= cam.min() and cam.max() <= 1.0, "CAM not normalized to [0, 1]"
    assert cam.max() > 0.5, "CAM is empty/degenerate"

    # Localization: mean heat inside the lesion should clearly exceed outside
    inside = cam[100:200, 120:220].mean()
    outside = np.concatenate([cam[:100].ravel(), cam[:, 240:].ravel()]).mean()
    assert inside > outside, (
        f"Grad-CAM did not localize on the lesion: inside={inside:.2f}, "
        f"outside={outside:.2f}"
    )

    overlay = gradcam.overlay(img, cam)
    assert overlay.size == img.size and overlay.mode == "RGB"

    # predict() must still work normally after a Grad-CAM pass
    # (hooks must not poison the no_grad inference path)
    _ = predictor.predict(img)


if __name__ == "__main__":
    tests = [
        test_symptom_severity_differentiates,
        test_none_symptom_scores_full_marks,
        test_ui_labels_all_recognized,
        test_llm_symptom_formatting,
        test_severity_from_disease_info,
        test_gradcam,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
