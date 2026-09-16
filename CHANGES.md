# Changelog — Bug Fixes & Grad-CAM

## Fixed

### 1. Privacy bug — cross-user data leak in scan history (HIGH)
`app.py` wrote every scan — pet photos (embedded as base64), pet names,
weights, symptoms and full results — to a single `scan_history.json` on the
server's disk. On a shared deployment (e.g. Streamlit Community Cloud) every
visitor read from and wrote to that **same** file, so any user could open
"Scan History" and see every other user's pet photos and medical details.
Concurrent writes could also corrupt the file.

**Fix:** history now lives in `st.session_state` (per browser session, in
memory only). Each user's data is fully isolated; nothing touches disk.
`HISTORY_FILE` and all file I/O were removed. Documented the production
multi-user path (auth + per-user DB + object storage) in README →
"Scaling & Privacy".

### 2. Symptom scoring was effectively broken (HIGH)
The app's UI labels (`"Lethargy"`, `"Hair loss"`, `"Excessive scratching"`…)
never matched the keys in `health_score.py`'s `SYMPTOM_SEVERITY` table
(`"lethargy"`, `"hair_loss"`, `"scratching"…`), so every symptom silently
fell through to the default weight of 1:

- `"Lethargy"` (worst, weight 3) scored exactly the same as `"Itching"` (weight 1)
- Selecting only `"None"` was treated as a mild symptom — a perfectly healthy
  report lost 5 points ("Mild symptoms: None", 15/20 instead of 20/20)

**Fix:** added symptom normalization (`_normalize_symptom`) that maps both UI
labels and snake_case keys to one canonical form, plus explicit `"None"`
handling. Verified: `None` → 20, `Itching` → 15, `Lethargy` → 10,
`Lethargy+Bleeding` → 5.

### 3. Literal "None" sent to the LLM as a symptom (MEDIUM)
`llm_advisor.py` checked `"none" not in symptoms` (lowercase) but the app
passes `"None"`, so the care-plan prompt contained `Reported symptoms: None`
instead of "No symptoms reported". Also present in the HTML report
generator. Both now share proper filtering.

### 4. Minor
- `predict.py`: `import os` was at the *bottom* of the file (worked by
  accident of module-execution order); moved to the top.
- `predict.py`: severity was hardcoded in a second place, duplicating
  `data/disease_info.json`. The predictor now takes an optional
  `disease_info_path` and reads severity from it, with the static map kept
  only as a fallback.
- `app.py`: replaced the fragile `api_key if 'api_key' in locals()` check
  with a straightforward variable.

## Round 3 — honest evaluation pass

### 8. Verified real training facts from the checkpoint
The released `pawscan_model.pth` stores `val_acc = 96.63%`, best epoch = 15
(of a 20-epoch run), and the optimizer config (AdamW, initial LR 1e-4
cosine-decayed, weight decay 1e-4). README and the app's About page now quote
these verified numbers instead of "~99% / ~97%".

### 9. Added `src/evaluate.py` — proper held-out evaluation
The dataset (Kaggle "Dog's skin diseases", youssefmohmmed) has a train/valid
TEST split that was never used during training or early stopping. The script
evaluates the checkpoint on any split and reports accuracy, per-class
precision/recall/F1, macro-F1, weighted-F1, top confusion pairs, and saves a
confusion-matrix PNG + JSON report to `results/`. Metrics are computed with
numpy (no sklearn dependency). Folder names are matched case/separator-
insensitively. Logic verified against hand-computed examples.

### 10. Test-set results (run on Kaggle, 433 held-out images)
- Overall accuracy **97.69%**, macro-F1 **97.17%**, weighted-F1 97.69%
- Weakest class: Fungal_infections (F1 93.5%) — clinically the most ambiguous
- Strongest: ringworm 99.1% F1, demodicosis 99.0% F1 (100% recall)
- Top confusions: Healthy ↔ Fungal_infections (2+2), ringworm → demodicosis (1)
- Confusion matrix + JSON report committed under `results/`; README and the
  app's About page now quote these held-out numbers instead of
  training/validation accuracy.

## Round 2 — second audit pass (found via full-UI testing)

### 5. Grad-CAM was silently failing in the real app (HIGH)
`@st.cache_resource def load_gradcam(predictor)` could not hash the
`PawScanPredictor` argument — Streamlit raised `Cannot hash argument`, the
non-fatal wrapper caught it, and **every real scan ran without a heatmap**
(unit tests passed because they bypassed the cached function). Found only by
driving the real UI (upload → Scan Now) with Streamlit's `AppTest`.

**Fix:** renamed the argument to `_predictor` (Streamlit's convention for
unhashable, ignore-for-caching arguments).

### 6. `use_container_width` deprecated — removal after 2025-12-31 (MEDIUM)
The deployed app installs the latest Streamlit, which warns that
`use_container_width` will be removed. All 12 call sites migrated to
`width="stretch"`, and `requirements.txt` now pins `streamlit>=1.49`
(the first version supporting the new API).

### 7. Corrupt/renamed upload crashed the app with a raw traceback (LOW)
A non-image file renamed to `.jpg` raised an unhandled exception in
`Image.open`. Both the preview and the scan path now catch it and show a
friendly error instead.

### Deployment hardening
- `requirements.txt`: version floors with comments
- `.gitignore`: pycache, `.env`, `.streamlit/secrets.toml`, and the legacy
  `scan_history.json` (so it can never be committed)
- `.streamlit/config.toml`: theme, minimal toolbar, 25 MB upload cap
- `.streamlit/secrets.example.toml`: local-dev secrets template
- README: step-by-step Streamlit Cloud deployment guide

## Added

### Grad-CAM explainability (`src/gradcam.py`)
Every scan now produces a Grad-CAM heatmap (Selvaraju et al., 2017) showing
**where** the EfficientNet-B0 model looked when making its prediction —
computed on the final 7×7×1280 convolutional feature map and blended over
the photo. It appears under the photo on the results page, in scan-history
views, and as a dedicated section of the downloadable HTML report.

Grad-CAM failures are non-fatal (wrapped in try/except) so the scan itself
never breaks.

### Tests (`tests/test_fixes.py`)
Six regression tests covering the symptom fix, `"None"` handling, UI-label
coverage, the severity table, and Grad-CAM output shape/normalization/
localization. Run with `python tests/test_fixes.py` or `pytest tests/ -v`.
All 6 pass against the trained checkpoint.

## Files changed
- `app.py` — session-scoped history, Grad-CAM integration, `width="stretch"`
  migration, upload error handling, api_key cleanup
- `src/health_score.py` — symptom normalization + "None" fix
- `src/llm_advisor.py` — symptom formatting fix
- `src/predict.py` — import fix, severity from disease_info.json
- `src/gradcam.py` — **new**
- `tests/test_fixes.py` — **new**
- `README.md` — Grad-CAM section, "Scaling & Privacy" section, deployment guide
- `requirements.txt` — version floors
- `.gitignore`, `.streamlit/config.toml`, `.streamlit/secrets.example.toml` — **new**
