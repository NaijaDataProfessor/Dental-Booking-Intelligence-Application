# Dental Booking Intelligence — Streamlit starter project

A stakeholder-facing Streamlit application for the dental inquiry booking-classification model.
Branded for **Tolulope Emuleomo | Naija Data Professor | Waterside Tree Limited**.

## Important model preparation

The uploaded notebook's current feature selection retains `inquiry_id`. This is a unique record identifier, not a meaningful patient/inquiry predictor. Retrain the model after excluding it. The app expects these 19 features:

```python
feature_columns = [
    "patient_age", "patient_gender", "patient_area",
    "distance_to_clinic_km", "marketing_source", "service_interest",
    "inquiry_urgency", "insurance_type", "prior_clinic_visits",
    "previous_booking_outcome", "response_time_minutes", "follow_up_count",
    "customer_sentiment", "price_sensitivity", "quoted_price_band",
    "payment_readiness", "appointment_availability",
    "preferred_appointment_slot", "reminder_opt_in"
]
x = df[feature_columns].copy()
y = df["booked"]
```

Retrain the complete preprocessing + Random Forest pipeline using these columns. Do not fit preprocessing separately on the full dataset; keep it inside the pipeline and fit only on training folds.

## Export the model and category options

Run this in the notebook after retraining and evaluating the corrected model. It assumes `best_rf_model` is the fitted best pipeline and `x_train` contains the corrected 19 columns.

```python
import json
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Save the entire fitted pipeline (preprocessing + classifier)
joblib.dump(best_rf_model, "best_rf_model.joblib")

# Save category values so the Streamlit form uses the same labels as training
categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns
feature_options = {
    col: sorted(x_train[col].dropna().astype(str).unique().tolist())
    for col in categorical_cols
}
with open("feature_options.json", "w", encoding="utf-8") as f:
    json.dump(feature_options, f, indent=2, ensure_ascii=False)

# Evaluate at the chosen operating threshold on the held-out test set
threshold = 0.40
test_prob = best_rf_model.predict_proba(x_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, test_pred)),
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, test_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
    "threshold": threshold,
    "evaluation_note": "Metrics calculated on the held-out test set at a 0.40 classification threshold."
}
with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Export complete.")
```

Copy these three generated files into the same folder as `app.py`:

- `best_rf_model.joblib`
- `feature_options.json`
- `model_metrics.json`

**Important:** The application must receive the same feature names and compatible data types used to train the pipeline. The notebook's existing fitted model should not be used with this app until it has been retrained using the corrected feature list.

## Run locally

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start Streamlit
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository for this project.
2. Add `app.py`, `requirements.txt`, and the three model export files.
3. Push the repository to GitHub.
4. Sign in to Streamlit Community Cloud and create an app from the repository.
5. Select `app.py` as the application entry point and deploy.

Keep the repository private if the model or data is confidential. Do not upload patient-level records or personally identifiable information. A serialized model can also reveal information about its training data, so follow the clinic's data governance and approval requirements before hosting it.

## Application sections

- Executive Overview
- Booking Prediction
- Model Performance
- About the Developer

## Operational and governance notes

- The default classification threshold is 0.40, matching the selected operating point in the notebook.
- A predicted booking probability is an estimate, not a guarantee.
- This is an operational decision-support tool, not a clinical diagnostic system.
- Validate the model on representative, recent data before stakeholder use.
- Review subgroup performance, calibration, drift, access controls and data retention.
- Do not enter patient names, phone numbers, addresses or other direct identifiers into the app.
- Treat predictions as one input to staff judgement, not an automated decision to deny service.

## Developer and contact

**Tolulope Emuleomo — Naija Data Professor**  
Waterside Tree Limited

- LinkedIn: https://www.linkedin.com/in/tolulope-emuleomo-77a231270/
- X: https://x.com/naijadataprof
- Medium: https://medium.com/@dataprofessor_
- Website: https://watersidetree.com
- Email: dataprofessor290@gmail.com
