"""
Dental Healthcare Booking Prediction App
----------------------------------------
Run:
    streamlit run streamlit_app.py

Place the fitted 10-feature pipeline in the same directory:
    dental_booking_model_10_features.joblib

IMPORTANT:
Set DEFAULT_NO_BOOKING_THRESHOLD below to the threshold selected on your
validation data. Do not tune it against the final test set.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "dental_booking_model_10_features.joblib"

# Replace 0.50 with the best_threshold printed by your validation analysis.
DEFAULT_NO_BOOKING_THRESHOLD = 0.50

FEATURES = [
    "payment_readiness",
    "customer_sentiment",
    "previous_booking_outcome",
    "price_sensitivity",
    "appointment_availability",
    "response_time_minutes",
    "inquiry_urgency",
    "prior_clinic_visits",
    "follow_up_count",
    "distance_to_clinic_km",
]

NUMERIC_RANGES = {
    "response_time_minutes": (1, 1440, 5),
    "prior_clinic_visits": (0, 7, 0),
    "follow_up_count": (0, 8, 0),
    "distance_to_clinic_km": (0.2, 35.0, 5.0),
}

FEATURE_LABELS = {
    "payment_readiness": "Payment readiness",
    "customer_sentiment": "Customer sentiment",
    "previous_booking_outcome": "Previous booking outcome",
    "price_sensitivity": "Price sensitivity",
    "appointment_availability": "Appointment availability",
    "response_time_minutes": "Response time (minutes)",
    "inquiry_urgency": "Inquiry urgency",
    "prior_clinic_visits": "Prior clinic visits",
    "follow_up_count": "Follow-up count",
    "distance_to_clinic_km": "Distance to clinic (km)",
}


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def get_encoder_categories(pipeline):
    """
    Read the categories learned by the fitted OrdinalEncoder.
    This ensures the app offers the same categorical values used in training.
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "cat" and transformer not in ("drop", "passthrough"):
            encoder = transformer.named_steps.get("encoder")
            if encoder is not None:
                return {
                    column: list(categories)
                    for column, categories in zip(columns, encoder.categories_)
                }

    return {}


def get_input_data(categorical_options):
    """Collect the ten predictor values from the user."""
    values = {}

    st.subheader("Patient inquiry information")
    st.caption(
        "Enter information available at the point when the clinic wants "
        "to assess the likelihood of a booking not being completed."
    )

    left, right = st.columns(2)

    for index, feature in enumerate(FEATURES):
        container = left if index % 2 == 0 else right
        label = FEATURE_LABELS[feature]

        with container:
            if feature in NUMERIC_RANGES:
                minimum, maximum, default = NUMERIC_RANGES[feature]

                if feature == "distance_to_clinic_km":
                    values[feature] = st.number_input(
                        label,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(default),
                        step=0.1,
                    )
                else:
                    values[feature] = st.number_input(
                        label,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default),
                        step=1,
                    )
            else:
                options = categorical_options.get(feature, [])

                if options:
                    # Convert NumPy scalar values to ordinary Python values.
                    options = [
                        item.item() if isinstance(item, np.generic) else item
                        for item in options
                    ]
                    values[feature] = st.selectbox(label, options=options)
                else:
                    # Fail visibly rather than silently sending an invented category.
                    st.error(
                        f"No trained categories were found for {feature}. "
                        "Check that the saved model contains the fitted preprocessor."
                    )
                    values[feature] = None

    return values


# ---------------------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Dental Booking Prediction",
    page_icon="🦷",
    layout="wide",
)

st.title("🦷 Dental Healthcare Booking Prediction")
st.write(
    "Predict the likelihood that a patient inquiry will **not result in a "
    "confirmed appointment**. Use the result to support timely, appropriate "
    "patient follow-up."
)

st.info(
    "The model is a decision-support tool. A prediction is not a certainty "
    "and should not be used to deny care or make clinical decisions."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Place the fitted 10-feature Joblib pipeline in the same folder as "
        "streamlit_app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.exception(exc)
    st.stop()

missing_features = [
    feature for feature in FEATURES
    if feature not in getattr(model, "feature_names_in_", FEATURES)
]

# Pipeline feature_names_in_ may not be available in every scikit-learn version.
if missing_features and hasattr(model, "feature_names_in_"):
    st.error(
        "The loaded model does not match the expected ten deployment features: "
        + ", ".join(missing_features)
    )
    st.stop()

categorical_options = get_encoder_categories(model)

with st.sidebar:
    st.header("Prediction settings")
    no_booking_threshold = st.slider(
        "No Booking probability threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(DEFAULT_NO_BOOKING_THRESHOLD),
        step=0.01,
        help=(
            "Set this to the threshold selected using your validation dataset. "
            "Higher thresholds make the model more conservative when flagging "
            "an inquiry as No Booking."
        ),
    )
    st.caption(
        "Class 0 = No Booking | Class 1 = Booked"
    )

with st.form("booking_prediction_form"):
    input_values = get_input_data(categorical_options)
    submitted = st.form_submit_button(
        "Predict booking outcome",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if any(value is None for value in input_values.values()):
        st.error("Please resolve the missing categorical options before predicting.")
        st.stop()

    # Preserve the exact feature names and order used during model training.
    input_df = pd.DataFrame(
        [[input_values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    try:
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.named_steps["classifier"].classes_)

        no_booking_class_index = classes.index(0)
        booked_class_index = classes.index(1)

        no_booking_probability = float(probabilities[no_booking_class_index])
        booked_probability = float(probabilities[booked_class_index])

        # Apply the threshold to the probability of Class 0.
        predicted_class = (
            0 if no_booking_probability >= no_booking_threshold else 1
        )

        st.divider()
        st.subheader("Prediction result")

        metric1, metric2 = st.columns(2)
        metric1.metric(
            "Probability of No Booking",
            f"{no_booking_probability:.1%}",
        )
        metric2.metric(
            "Probability of Booking",
            f"{booked_probability:.1%}",
        )

        if predicted_class == 0:
            st.warning(
                "### Predicted outcome: No Booking\n"
                "This inquiry meets the configured threshold for No Booking. "
                "Consider reviewing the patient's concerns and offering an "
                "appropriate follow-up."
            )
        else:
            st.success(
                "### Predicted outcome: Booked\n"
                "This inquiry does not meet the configured No Booking threshold. "
                "Continue the normal booking process and confirm the appointment."
            )

        st.write("**Probability distribution**")
        probability_df = pd.DataFrame(
            {
                "Outcome": ["No Booking", "Booked"],
                "Probability": [
                    no_booking_probability,
                    booked_probability,
                ],
            }
        )
        st.bar_chart(
            probability_df.set_index("Outcome"),
            y="Probability",
        )

        with st.expander("Review submitted inquiry information"):
            readable_input = {
                FEATURE_LABELS[key]: value
                for key, value in input_values.items()
            }
            st.dataframe(
                pd.DataFrame([readable_input]),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"Decision threshold for No Booking: {no_booking_threshold:.2f}. "
            "The displayed probabilities may require calibration because "
            "the model was trained using undersampled data."
        )

    except Exception as exc:
        st.error(
            "Prediction could not be completed. Confirm that the input fields "
            "and saved model were created using the same ten-feature schema."
        )
        st.exception(exc)

st.divider()
st.caption(
    "Dental Healthcare Inquiry & Booking Prediction | "
    "Random Forest | Explainable Machine Learning"
)
