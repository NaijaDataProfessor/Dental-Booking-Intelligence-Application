"""
Dental Healthcare Booking Prediction App

Run:
    streamlit run dental_booking_app_update.py

Place dental_booking_model_10_features.joblib in the same directory.
Use the same Python and package versions used to train the saved pipeline.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import streamlit as st


# Streamlit page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="Dental Healthcare Booking Prediction",
    page_icon="🦷",
    layout="wide",
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "dental_booking_model_10_features.joblib"

# Replace with the threshold selected using validation data.
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

# UI bounds/defaults. Adjust these to the valid ranges in your training data.
NUMERIC_RANGES = {
    "response_time_minutes": (0, 1440, 5),
    "prior_clinic_visits": (0, 100, 0),
    "follow_up_count": (0, 100, 0),
    "distance_to_clinic_km": (0.0, 500.0, 5.0),
}


# ---------------------------------------------------------------------
# Environment information
# ---------------------------------------------------------------------

with st.expander("Environment information"):
    st.write("Python:", sys.version.split()[0])
    st.write("Scikit-learn:", sklearn.__version__)
    st.write("NumPy:", np.__version__)
    st.write("SciPy:", scipy.__version__)
    st.write("Joblib:", joblib.__version__)


# ---------------------------------------------------------------------
# Model loading and validation
# ---------------------------------------------------------------------

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def validate_model(pipeline):
    """Validate that the artifact is the expected fitted 10-feature pipeline."""
    if not hasattr(pipeline, "named_steps"):
        raise TypeError(
            "The saved artifact is not an sklearn Pipeline. "
            "Save the complete fitted preprocessing + classifier pipeline."
        )

    if "preprocessor" not in pipeline.named_steps:
        raise ValueError("Pipeline is missing the 'preprocessor' step.")

    if "classifier" not in pipeline.named_steps:
        raise ValueError("Pipeline is missing the 'classifier' step.")

    actual_features = list(getattr(pipeline, "feature_names_in_", []))
    if not actual_features:
        actual_features = list(
            getattr(
                pipeline.named_steps["preprocessor"],
                "feature_names_in_",
                [],
            )
        )

    if actual_features and actual_features != FEATURES:
        raise ValueError(
            "Model feature schema does not match the app. "
            f"Expected {FEATURES}; model has {actual_features}."
        )

    classes = list(pipeline.named_steps["classifier"].classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(
            "The classifier must contain class 0 (No Booking) "
            "and class 1 (Booked)."
        )


def get_encoder_categories(pipeline):
    """Read the fitted categorical categories from the preprocessing pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name != "cat":
            continue
        if transformer in ("drop", "passthrough"):
            continue

        encoder = None

        if hasattr(transformer, "named_steps"):
            for step in transformer.named_steps.values():
                if hasattr(step, "categories_"):
                    encoder = step
                    break
        elif hasattr(transformer, "categories_"):
            encoder = transformer

        if encoder is not None:
            return {
                column: list(categories)
                for column, categories in zip(columns, encoder.categories_)
            }

    raise ValueError(
        "Could not find the fitted OrdinalEncoder categories. "
        "Check the categorical transformer name and preprocessing pipeline."
    )


if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Place the fitted 10-feature Joblib pipeline beside this script."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    validate_model(model)
    categorical_options = get_encoder_categories(model)
except Exception as exc:
    st.error(
        "The model could not be loaded or validated. Confirm that this is "
        "the fitted 10-feature pipeline and that deployment package versions "
        "match the training environment."
    )
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------

def to_python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def get_input_data(category_map):
    values = {}

    st.subheader("Patient inquiry information")
    st.caption(
        "Enter information available at the point when the clinic wants "
        "to assess the likelihood that an inquiry will not become a booking."
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
                options = [
                    to_python_scalar(item)
                    for item in category_map.get(feature, [])
                ]

                if not options:
                    st.error(
                        f"No trained categories were found for {feature}. "
                        "Check the fitted categorical encoder."
                    )
                    values[feature] = None
                else:
                    values[feature] = st.selectbox(
                        label,
                        options=options,
                        key=f"input_{feature}",
                    )

    return values


# ---------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------

st.title("🦷 Dental Healthcare Booking Prediction")
st.write(
    "Estimate whether a patient inquiry is likely to result in a confirmed "
    "appointment. Use the result to support timely and appropriate follow-up."
)

st.info(
    "This application is a decision-support tool. Predictions are estimates, "
    "not certainties, and should not be used to deny patients access to care."
)

with st.sidebar:
    st.header("Prediction settings")
    no_booking_threshold = st.slider(
        "No Booking probability threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(DEFAULT_NO_BOOKING_THRESHOLD),
        step=0.01,
        help=(
            "Use the threshold selected on validation data. An inquiry is "
            "flagged as No Booking when its Class 0 probability reaches "
            "or exceeds this threshold."
        ),
    )
    st.caption("Class 0 = No Booking | Class 1 = Booked")

with st.form("booking_prediction_form"):
    input_values = get_input_data(categorical_options)
    submitted = st.form_submit_button(
        "Predict booking outcome",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

if submitted:
    if any(value is None for value in input_values.values()):
        st.error("Please resolve the missing categorical inputs before predicting.")
        st.stop()

    input_df = pd.DataFrame(
        [[input_values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    try:
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.named_steps["classifier"].classes_)

        no_booking_index = classes.index(0)
        booked_index = classes.index(1)

        no_booking_probability = float(probabilities[no_booking_index])
        booked_probability = float(probabilities[booked_index])

        # Apply the selected threshold to Class 0 (No Booking).
        predicted_class = (
            0 if no_booking_probability >= no_booking_threshold else 1
        )

        st.divider()
        st.subheader("Prediction result")

        col1, col2 = st.columns(2)
        col1.metric(
            "Probability of No Booking",
            f"{no_booking_probability:.1%}",
        )
        col2.metric(
            "Probability of Booking",
            f"{booked_probability:.1%}",
        )

        if predicted_class == 0:
            st.warning(
                "### Predicted outcome: No Booking\n\n"
                "This inquiry meets the configured No Booking threshold. "
                "Consider reviewing the patient's concerns and offering "
                "appropriate follow-up."
            )
        else:
            st.success(
                "### Predicted outcome: Booked\n\n"
                "This inquiry does not meet the configured No Booking "
                "threshold. Continue the normal booking process and confirm "
                "the appointment."
            )

        st.write("**Predicted probability distribution**")
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
                FEATURE_LABELS[feature]: input_values[feature]
                for feature in FEATURES
            }
            st.dataframe(
                pd.DataFrame([readable_input]),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"Decision threshold for No Booking: {no_booking_threshold:.2f}. "
            "Because the model was trained using undersampling, probability "
            "calibration should be checked using data that retains the "
            "original class distribution."
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Check that the saved model, input schema "
            "and installed package versions are compatible."
        )
        st.exception(exc)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()
st.caption(
    "Dental Healthcare Inquiry & Booking Prediction | "
    "Random Forest | Explainable Machine Learning"
)
