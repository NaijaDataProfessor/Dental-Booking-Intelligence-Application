"""
Dental Healthcare Booking Prediction App

Run:
    streamlit run dental_booking_app.py

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
# Developer and consultancy contact details
# ---------------------------------------------------------------------

DEVELOPER_NAME = "Tolulope Emuleomo | Naija Data Professor"
CONSULTANCY_NAME = "Waterside Tree Limited"
CONSULTANCY_WEBSITE = "https://watersidetree.com"
CONTACT_EMAIL = "dataprofessor290@gmail.com"

SOCIAL_LINKS = {
    "LinkedIn": "https://www.linkedin.com/in/tolulope-emuleomo-77a231270/",
    "X (Twitter)": "https://x.com/naijadataprof",
    "Medium": "https://medium.com/@dataprofessor_",
    "GitHub": "https://github.com/NaijaDataProfessor",
}

# ---------------------------------------------------------------------
# Environment information
# ---------------------------------------------------------------------

with st.expander("Environment information"):
