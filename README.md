# 🦷 Dental Healthcare Inquiry & Booking Prediction System

### An Explainable Machine Learning Approach to Identifying Patients Unlikely to Book Dental Appointments

## 📌 Project Overview

Dental healthcare centres receive patient inquiries through multiple communication channels. However, not every inquiry results in a confirmed appointment.

Some patients express interest in dental services but eventually fail to book appointments due to factors such as payment readiness, pricing concerns, customer experience, appointment availability and previous booking behaviour.

This project develops a machine learning classification system to predict whether a patient inquiry will result in a confirmed dental appointment.

The primary business objective is to identify patients who are unlikely to book, enabling dental healthcare providers to prioritise follow-up activities, understand potential booking barriers and improve patient conversion rates.

The project also incorporates SHAP (SHapley Additive exPlanations) to interpret model predictions and identify the factors influencing patients' booking decisions.

## 🎯 Business Objectives

- Identify patient inquiries that are unlikely to convert into appointments.
- Understand the factors associated with unsuccessful booking outcomes.
- Evaluate patient behaviour and booking patterns across different inquiry channels.
- Help dental clinics prioritise follow-up activities.
- Apply explainable artificial intelligence to improve model transparency.
- Develop a lightweight predictive model suitable for application deployment.

## 🗂️ Dataset Overview

The project uses a dental healthcare inquiry dataset stored in Databricks.

The dataset is accessed through a Databricks table and converted into a Pandas DataFrame for further analysis.

The target variable is:

| Target | Description |
|---|---|
| 0 | No Booking |
| 1 | Booked |

The dataset contains patient inquiry information, demographic characteristics, financial readiness indicators, historical booking behaviour and operational variables.

The notebook initially selects up to 500,000 records for the modelling exercise.

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand patient inquiry patterns and booking behaviour.

The analysis included:

- Dataset structure and descriptive statistics.
- Duplicate and missing-value checks.
- Unique value analysis.
- Patient inquiry distribution by geographical area.
- Booked and unbooked inquiries by patient area.
- Inquiry channel distribution.
- Booking patterns across inquiry channels.
- Marketing source analysis.
- Comparison of booked and unbooked inquiries.

SQL and Python were used together to investigate the underlying data and generate analytical visualisations.

## ⚙️ Data Preparation and Preprocessing

The following preparation steps were implemented:

1. Standardisation of column names.
2. Inspection of missing values and duplicate records.
3. Separation of independent variables and the target variable.
4. Training and testing data partitioning.
5. Handling missing numerical values using median imputation.
6. Handling missing categorical values using most-frequent imputation.
7. Numerical feature scaling using StandardScaler.
8. Categorical encoding using OrdinalEncoder.
9. Class balancing using RandomUnderSampler on the training data.

The test dataset retains its original class distribution to support evaluation on data that better reflects the observed population.

## 🤖 Machine Learning Models

Two tree-based classification algorithms were explored.

### 1. Random Forest Classifier

Random Forest was implemented as the primary classification model.

The model combines multiple decision trees to identify patterns associated with patient booking outcomes.

The model was subsequently developed into a reduced-feature version using the ten predictors identified through SHAP analysis.

### 2. XGBoost Classifier

XGBoost was implemented as an additional classification experiment.

The model uses gradient boosting to identify complex relationships between patient inquiry characteristics and booking outcomes.

Both models were trained using the balanced training dataset.

## 📊 Model Evaluation

Model performance was assessed using the following evaluation metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Reliability of positive predictions |
| Recall | Ability to identify actual cases belonging to a class |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between the two classes |
| Confusion Matrix | Distribution of correct and incorrect predictions |

**Primary evaluation objective:**

Since the business focus is identifying patients who do not book appointments, particular attention is given to:

- Recall for Class 0 (No Booking).
- Precision for Class 0 (No Booking).
- F1-score for Class 0 (No Booking).

## 🎚️ Classification Threshold Optimisation

Rather than relying exclusively on the default 0.50 classification threshold, different probability thresholds were investigated.

The objective is to identify an appropriate operating threshold that improves the model's ability to identify patients unlikely to book.

The analysis evaluates thresholds between 0.10 and 0.95.

Performance is assessed using:

- No Booking precision.
- No Booking recall.
- No Booking F1-score.
- Booked precision and recall.
- Predicted booking and non-booking distributions.

The initial threshold-selection criterion is the highest F1-score for Class 0.

This provides a balance between identifying actual non-bookings and limiting incorrect non-booking predictions.

## 🧠 Explainable AI: SHAP Analysis

SHAP was incorporated to understand how individual predictors contribute to the model's predictions.

Three major visualisation techniques were used:

### 1. SHAP Feature Importance

Identifies the variables with the greatest overall influence on the model's predictions.

### 2. SHAP Summary (Beeswarm) Plot

Shows how high and low feature values influence individual predictions and the direction of their contributions.

### 3. SHAP Waterfall Plot

Explains how individual patient characteristics contribute to a specific prediction, starting from the model's baseline output.

The SHAP analysis informed the selection of a reduced set of predictors for the deployment model.

## ⭐ Selected Features for Application Deployment

Based on the global SHAP feature importance results, ten predictors were selected for the reduced-feature Random Forest model.

| No. | Predictor | Description |
|---:|---|---|
| 1 | payment_readiness | Patient's financial readiness to proceed |
| 2 | customer_sentiment | Patient sentiment during inquiry |
| 3 | previous_booking_outcome | Historical booking outcome |
| 4 | price_sensitivity | Patient sensitivity to treatment pricing |
| 5 | appointment_availability | Availability of suitable appointment slots |
| 6 | response_time_minutes | Time taken to respond to an inquiry |
| 7 | inquiry_urgency | Urgency associated with the inquiry |
| 8 | prior_clinic_visits | Previous visits to the dental clinic |
| 9 | follow_up_count | Number of follow-up interactions |
| 10 | distance_to_clinic_km | Patient's distance from the clinic |

These predictors were selected to support a more manageable application interface while retaining important financial, behavioural and operational information.

A new Random Forest pipeline was trained using these ten predictors.

The reduced-feature model is intended for subsequent application deployment.

## 🏗️ Project Workflow

The analytical workflow follows these stages:

1. Data extraction from Databricks.
2. Exploratory data analysis using SQL and Python.
3. Data cleaning and preprocessing.
4. Train-test partitioning.
5. Class balancing through random undersampling.
6. Random Forest and XGBoost model development.
7. Classification threshold evaluation.
8. SHAP-based model explainability.
9. Selection of ten deployment predictors.
10. Reduced-feature Random Forest training.
11. Threshold optimisation for No Booking.
12. Final model evaluation.
13. Model persistence using Joblib.

## 🛠️ Technologies and Libraries

| Technology | Application |
|---|---|
| Python | Machine learning and data analysis |
| SQL | Data extraction and exploration |
| Databricks | Data storage and processing |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Preprocessing, classification and evaluation |
| Imbalanced-learn | Class balancing |
| XGBoost | Gradient boosting classification |
| SHAP | Model explainability |
| Joblib | Model persistence |

## 💾 Model Persistence

The reduced-feature Random Forest pipeline is saved using Joblib.

The notebook saves the model to a Databricks Unity Catalog Volume:

`/Volumes/dental_healthcare/default/models/dental_booking/dental_booking_model_10_features.joblib`

The complete pipeline includes preprocessing and classification components.

The selected classification threshold should also be saved as a deployment configuration alongside the model.

## 🚀 Future Deployment

The next development phase is to build a Streamlit application around the reduced-feature Random Forest model.

The proposed application will allow dental healthcare personnel to:

- Enter patient inquiry information.
- Obtain booking probability estimates.
- Identify inquiries with a high probability of No Booking.
- View the model's main contributing factors.
- Prioritise patients for follow-up.
- Support data-driven patient engagement decisions.

The application is intended as a decision-support tool rather than an automatic replacement for clinical or administrative judgement.

## ⚠️ Limitations and Considerations

- Model predictions depend on the quality and representativeness of the underlying inquiry data.
- Undersampling changes the training class distribution and may affect probability calibration.
- SHAP identifies model associations and contributions, not causal relationships.
- Predictors must be available at the time the prediction is made to avoid data leakage.
- The reduced-feature model requires independent performance validation.
- Threshold selection should use a separate validation dataset, while the final test dataset must remain untouched until final evaluation.
- Historical booking outcomes and follow-up variables must be assessed for availability and potential target leakage before deployment.

## 📌 Business Impact

This project demonstrates how machine learning and explainable artificial intelligence can support dental healthcare operations beyond traditional reporting.

By identifying inquiries with a high likelihood of non-booking, dental healthcare providers can gain better visibility into potential booking losses and direct follow-up resources towards patients who may need additional assistance.

The ultimate goal is to transform patient inquiry data into actionable intelligence that supports improved booking conversion, better patient engagement and more informed operational decisions.

---

**Project Category:** Healthcare Analytics | Predictive Modelling | Explainable AI | Binary Classification

**Primary Model:** Random Forest Classifier

**Additional Model:** XGBoost Classifier

**Explainability Framework:** SHAP

**Deployment Direction:** Streamlit

**Developed by:** Tolulope Emuleomo | Naija Data Professor
