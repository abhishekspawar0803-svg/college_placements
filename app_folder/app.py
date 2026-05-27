import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="College Placements Predictor", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR

CLASSIFIER_PATH = APP_DIR / "placement_classifier.keras"
REGRESSOR_PATH = APP_DIR / "lpa_regressor.keras"
SCALER_CANDIDATES = [APP_DIR / "scaler.pkl", APP_DIR / "full_data_scaler.pkl"]
DATA_CANDIDATES = [
    APP_DIR / "indian_engineering_student_placement.csv",
    BASE_DIR.parent / "indian_engineering_student_placement.csv",
]
TARGET_CANDIDATES = [
    APP_DIR / "placement_targets.csv",
    BASE_DIR.parent / "placement_targets.csv",
]

CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female"],
    "branch": ["CSE", "IT", "ECE", "EE", "ME", "CE"],
    "communication_skill_rating": [1, 2, 3, 4, 5],
    "internship_experience": ["Yes", "No"],
    "hackathons_participated": [0, 1, 2, 3, 4, 5, 6],
    "extra_curricular_score": [1, 2, 3, 4, 5],
}

DEFAULTS = {
    "gender": "Male",
    "branch": "CSE",
    "cgpa": 8.0,
    "tenth_percentage": 80.0,
    "twelfth_percentage": 80.0,
    "communication_skill_rating": 3,
    "internship_experience": "No",
    "projects_completed": 3,
    "hackathons_participated": 1,
    "extra_curricular_score": 3,
}

NUMERIC_RANGES = {
    "cgpa": (0.0, 10.0, 8.0, 0.01),
    "tenth_percentage": (0.0, 100.0, 80.0, 0.1),
    "twelfth_percentage": (0.0, 100.0, 80.0, 0.1),
    "projects_completed": (0, 10, 3, 1),
    "hackathons_participated": (0, 10, 1, 1),
    "extra_curricular_score": (1, 5, 3, 1),
}


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_assets():
    classifier = load_model(CLASSIFIER_PATH) if CLASSIFIER_PATH.exists() else None
    regressor = load_model(REGRESSOR_PATH) if REGRESSOR_PATH.exists() else None
    scaler_path = first_existing(SCALER_CANDIDATES)
    scaler = joblib.load(scaler_path) if scaler_path else None
    return classifier, regressor, scaler, scaler_path


@st.cache_data
def load_reference_data():
    data_path = first_existing(DATA_CANDIDATES)
    target_path = first_existing(TARGET_CANDIDATES)
    x = pd.read_csv(data_path) if data_path else None
    y = pd.read_csv(target_path) if target_path else None
    return x, y, data_path, target_path


classifier, regressor, scaler, scaler_path = load_assets()
ref_x, ref_y, data_path, target_path = load_reference_data()

st.title("College Placements Predictor")
st.markdown("Interactive placement prediction app using the exported Keras models and scaler.")

if classifier is None:
    st.error("placement_classifier.keras not found beside app.py.")
    st.stop()

with st.sidebar:
    st.header("Model Files")
    st.write(f"Classifier: {'Found' if classifier is not None else 'Missing'}")
    st.write(f"Regressor: {'Found' if regressor is not None else 'Missing'}")
    st.write(f"Scaler: {scaler_path.name if scaler_path else 'Missing'}")
    st.write(f"Data: {data_path.name if data_path else 'Not found'}")
    st.write(f"Targets: {target_path.name if target_path else 'Not found'}")

st.subheader("Student Profile")
left, right = st.columns(2)

with left:
    gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"], index=CATEGORICAL_OPTIONS["gender"].index(DEFAULTS["gender"]))
    branch = st.selectbox("Branch", CATEGORICAL_OPTIONS["branch"], index=CATEGORICAL_OPTIONS["branch"].index(DEFAULTS["branch"]))
    cgpa = st.slider("CGPA", *NUMERIC_RANGES["cgpa"])
    tenth_percentage = st.slider("10th Percentage", *NUMERIC_RANGES["tenth_percentage"])
    twelfth_percentage = st.slider("12th Percentage", *NUMERIC_RANGES["twelfth_percentage"])

with right:
    communication_skill_rating = st.select_slider("Communication Skill Rating", options=CATEGORICAL_OPTIONS["communication_skill_rating"], value=DEFAULTS["communication_skill_rating"])
    internship_experience = st.selectbox("Internship Experience", CATEGORICAL_OPTIONS["internship_experience"], index=CATEGORICAL_OPTIONS["internship_experience"].index(DEFAULTS["internship_experience"]))
    projects_completed = st.slider("Projects Completed", *NUMERIC_RANGES["projects_completed"])
    hackathons_participated = st.slider("Hackathons Participated", *NUMERIC_RANGES["hackathons_participated"])
    extra_curricular_score = st.slider("Extra-curricular Score", *NUMERIC_RANGES["extra_curricular_score"])

input_dict = {
    "gender": gender,
    "branch": branch,
    "cgpa": cgpa,
    "tenth_percentage": tenth_percentage,
    "twelfth_percentage": twelfth_percentage,
    "communication_skill_rating": communication_skill_rating,
    "internship_experience": internship_experience,
    "projects_completed": projects_completed,
    "hackathons_participated": hackathons_participated,
    "extra_curricular_score": extra_curricular_score,
}

input_df = pd.DataFrame([input_dict])

if ref_x is not None:
    st.caption("Reference dataset loaded. The app will align one-hot columns to the training data where possible.")
    ref_encoded = pd.get_dummies(ref_x.drop(columns=[c for c in ["Student_ID"] if c in ref_x.columns]), drop_first=False)
    input_encoded = pd.get_dummies(input_df, drop_first=False)
    input_encoded = input_encoded.reindex(columns=ref_encoded.columns, fill_value=0)
else:
    input_encoded = pd.get_dummies(input_df, drop_first=False)

model_input = input_encoded.copy()

if scaler is not None:
    try:
        model_input_arr = scaler.transform(model_input)
    except Exception as e:
        st.warning(f"Scaler could not be applied cleanly: {e}")
        model_input_arr = model_input.values
else:
    model_input_arr = model_input.values

if st.button("Predict Placement", type="primary"):
    try:
        placement_prob = float(classifier.predict(model_input_arr, verbose=0).ravel()[0])
        placed = placement_prob >= 0.5

        col1, col2 = st.columns(2)
        with col1:
            if placed:
                st.success("Prediction: Placed")
            else:
                st.error("Prediction: Not Placed")
            st.metric("Placement Probability", f"{placement_prob:.2%}")

        with col2:
            if regressor is not None and placed:
                try:
                    salary_pred = float(regressor.predict(model_input_arr, verbose=0).ravel()[0])
                    st.metric("Estimated Salary (LPA)", f"{max(salary_pred, 0):.2f}")
                except Exception as e:
                    st.info(f"Salary model could not run: {e}")
            else:
                st.metric("Estimated Salary (LPA)", "0.00" if not placed else "N/A")

        st.subheader("Input Snapshot")
        st.dataframe(input_df, use_container_width=True)
        st.subheader("Encoded Model Input")
        st.dataframe(pd.DataFrame(model_input, columns=model_input.columns), use_container_width=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

if ref_x is not None and ref_y is not None:
    st.subheader("Dataset Preview")
    preview_cols = [c for c in ["branch", "cgpa", "tenth_percentage", "twelfth_percentage", "projects_completed"] if c in ref_x.columns]
    preview = ref_x[preview_cols].head(10).copy() if preview_cols else ref_x.head(10).copy()
    if "placement_status" in ref_y.columns:
        preview["placement_status"] = ref_y["placement_status"].head(10).values
    st.dataframe(preview, use_container_width=True)
