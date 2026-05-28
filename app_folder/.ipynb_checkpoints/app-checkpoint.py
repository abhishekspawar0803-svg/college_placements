import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="College Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #E51212;
    margin-bottom: 0.25rem;
}
.sub-title {
    color: #475569;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}
.card {
    padding: 1rem 1.25rem;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_assets():
    classifier = load_model(BASE_DIR / "placement_classifier.keras")
    regressor = load_model(BASE_DIR / "lpa_regressor.keras")
    classifier_scaler = joblib.load(BASE_DIR / "classifier_scaler.pkl")
    regressor_scaler = joblib.load(BASE_DIR / "regressor_scaler.pkl")

    with open(BASE_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    with open(BASE_DIR / "encoder_mappings.json", "r", encoding="utf-8") as f:
        mappings = json.load(f)

    return classifier, regressor, classifier_scaler, regressor_scaler, feature_columns, mappings

classifier, regressor, classifier_scaler, regressor_scaler, feature_columns, mappings = load_assets()

st.markdown("<div class='main-title'>🎓 College Placement Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Cascade pipeline: raw input → classifier → append placement_status → regressor</div>",
    unsafe_allow_html=True
)

with st.expander("Loaded assets", expanded=False):
    st.write("Classifier: `placement_classifier.keras`")
    st.write("Regressor: `lpa_regressor.keras`")
    st.write("Classifier scaler: `classifier_scaler.pkl`")
    st.write("Regressor scaler: `regressor_scaler.pkl`")
    st.write("Feature schema: `feature_columns.json`")
    st.write("Mappings: `encoder_mappings.json`")

st.markdown("### Student Profile")

def get_feature_columns_list(feature_columns_obj):
    if isinstance(feature_columns_obj, list):
        return feature_columns_obj
    if isinstance(feature_columns_obj, dict):
        for key in ["feature_columns", "columns", "input_columns", "features"]:
            if key in feature_columns_obj and isinstance(feature_columns_obj[key], list):
                return feature_columns_obj[key]
    return []

def get_mapping(mappings_dict, *possible_keys):
    for key in possible_keys:
        if key in mappings_dict and isinstance(mappings_dict[key], dict):
            return mappings_dict[key]
    return {}

branch_labels = get_mapping(mappings, "branch_labels", "branch_encoder", "branch_mapping")
part_time_labels = get_mapping(mappings, "part_time_labels", "part_time_job_labels", "part_time_mapping")
family_income_labels = get_mapping(mappings, "family_income_labels", "family_income_mapping")
city_tier_labels = get_mapping(mappings, "city_tier_labels", "city_tier_mapping")
internet_encoder = get_mapping(mappings, "internet_encoder", "internet_access_encoder", "internet_mapping")
extra_encoder = get_mapping(mappings, "extra_encoder", "extracurricular_labels", "extracurricular_mapping")

resolved_feature_columns = get_feature_columns_list(feature_columns)

col1, col2, col3 = st.columns(3)

with col1:
    branch = st.selectbox("Branch", list(branch_labels.keys()) if branch_labels else ["CSE", "IT", "ECE", "CE", "ME"])
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    tenth_percentage = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    twelfth_percentage = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    backlogs = st.number_input("Backlogs", min_value=0, max_value=20, value=0, step=1)
    study_hours_per_day = st.slider("Study Hours / Day", 0.0, 12.0, 4.0, 0.5)
    attendance_percentage = st.slider("Attendance Percentage", 0.0, 100.0, 75.0, 1.0)

with col2:
    projects_completed = st.number_input("Projects Completed", min_value=0, max_value=20, value=3, step=1)
    internships_completed = st.number_input("Internships Completed", min_value=0, max_value=10, value=1, step=1)
    coding_skill_rating = st.slider("Coding Skill Rating", 0, 10, 5, 1)
    communication_skill_rating = st.slider("Communication Skill Rating", 0, 10, 5, 1)
    aptitude_skill_rating = st.slider("Aptitude Skill Rating", 0, 10, 5, 1)
    hackathons_participated = st.number_input("Hackathons Participated", min_value=0, max_value=20, value=1, step=1)
    certifications_count = st.number_input("Certifications Count", min_value=0, max_value=30, value=2, step=1)

with col3:
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
    stress_level = st.slider("Stress Level", 0, 10, 5, 1)
    part_time_job = st.selectbox("Part Time Job", list(part_time_labels.keys()) if part_time_labels else ["Yes", "No"])
    family_income_level = st.selectbox("Family Income Level", list(family_income_labels.keys()) if family_income_labels else ["Low", "Medium", "High"])
    city_tier = st.selectbox("City Tier", list(city_tier_labels.keys()) if city_tier_labels else ["Tier 1", "Tier 2", "Tier 3"])
    internet_access = st.selectbox("Internet Access", list(internet_encoder.keys()) if internet_encoder else ["Yes", "No"])
    extracurricular_involvement = st.selectbox("Extracurricular Involvement", list(extra_encoder.keys()) if extra_encoder else ["None", "Low", "Medium", "High"])

def safe_map(mapping_dict, key, default=0):
    if not mapping_dict:
        return default

    if key in mapping_dict:
        return mapping_dict[key]

    key_str = str(key).strip()
    if key_str in mapping_dict:
        return mapping_dict[key_str]

    key_lower = key_str.lower()
    for k, v in mapping_dict.items():
        if str(k).strip().lower() == key_lower:
            return v

    return default

def encode_input(raw_input, mappings_dict):
    encoded = {
        "branch": safe_map(branch_labels, raw_input["branch"], 0),
        "cgpa": raw_input["cgpa"],
        "tenth_percentage": raw_input["tenth_percentage"],
        "twelfth_percentage": raw_input["twelfth_percentage"],
        "backlogs": raw_input["backlogs"],
        "study_hours_per_day": raw_input["study_hours_per_day"],
        "attendance_percentage": raw_input["attendance_percentage"],
        "projects_completed": raw_input["projects_completed"],
        "internships_completed": raw_input["internships_completed"],
        "coding_skill_rating": raw_input["coding_skill_rating"],
        "communication_skill_rating": raw_input["communication_skill_rating"],
        "aptitude_skill_rating": raw_input["aptitude_skill_rating"],
        "hackathons_participated": raw_input["hackathons_participated"],
        "certifications_count": raw_input["certifications_count"],
        "sleep_hours": raw_input["sleep_hours"],
        "stress_level": raw_input["stress_level"],
        "part_time_job": safe_map(part_time_labels, raw_input["part_time_job"], 0),
        "family_income_level": safe_map(family_income_labels, raw_input["family_income_level"], 0),
        "city_tier": safe_map(city_tier_labels, raw_input["city_tier"], 0),
        "internet_access": safe_map(internet_encoder, raw_input["internet_access"], 0),
        "extracurricular_involvement": safe_map(extra_encoder, raw_input["extracurricular_involvement"], 0),
    }
    return encoded

if st.button("Predict Placement & Salary", type="primary"):
    raw_input = {
        "branch": branch,
        "cgpa": cgpa,
        "tenth_percentage": tenth_percentage,
        "twelfth_percentage": twelfth_percentage,
        "backlogs": backlogs,
        "study_hours_per_day": study_hours_per_day,
        "attendance_percentage": attendance_percentage,
        "projects_completed": projects_completed,
        "internships_completed": internships_completed,
        "coding_skill_rating": coding_skill_rating,
        "communication_skill_rating": communication_skill_rating,
        "aptitude_skill_rating": aptitude_skill_rating,
        "hackathons_participated": hackathons_participated,
        "certifications_count": certifications_count,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "part_time_job": part_time_job,
        "family_income_level": family_income_level,
        "city_tier": city_tier,
        "internet_access": internet_access,
        "extracurricular_involvement": extracurricular_involvement,
    }

    encoded_input = encode_input(raw_input, mappings)
    base_df = pd.DataFrame([encoded_input])

    if resolved_feature_columns:
        base_df = base_df.reindex(columns=resolved_feature_columns, fill_value=0)

    clf_scaled = classifier_scaler.transform(base_df)
    clf_scaled = np.asarray(clf_scaled, dtype=np.float32)

    placement_prob = float(classifier.predict(clf_scaled, verbose=0).ravel()[0])
    placement_label = 1 if placement_prob >= 0.5 else 0
    placement_text = "Placed" if placement_label == 1 else "Not Placed"

    reg_df = base_df.copy()
    reg_df["placement_status"] = placement_label

    reg_scaled = regressor_scaler.transform(reg_df)
    reg_scaled = np.asarray(reg_scaled, dtype=np.float32)

    salary_lpa = float(regressor.predict(reg_scaled, verbose=0).ravel()[0])
    salary_lpa = max(0.0, salary_lpa)

    st.markdown("---")
    m1, m2 = st.columns(2)

    with m1:
        st.metric("Placement Probability", f"{placement_prob:.2%}")
        if placement_label == 1:
            st.success(f"Prediction: {placement_text}")
        else:
            st.error(f"Prediction: {placement_text}")

    with m2:
        shown_salary = salary_lpa if placement_label == 1 else 0.0
        st.metric("Estimated Salary (LPA)", f"{shown_salary:.2f}")

    with st.expander("Pipeline tables", expanded=False):
        st.write("Raw input dataframe")
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)

        st.write("Encoded classifier dataframe")
        st.dataframe(base_df, use_container_width=True)

        st.write("Regressor dataframe with appended placement_status")
        st.dataframe(reg_df, use_container_width=True)