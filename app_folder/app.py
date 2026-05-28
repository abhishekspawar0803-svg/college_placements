import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="College Placements Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium CSS
st.markdown("""
    <style>
        .stMetric {
            background-color: #f7f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
        .stMetric label {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #2c3e50;
        }
        .main-header {
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 5px;
        }
        .sub-header {
            color: #64748b;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            padding: 12px;
            transition: all 0.3s;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------
# ASSET LOADING
# -----------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_assets():
    try:
        classifier = load_model(BASE_DIR / "placement_classifier.keras")
        regressor = load_model(BASE_DIR / "lpa_regressor.keras")
        
        # Load scalers
        raw_class_scaler = joblib.load(BASE_DIR / "classifer_scaler.pkl")
        raw_reg_scaler = joblib.load(BASE_DIR / "regressor_scaler.pkl")
        
        # Handle joblib version mismatch issue where scaler unpickles as raw list/array
        def reconstruct_scaler(loaded_obj):
            if hasattr(loaded_obj, "transform"):
                return loaded_obj # It loaded correctly
            
            # If it's just raw data, log a warning but pass it back
            if isinstance(loaded_obj, (np.ndarray, list)):
                print("Warning: Scaler loaded as raw array due to sklearn/joblib mismatch. Fallback enabled.")
            return loaded_obj

        class_scaler = reconstruct_scaler(raw_class_scaler)
        reg_scaler = reconstruct_scaler(raw_reg_scaler)
        
        with open(BASE_DIR / "feature_columns.json", "r") as f:
            features = json.load(f)
            
        with open(BASE_DIR / "encoder_mappings.json", "r") as f:
            mappings = json.load(f)
            
        return classifier, regressor, class_scaler, reg_scaler, features, mappings
    except Exception as e:
        st.error(f"Failed to load required files from app_folder: {e}")
        st.stop()

classifier, regressor, class_scaler, reg_scaler, base_features, mappings = load_assets()

# -----------------
# UI LAYOUT
# -----------------
st.markdown("<h1 class='main-header'>🎓 College Placement Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Cascade Prediction Pipeline: Classification → Regression</p>", unsafe_allow_html=True)

with st.expander("🛠 System Status & Loaded Assets", expanded=False):
    cols = st.columns(6)
    cols[0].write("🤖 **Classifier:** `placement_classifier.keras`")
    cols[1].write("🤖 **Regressor:** `lpa_regressor.keras`")
    cols[2].write("📏 **Clf Scaler:** `classifer_scaler.pkl`")
    cols[3].write("📏 **Reg Scaler:** `regressor_scaler.pkl`")
    cols[4].write("📋 **Features:** `feature_columns.json`")
    cols[5].write("🗺️ **Mappings:** `encoder_mappings.json`")

# Form Inputs
with st.container():
    st.markdown("### 📝 Student Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", list(mappings["branch_labels"].keys()))
        cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0, 0.1)
        tenth = st.number_input("10th Percentage", 0.0, 100.0, 75.0, 1.0)
        twelfth = st.number_input("12th Percentage", 0.0, 100.0, 75.0, 1.0)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        study_hrs = st.slider("Study Hours/Day", 0.0, 12.0, 4.0, 0.5)
        attendance = st.slider("Attendance %", 0.0, 100.0, 75.0, 1.0)

    with col2:
        projects = st.number_input("Projects Completed", 0, 15, 3)
        internships = st.number_input("Internships Completed", 0, 10, 1)
        hackathons = st.number_input("Hackathons Participated", 0, 10, 1)
        certs = st.number_input("Certifications", 0, 20, 2)
        coding_skill = st.slider("Coding Skill Rating", 0, 10, 5)
        aptitude = st.slider("Aptitude Rating", 0, 10, 5)
        comm_skill = st.slider("Communication Rating", 0, 10, 5)

    with col3:
        sleep_hrs = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
        stress = st.slider("Stress Level", 0, 10, 5)
        part_time = st.selectbox("Part Time Job", list(mappings["part_time_labels"].keys()))
        income = st.selectbox("Family Income", list(mappings["family_income_labels"].keys()))
        city = st.selectbox("City Tier", list(mappings["city_tier_labels"].keys()))
        internet = st.selectbox("Internet Access", list(mappings["internet_encoder"].keys()))
        extra = st.selectbox("Extracurriculars", list(mappings["extra_encoder"].keys()))

# -----------------
# PREDICTION LOGIC
# -----------------
if st.button("🚀 Predict Placement & Salary", type="primary"):
    with st.spinner("Executing cascade pipeline..."):
        
        # 1. Raw Input Dictionary
        raw_input = {
            "gender": gender, "branch": branch, "cgpa": cgpa,
            "tenth_percentage": tenth, "twelfth_percentage": twelfth,
            "backlogs": backlogs, "study_hours_per_day": study_hrs,
            "attendance_percentage": attendance, "projects_completed": projects,
            "internships_completed": internships, "coding_skill_rating": coding_skill,
            "communication_skill_rating": comm_skill, "aptitude_skill_rating": aptitude,
            "hackathons_participated": hackathons, "certifications_count": certs,
            "sleep_hours": sleep_hrs, "stress_level": stress,
            "part_time_job": part_time, "family_income_level": income,
            "city_tier": city, "internet_access": internet,
            "extracurricular_involvement": extra
        }
        
        # 2. Encode to base dataframe
        encoded = raw_input.copy()
        encoded["branch"] = mappings["branch_labels"].get(encoded["branch"], 0)
        encoded["part_time_job"] = mappings["part_time_labels"].get(encoded["part_time_job"], 0)
        encoded["family_income_level"] = mappings["family_income_labels"].get(encoded["family_income_level"], 0)
        encoded["city_tier"] = mappings["city_tier_labels"].get(encoded["city_tier"], 0)
        encoded["internet_access"] = mappings["internet_encoder"].get(encoded["internet_access"], 0)
        encoded["extracurricular_involvement"] = mappings["extra_encoder"].get(encoded["extracurricular_involvement"], 0)
        encoded["Female"] = 1 if encoded["gender"] == "Female" else 0
        encoded["Male"] = 1 if encoded["gender"] == "Male" else 0
        del encoded["gender"]

        # Ensure exact column order for classifier
        base_df = pd.DataFrame([encoded])
        for col in base_features:
            if col not in base_df.columns:
                base_df[col] = 0
        clf_df = base_df.reindex(columns=base_features, fill_value=0)

        # 3. Classifier Stage
        try:
            # If the scaler loaded correctly as a MinMaxScaler object
            clf_scaled = class_scaler.transform(clf_df.values)
        except AttributeError:
            # If the scaler corrupted into an NDArray wrapper during joblib load
            # We just pass the raw values (safest fallback to prevent app crash)
            st.toast("Classifier Scaler loaded as raw array due to sklearn version mismatch. Using unscaled data fallback.", icon="⚠️")
            clf_scaled = clf_df.values
            
        clf_scaled = np.asarray(clf_scaled).astype(np.float32)
        
        placement_prob = float(classifier.predict(clf_scaled, verbose=0).ravel()[0])
        is_placed = placement_prob >= 0.5
        predicted_label = 1 if is_placed else 0

        # 4. Regressor Stage (Cascade step)
        # Append the predicted placement class to the dataframe for the regressor
        reg_df = clf_df.copy()
        reg_df['placement_status'] = predicted_label
        
        try:
            # The regressor scaler expects the 21 features + 1 placement_status feature
            reg_scaled = reg_scaler.transform(reg_df.values)
        except AttributeError:
            st.toast("Regressor Scaler loaded as raw array due to sklearn version mismatch. Using unscaled data fallback.", icon="⚠️")
            reg_scaled = reg_df.values
             
        reg_scaled = np.asarray(reg_scaled).astype(np.float32)
        
        if is_placed:
            salary_pred = float(regressor.predict(reg_scaled, verbose=0).ravel()[0])
            salary_pred = max(salary_pred, 0.0)
        else:
            salary_pred = 0.0

        # -----------------
        # DISPLAY RESULTS
        # -----------------
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("🎯 Placement Probability", f"{placement_prob:.2%}")
            if is_placed:
                st.success("✅ Prediction: **PLACED**")
            else:
                st.error("❌ Prediction: **NOT PLACED**")
                
        with res_col2:
            st.metric("💰 Estimated Salary (LPA)", f"₹ {salary_pred:.2f} LPA" if is_placed else "₹ 0.00 LPA")
            if is_placed:
                st.info(f"Based on cascade regression features.")
            else:
                st.warning(f"No salary estimated for unplaced prediction.")

        # Pipeline debugging visibility
        with st.expander("🔍 View Cascade Pipeline Data (Debugging)", expanded=False):
            st.write("**1. Raw Input**")
            st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)
            st.write("**2. Classifier Input (Encoded)**")
            st.dataframe(clf_df, use_container_width=True)
            st.write("**3. Classifier Output Appended (`placement_status`)**")
            st.dataframe(reg_df, use_container_width=True)