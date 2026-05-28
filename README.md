# College Placements Predictor

A machine learning project for predicting student placement outcomes and estimating salary packages using academic, skill-based, and background-related features.

## Overview

This repository focuses on the **machine learning pipeline** behind student placement prediction.  
The primary goal of the project is to preprocess the dataset, train predictive models, evaluate their performance, and package the final inference pipeline in a simple Streamlit interface for demonstration.

The app is included only as a **wrapper for display and interactive testing** of the trained models.

## Problem Statement

The objective is to predict:

1. Whether a student is likely to get placed.
2. The expected salary package in LPA for predicted placements.

This is handled as a two-stage ML task:

- **Binary Classification** for placement prediction
- **Regression** for salary estimation

## Dataset

The project uses student-related placement data with features such as:

- Branch
- CGPA
- 10th percentage
- 12th percentage
- Backlogs
- Study hours per day
- Attendance percentage
- Projects completed
- Internships completed
- Coding skill rating
- Communication skill rating
- Aptitude skill rating
- Hackathons participated
- Certifications count
- Sleep hours
- Stress level
- Part-time job
- Family income level
- City tier
- Internet access
- Extracurricular involvement

Target information is stored separately for placement outcome modeling.

## ML Pipeline

### 1. Data Preprocessing

The preprocessing stage includes:

- Feature selection
- Categorical encoding
- Numerical scaling
- Target preparation
- Exporting feature schema and encoder mappings for inference consistency

Saved preprocessing artifacts:

- `encoder_mappings.json`
- `feature_columns.json`
- `classifier_scaler.pkl`
- `regressor_scaler.pkl`

### 2. Classification Model

The classifier predicts whether a student will be placed or not.

Output:

- `0` → Not Placed
- `1` → Placed

Saved model:

- `placement_classifier.keras`

### 3. Regression Model

The regression model estimates salary in LPA using the encoded feature set along with predicted `placement_status`.

Saved model:

- `lpa_regressor.keras`

## Evaluation and Analysis

The repository includes multiple plots and evaluation visuals for understanding the dataset and model performance.

### Data Visualizations

#### Branch Distribution

![Branch Distribution](images/branch_distribution.png)

#### Initial Placement Distribution

![Initial Placement Distribution](images/initial_placement_distribution.png)

#### Placement Distribution After Processing

![Placement Distribution After Processing](images/latter_placement_distribution.png)

### Model Evaluation

#### Final Confusion Matrix

![Final Confusion Matrix](images/final_conf_matrix.png)

#### Regressor Loss Plot

![Regressor Loss Plot](images/regressor_loss_plot.png)

#### Regressor True vs Predicted

![Regressor True vs Predicted](images/regressor_true_pred.png)

## Streamlit App

A lightweight Streamlit app is included to wrap the trained ML pipeline and allow interactive testing of predictions.

The app:

- Takes raw student input
- Encodes it using saved mappings
- Aligns features using exported schema
- Runs the classifier
- Appends predicted placement status
- Runs the regressor
- Displays placement probability and salary estimate

This app is meant only for **presentation and demonstration of the trained ML pipeline**, not as the main focus of the repository.

### App Interface

![Streamlit Interface](images/interface.png)

## Repository Structure

```text
college_placements/
├── app_folder/
│   ├── app.py
│   ├── placement_classifier.keras
│   ├── lpa_regressor.keras
│   ├── classifier_scaler.pkl
│   ├── regressor_scaler.pkl
│   ├── encoder_mappings.json
│   └── feature_columns.json
├── images/
│   ├── interface.png
│   ├── branch_distribution.png
│   ├── initial_placement_distribution.png
│   ├── latter_placement_distribution.png
│   ├── final_conf_matrix.png
│   ├── regressor_loss_plot.png
│   └── regressor_true_pred.png
├── indian_college_placements.ipynb
├── indian_engineering_student_placement.csv
├── placement_targets.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/abhishekspawar0803-svg/college_placements.git
cd college_placements
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit wrapper

```bash
streamlit run app_folder/app.py
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Joblib

## Key Takeaway

This repository is primarily an **ML project** centered on preprocessing, model training, evaluation, and inference packaging.  
The Streamlit app is included only as a simple interface to demonstrate the trained models in action.

## Author

**Abhishek Pawar**
