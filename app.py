import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


model = joblib.load('stroke_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')


st.title("Stroke Prediction System")
st.write("This application predicts the likelihood of stroke based on patient data.")



age = st.slider("Age", 0, 100, 50)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
avg_glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, 120.0)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "children", "Never_worked", "Govt_job"])
residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])


user_data = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender_Female': 1 if gender == "Female" else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'gender_Other': 1 if gender == "Other" else 0,
    'ever_married_Yes': 1 if ever_married == "Yes" else 0,
    'work_type_Private': 1 if work_type == "Private" else 0,
    'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
    'work_type_children': 1 if work_type == "children" else 0,
    'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
    'Residence_type_Urban': 1 if residence_type == "Urban" else 0,
    'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
    'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
    'smoking_status_smokes': 1 if smoking_status == "smokes" else 0
}


input_df = pd.DataFrame([user_data])
for feature in feature_columns:
    if feature not in input_df.columns:
        input_df[feature] = 0
input_df = input_df[feature_columns]


input_scaled = scaler.transform(input_df)


prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

button = st.markdown(
    """
    <style>
    .green-button {
        display: inline-block;
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        border-radius: 4px;
        cursor: pointer;
    }
    .green-button:hover {
        background-color: #45a049;
    }
    </style>
    <a href="#" class="green-button" onclick="window.dispatchEvent(new Event('predictStroke'))">Predict Stroke Risk</a>
    """,
    unsafe_allow_html=True
)


if st.button("Predict Stroke Risk"):
    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Stroke (Probability: {probability:.2%})")


