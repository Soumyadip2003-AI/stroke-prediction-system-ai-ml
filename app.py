import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


model = joblib.load('stroke_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.write.title("Patient Information")
age = st.sidebar.slider("Age", 0, 100, 50)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 120.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "children", "Never_worked", "Govt_job"])
residence_type = st.sidebar.selectbox("Residence Type", ["Rural", "Urban"])
smoking_status = st.sidebar.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])


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

st.title("Stroke Prediction System")
st.write("This application predicts the likelihood of stroke based on patient data.")
st.write("Made by soumyadip sarkar")

if st.button("Predict Stroke Risk"):
    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Stroke (Probability: {probability:.2%})")


