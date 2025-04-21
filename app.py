import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image
from io import BytesIO
import base64
import time

# Configure the page
st.set_page_config(
    page_title="Advanced Stroke Risk Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #555;
        font-style: italic;
    }
    .stProgress > div > div {
        background-image: linear-gradient(to right, #4CAF50, #FFC107, #F44336);
    }
    .feature-importance {
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# App header with logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 class='main-header'>🧠 Advanced Stroke Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("Powered by AI to help identify potential stroke risks based on health data")

# Initialize session state for tracking changes and history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'last_input' not in st.session_state:
    st.session_state.last_input = {}
if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Insights & Recommendations", "About Stroke"])

with tab1:
    try:
        # Load model components
        @st.cache_resource
        def load_model():
            model = joblib.load('stroke_prediction_model.pkl')
            scaler = joblib.load('scaler.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            return model, scaler, feature_columns
        
        model, scaler, feature_columns = load_model()
        
        # Function to get BMI category and color
        def get_bmi_category(bmi):
            if bmi < 18.5:
                return "Underweight", "#3498db"
            elif 18.5 <= bmi < 25:
                return "Normal weight", "#2ecc71"
            elif 25 <= bmi < 30:
                return "Overweight", "#f39c12"
            else:
                return "Obese", "#e74c3c"
        
        # Function to get glucose category and color
        def get_glucose_category(glucose):
            if glucose < 100:
                return "Normal", "#2ecc71"
            elif 100 <= glucose < 126:
                return "Prediabetes", "#f39c12"
            else:
                return "Diabetes range", "#e74c3c"

        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        # Create form to collect user inputs
        with st.form("patient_data_form"):
            st.markdown("<h3 class='sub-header'>Enter Patient Information</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Personal Details")
                age = st.slider("Age", 1, 100, 50, help="Patient's age in years")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                ever_married = st.selectbox("Ever Married", ["No", "Yes"])
                
            with col2:
                st.subheader("Lifestyle & Environment")
                work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Children", "Never_worked", "Govt_job"])
                residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
                smoking_status = st.selectbox("Smoking Status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])
                
            with col3:
                st.subheader("Health Metrics")
                hypertension = st.selectbox("Hypertension", ["No", "Yes"], 
                                        help="Has the patient been diagnosed with hypertension?")
                heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], 
                                            help="Has the patient been diagnosed with heart disease?")
                avg_glucose_level = st.slider("Average Glucose Level (mg/dL)", 50.0, 300.0, 120.0, 
                                            help="Patient's average blood glucose level")
                bmi = st.slider("BMI", 10.0, 50.0, 25.0, 
                                help="Body Mass Index (weight in kg / height in meters squared)")
            
            submit_button = st.form_submit_button(label="Predict Stroke Risk")
            
        st.markdown("</div>", unsafe_allow_html=True)
                
        # Process form submission
        if submit_button:
            with st.spinner("Analyzing patient data..."):
                # Convert categorical inputs to numeric values
                hypertension_value = 1 if hypertension == "Yes" else 0
                heart_disease_value = 1 if heart_disease == "Yes" else 0
                
                # Create user data dictionary with correct feature names
                user_data = {
                    'age': age,
                    'hypertension': hypertension_value,
                    'heart_disease': heart_disease_value,
                    'avg_glucose_level': avg_glucose_level,
                    'bmi': bmi,
                    'gender_Female': 1 if gender == "Female" else 0,
                    'gender_Male': 1 if gender == "Male" else 0,
                    'gender_Other': 1 if gender == "Other" else 0,
                    'ever_married_Yes': 1 if ever_married == "Yes" else 0,
                    'work_type_Private': 1 if work_type == "Private" else 0,
                    'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
                    'work_type_children': 1 if work_type == "Children" else 0,
                    'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
                    'work_type_Govt_job': 1 if work_type == "Govt_job" else 0,
                    'Residence_type_Urban': 1 if residence_type == "Urban" else 0,
                    'Residence_type_Rural': 1 if residence_type == "Rural" else 0,
                    'smoking_status_formerly smoked': 1 if smoking_status == "Formerly smoked" else 0,
                    'smoking_status_never smoked': 1 if smoking_status == "Never smoked" else 0,
                    'smoking_status_smokes': 1 if smoking_status == "Smokes" else 0,
                    'smoking_status_Unknown': 1 if smoking_status == "Unknown" else 0
                }
                
                # Store current data for later comparison
                current_input = {
                    'age': age,
                    'gender': gender,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'avg_glucose_level': avg_glucose_level,
                    'bmi': bmi,
                    'ever_married': ever_married,
                    'work_type': work_type,
                    'residence_type': residence_type,
                    'smoking_status': smoking_status
                }
                
                # Create DataFrame from user data
                input_df = pd.DataFrame([user_data])
                
                # Ensure all required features are present
                for feature in feature_columns:
                    if feature not in input_df.columns:
                        input_df[feature] = 0
                
                # Select only the features required by the model
                input_df = input_df[feature_columns]
                
                # Scale the features
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = float(model.predict_proba(input_scaled)[0][1])
                
                # Store result in history
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                history_entry = {
                    "timestamp": timestamp,
                    "inputs": current_input,
                    "probability": probability,
                    "prediction": prediction
                }
                st.session_state.history.append(history_entry)
                st.session_state.last_input = current_input
                
                # Calculate risk category
                if probability < 0.2:
                    risk_category = "Low"
                    risk_color = "#4CAF50"
                elif probability < 0.5:
                    risk_category = "Moderate"
                    risk_color = "#FFC107"
                else:
                    risk_category = "High"
                    risk_color = "#F44336"
                
                # Create SHAP explainer for this prediction
                try:
                    explainer = shap.Explainer(model, input_scaled)
                    shap_values = explainer(input_scaled)
                    has_shap = True
                except Exception as e:
                    has_shap = False
                    st.warning(f"Could not generate detailed explanations: {str(e)}")
                
                # Display results
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown("<h3 class='sub-header'>Risk Assessment Results</h3>", unsafe_allow_html=True)
                
                # Risk display
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display risk score with gauge visualization
                    st.markdown(f"<h2 style='text-align: center; color:{risk_color};'>{risk_category} Risk</h2>", unsafe_allow_html=True)
                    
                    # Create a circular gauge
                    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                    
                    # Set gauge properties
                    theta = np.linspace(0, 1.8*np.pi, 100)
                    ax.plot(theta, [1]*100, color='lightgray', lw=30, alpha=0.3)
                    
                    # Plot actual value
                    gauge_value = min(probability*1.8*np.pi, 1.8*np.pi)
                    cmap = cm.get_cmap('RdYlGn_r')
                    gauge_theta = np.linspace(0, gauge_value, 100)
                    colors = [cmap(val/(1.8*np.pi)) for val in gauge_theta]
                    
                    for i in range(len(gauge_theta)-1):
                        ax.plot(gauge_theta[i:i+2], [1, 1], color=colors[i], lw=30)
                    
                    # Clean up the plot
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    ax.spines['polar'].set_visible(False)
                    ax.grid(False)
                    
                    # Add percentage text in the middle
                    ax.text(0, 0, f"{probability*100:.1f}%", fontsize=22, ha='center', va='center')
                    
                    st.pyplot(fig)
                
                with col2:
                    # Health metrics analysis
                    st.subheader("Health Metrics Analysis")
                    
                    # Create three columns for metrics
                    mc1, mc2, mc3 = st.columns(3)
                    
                    with mc1:
                        # BMI status
                        bmi_category, bmi_color = get_bmi_category(bmi)
                        st.markdown(f"**BMI Status**: <span style='color:{bmi_color};'>{bmi_category} ({bmi:.1f})</span>", 
                                    unsafe_allow_html=True)
                    
                    with mc2:
                        # Glucose status
                        glucose_category, glucose_color = get_glucose_category(avg_glucose_level)
                        st.markdown(f"**Glucose Status**: <span style='color:{glucose_color};'>{glucose_category} ({avg_glucose_level:.1f} mg/dL)</span>", 
                                    unsafe_allow_html=True)
                    
                    with mc3:
                        # Age risk
                        if age > 65:
                            age_risk = "Higher risk age group"
                            age_color = "#e74c3c"
                        elif age > 45:
                            age_risk = "Moderate risk age group" 
                            age_color = "#f39c12"
                        else:
                            age_risk = "Lower risk age group"
                            age_color = "#2ecc71"
                        
                        st.markdown(f"**Age Factor**: <span style='color:{age_color};'>{age_risk}</span>", 
                                    unsafe_allow_html=True)
                    
                    # Risk factors summary
                    st.subheader("Key Risk Factors")
                    risk_factors = []
                    
                    if hypertension == "Yes":
                        risk_factors.append("Hypertension")
                    if heart_disease == "Yes":
                        risk_factors.append("Heart Disease")
                    if avg_glucose_level > 125:
                        risk_factors.append("High Blood Glucose")
                    if bmi > 30:
                        risk_factors.append("Obesity")
                    if smoking_status == "Smokes":
                        risk_factors.append("Active Smoking")
                    if age > 65:
                        risk_factors.append("Advanced Age")
                    
                    if risk_factors:
                        st.markdown(", ".join(risk_factors))
                    else:
                        st.markdown("No major risk factors identified")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Feature importance display
                st.markdown("<div class='feature-importance'>", unsafe_allow_html=True)
                st.subheader("Understanding Your Risk Factors")
                
                if has_shap:
                    # Create waterfall plot with proper handling for multi-output models
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    try:
                        # First determine if we're dealing with a multi-output model
                        if isinstance(shap_values, list) or (hasattr(shap_values, 'shape') and len(shap_values.shape) > 2):
                            # Multi-output case: select the first output's explanation
                            if isinstance(shap_values, list):
                                # For list-type shap values
                                shap_explanation = shap_values[0][0]
                            else:
                                # For numpy array-type shap values
                                shap_explanation = shap_values[0, 0]
                        else:
                            # Single output case: just take the first explanation
                            shap_explanation = shap_values[0]
                        
                        # Generate the waterfall plot
                        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Create force plot with the same logic
                        plt.figure(figsize=(12, 3))
                        
                        # Apply the same multi-output detection logic
                        if isinstance(shap_values, list) or (hasattr(shap_values, 'shape') and len(shap_values.shape) > 2):
                            if isinstance(shap_values, list):
                                shap_explanation = shap_values[0][0]
                            else:
                                shap_explanation = shap_values[0, 0]
                        else:
                            shap_explanation = shap_values[0]
                            
                        shap_html = shap.plots.force(shap_explanation, matplotlib=False)
                        st.components.v1.html(shap_html, height=150)
                        
                    except Exception as e:
                        st.warning(f"Error displaying SHAP visualizations: {str(e)}")
                        st.info("Falling back to simplified feature importance display.")
                        
                        # Fallback to feature importance or coefficient display
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = pd.DataFrame({
                                'Feature': feature_columns,
                                'Importance': model.feature_importances_
                            })
                            feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(10)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['#1E88E5'] * len(feature_importance)  # Using a consistent color
                            ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
                            ax.set_title('Top 10 Risk Factors')
                            ax.set_xlabel('Feature Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                        elif hasattr(model, 'coef_'):
                            # For linear models
                            feature_importance = pd.DataFrame({
                                'Feature': feature_columns,
                                'Importance': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                            })
                            feature_importance['Absolute'] = abs(feature_importance['Importance'])
                            feature_importance = feature_importance.sort_values(by='Absolute', ascending=False).head(10)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['#FF4136' if x > 0 else '#0074D9' for x in feature_importance['Importance']]
                            ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
                            ax.set_title('Top 10 Risk Factors')
                            ax.set_xlabel('Impact on Risk (Positive = Higher Risk)')
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    # Alternative: Show coefficient importance if model has coef_
                    if hasattr(model, 'coef_'):
                        # Get feature importance
                        feature_importance = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': model.coef_[0]
                        })
                        feature_importance['Absolute'] = abs(feature_importance['Importance'])
                        feature_importance = feature_importance.sort_values(by='Absolute', ascending=False).head(10)
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#FF4136' if x > 0 else '#0074D9' for x in feature_importance['Importance']]
                        ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
                        ax.set_title('Top 10 Risk Factors')
                        ax.set_xlabel('Impact on Risk (Positive = Higher Risk)')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Explain the SHAP values
                st.markdown("""
                **Interpreting the charts above:**
                - Red factors increase stroke risk
                - Blue factors decrease stroke risk
                - The size of each bar represents how strongly that factor affects your prediction
                """)
                
                # Disclaimer
                st.markdown("<p class='disclaimer'>Note: This tool provides an estimate based on machine learning models and should not replace professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Offer comparison option
                if len(st.session_state.history) > 1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.subheader("Risk Tracking")
                    if st.button("Show Assessment History"):
                        st.session_state.show_history = not st.session_state.show_history
                        
                    if st.session_state.show_history:
                        # Display history table
                        history_df = pd.DataFrame([{
                            "Date": entry["timestamp"],
                            "Risk Score": f"{entry['probability']*100:.1f}%",
                            "Age": entry["inputs"]["age"],
                            "BMI": entry["inputs"]["bmi"],
                            "Glucose": entry["inputs"]["avg_glucose_level"],
                            "Hypertension": entry["inputs"]["hypertension"],
                            "Heart Disease": entry["inputs"]["heart_disease"],
                            "Smoking": entry["inputs"]["smoking_status"]
                        } for entry in st.session_state.history])
                        
                        st.dataframe(history_df)
                        
                        # Plot risk trend if more than 2 entries
                        if len(st.session_state.history) > 2:
                            trend_data = pd.DataFrame([{
                                "Assessment": i+1,
                                "Risk Score": entry["probability"]*100
                            } for i, entry in enumerate(st.session_state.history)])
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.lineplot(data=trend_data, x="Assessment", y="Risk Score", marker='o', ax=ax)
                            ax.set_title("Risk Score Trend")
                            ax.set_ylabel("Risk (%)")
                            plt.tight_layout()
                            st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)
    
    except FileNotFoundError:
        st.error("⚠️ Required model files not found. Please ensure 'stroke_prediction_model.pkl', 'scaler.pkl', and 'feature_columns.pkl' exist in the application directory.")
        st.info("If you're developing this application, place the model files in the same directory as this script.")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
        st.info("Please try again or contact support if the problem persists.")

with tab2:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.subheader("Personalized Health Insights")
    
    if len(st.session_state.history) > 0:
        latest = st.session_state.history[-1]
        
        # Extract key health metrics
        age = latest["inputs"]["age"]
        bmi = latest["inputs"]["bmi"]
        glucose = latest["inputs"]["avg_glucose_level"]
        hypertension = latest["inputs"]["hypertension"]
        heart_disease = latest["inputs"]["heart_disease"]
        smoking = latest["inputs"]["smoking_status"]
        risk_score = latest["probability"]
        
        # Personalized recommendations
        st.markdown("### Key Recommendations")
        
        recommendations = []
        
        # BMI recommendations
        if bmi < 18.5:
            recommendations.append("Consider a nutrition plan to achieve a healthy weight (BMI between 18.5-24.9)")
        elif bmi >= 30:
            recommendations.append("Work with healthcare providers on a weight management plan to reduce BMI below 30")
        elif bmi >= 25:
            recommendations.append("Maintain a balanced diet and regular exercise to work towards a BMI below 25")
        
        # Glucose recommendations
        if glucose >= 126:
            recommendations.append("Consult with a healthcare provider about your elevated blood glucose levels")
            recommendations.append("Consider dietary changes to help manage blood glucose levels")
        elif glucose >= 100:
            recommendations.append("Monitor your blood glucose levels regularly as they are in the prediabetic range")
        
        # Hypertension recommendations
        if hypertension == "Yes":
            recommendations.append("Continue taking prescribed medications for hypertension")
            recommendations.append("Monitor blood pressure regularly")
            recommendations.append("Reduce sodium intake to help manage blood pressure")
        
        # Heart disease recommendations
        if heart_disease == "Yes":
            recommendations.append("Follow your cardiologist's treatment plan for heart disease")
            recommendations.append("Consider cardiac rehabilitation programs if appropriate")
        
        # Smoking recommendations
        if smoking == "Smokes":
            recommendations.append("Consider a smoking cessation program - quitting smoking significantly reduces stroke risk")
        elif smoking == "Formerly smoked":
            recommendations.append("Continue to abstain from smoking to further reduce your stroke risk")
        
        # General recommendations based on risk score
        if risk_score > 0.5:
            recommendations.append("Schedule a comprehensive stroke risk assessment with your healthcare provider")
            recommendations.append("Learn to recognize signs of stroke: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services")
        elif risk_score > 0.2:
            recommendations.append("Discuss your stroke risk factors with your healthcare provider at your next visit")
        
        # Add general recommendations for everyone
        general_recs = [
            "Maintain regular physical activity (at least 150 minutes of moderate exercise weekly)",
            "Follow a Mediterranean or DASH diet rich in fruits, vegetables, and whole grains",
            "Limit alcohol consumption",
            "Manage stress through mindfulness, meditation, or other techniques",
            "Get regular health check-ups including blood pressure and cholesterol screening"
        ]
        
        # Display recommendations
        for i, rec in enumerate(recommendations[:5], 1):
            st.markdown(f"**{i}. {rec}**")
        
        st.markdown("### General Health Recommendations")
        for i, rec in enumerate(general_recs, 1):
            st.markdown(f"{i}. {rec}")
        
        # Lifestyle modifications impact
        st.subheader("Impact of Lifestyle Modifications")
        st.markdown("Explore how different changes might affect your stroke risk:")
        
        mod_col1, mod_col2 = st.columns(2)
        
        with mod_col1:
            selected_modification = st.selectbox("Select a potential lifestyle change:", [
                "Reduce BMI by 5 points",
                "Lower glucose levels by 20 mg/dL",
                "Quit smoking",
                "Control hypertension",
                "Multiple lifestyle changes"
            ])
        
        with mod_col2:
            st.write("Current risk probability:")
            current_percentage = risk_score * 100
            st.progress(min(risk_score, 1.0))
            st.write(f"{current_percentage:.1f}%")
        
        # Calculate modified risk
        if st.button("Calculate Modified Risk"):
            # Copy the latest data
            modified_data = latest["inputs"].copy()
            
            # Apply selected modification
            if selected_modification == "Reduce BMI by 5 points":
                modified_data["bmi"] = max(18.5, modified_data["bmi"] - 5)
                impact_description = "Reducing BMI by 5 points"
            
            elif selected_modification == "Lower glucose levels by 20 mg/dL":
                modified_data["avg_glucose_level"] = max(70, modified_data["avg_glucose_level"] - 20)
                impact_description = "Lowering glucose levels by 20 mg/dL"
            
            elif selected_modification == "Quit smoking":
                if modified_data["smoking_status"] == "Smokes":
                    modified_data["smoking_status"] = "Formerly smoked"
                impact_description = "Quitting smoking"
            
            elif selected_modification == "Control hypertension":
                if modified_data["hypertension"] == "Yes":
                    modified_data["hypertension"] = "No"
                impact_description = "Controlling hypertension"
            
            elif selected_modification == "Multiple lifestyle changes":
                if modified_data["bmi"] > 25:
                    modified_data["bmi"] = 25
                if modified_data["avg_glucose_level"] > 100:
                    modified_data["avg_glucose_level"] = 100
                if modified_data["smoking_status"] == "Smokes":
                    modified_data["smoking_status"] = "Formerly smoked"
                if modified_data["hypertension"] == "Yes":
                    modified_data["hypertension"] = "No"
                impact_description = "Implementing multiple lifestyle changes"
            
            # Convert to model input format
            hypertension_value = 1 if modified_data["hypertension"] == "Yes" else 0
            heart_disease_value = 1 if modified_data["heart_disease"] == "Yes" else 0
            
            user_data = {
                'age': modified_data["age"],
                'hypertension': hypertension_value,
                'heart_disease': heart_disease_value,
                'avg_glucose_level': modified_data["avg_glucose_level"],
                'bmi': modified_data["bmi"],
                'gender_Female': 1 if modified_data["gender"] == "Female" else 0,
                'gender_Male': 1 if modified_data["gender"] == "Male" else 0,
                'gender_Other': 1 if modified_data["gender"] == "Other" else 0,
                'ever_married_Yes': 1 if modified_data["ever_married"] == "Yes" else 0,
                'work_type_Private': 1 if modified_data["work_type"] == "Private" else 0,
                'work_type_Self-employed': 1 if modified_data["work_type"] == "Self-employed" else 0,
                'work_type_children': 1 if modified_data["work_type"] == "Children" else 0,
                'work_type_Never_worked': 1 if modified_data["work_type"] == "Never_worked" else 0,
                'work_type_Govt_job': 1 if modified_data["work_type"] == "Govt_job" else 0,
                'Residence_type_Urban': 1 if modified_data["residence_type"] == "Urban" else 0,
                'Residence_type_Rural': 1 if modified_data["residence_type"] == "Rural" else 0,
                'smoking_status_formerly smoked': 1 if modified_data["smoking_status"] == "Formerly smoked" else 0,
                'smoking_status_never smoked': 1 if modified_data["smoking_status"] == "Never smoked" else 0,
                'smoking_status_smokes': 1 if modified_data["smoking_status"] == "Smokes" else 0,
                'smoking_status_Unknown': 1 if modified_data["smoking_status"] == "Unknown" else 0
            }
            
            # Create DataFrame from user data
            # Continuing from where the code was cut off...

            # Create DataFrame from user data
            input_df = pd.DataFrame([user_data])
            
            # Ensure all required features are present
            for feature in feature_columns:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Select only the features required by the model
            input_df = input_df[feature_columns]
            
            # Scale the features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            modified_prediction = model.predict(input_scaled)[0]
            modified_probability = float(model.predict_proba(input_scaled)[0][1])
            
            # Display results
            modified_percentage = modified_probability * 100
            original_percentage = risk_score * 100
            difference = original_percentage - modified_percentage
            
            st.subheader("Risk Comparison")
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Current Risk:**")
                st.progress(min(risk_score, 1.0))
                st.write(f"{original_percentage:.1f}%")
            
            with comp_col2:
                st.markdown("**Modified Risk:**")
                st.progress(min(modified_probability, 1.0))
                st.write(f"{modified_percentage:.1f}%")
            
            # Risk reduction statement
            if difference > 0:
                st.success(f"**Potential Impact:** {impact_description} could reduce your stroke risk by approximately {difference:.1f} percentage points.")
            elif difference < 0:
                st.error(f"**Note:** {impact_description} may not reduce risk in your specific case based on the current model.")
            else:
                st.info(f"**Note:** {impact_description} shows minimal impact on your stroke risk based on the current model.")

    else:
        st.info("Please complete a risk assessment in the 'Risk Assessment' tab to view personalized recommendations.")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Resources section
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.subheader("Additional Resources")
    
    resources_col1, resources_col2 = st.columns(2)
    
    with resources_col1:
        st.markdown("### Professional Organizations")
        st.markdown("""
        - [American Stroke Association](https://www.stroke.org/)
        - [World Stroke Organization](https://www.world-stroke.org/)
        - [National Stroke Association](https://www.stroke.org/)
        - [CDC Stroke Information](https://www.cdc.gov/stroke/)
        """)
    
    with resources_col2:
        st.markdown("### Healthy Lifestyle Resources")
        st.markdown("""
        - [DASH Diet for Hypertension](https://www.nhlbi.nih.gov/health-topics/dash-eating-plan)
        - [Physical Activity Guidelines](https://health.gov/our-work/nutrition-physical-activity/physical-activity-guidelines)
        - [Smoking Cessation Resources](https://smokefree.gov/)
        - [Blood Pressure Management Guide](https://www.heart.org/en/health-topics/high-blood-pressure)
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>About Stroke</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ## What is a Stroke?
    
    A stroke occurs when blood flow to part of the brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die within minutes, making stroke a medical emergency.
    
    There are two main types of stroke:
    - **Ischemic stroke**: Caused by a blockage in an artery that supplies blood to the brain
    - **Hemorrhagic stroke**: Caused by a blood vessel leaking or rupturing in the brain
    
    A temporary disruption in blood flow, called a transient ischemic attack (TIA) or "mini-stroke," can also occur.
    """)
    
    st.markdown("## Warning Signs of Stroke - Think FAST")
    
    fast_col1, fast_col2 = st.columns(2)
    
    with fast_col1:
        st.markdown("""
        ### FAST Warning Signs
        - **F** - Face Drooping: Does one side of the face droop?
        - **A** - Arm Weakness: Is one arm weak or numb?
        - **S** - Speech Difficulty: Is speech slurred or strange?
        - **T** - Time to Call Emergency Services: If you observe any of these signs, call immediately!
        """)
    
    with fast_col2:
        st.markdown("""
        ### Additional Warning Signs
        - Sudden numbness or weakness in the face, arm, or leg, especially on one side of the body
        - Sudden confusion, trouble speaking, or difficulty understanding speech
        - Sudden trouble seeing in one or both eyes
        - Sudden trouble walking, dizziness, loss of balance, or lack of coordination
        - Sudden severe headache with no known cause
        """)
    
    st.markdown("## Risk Factors for Stroke")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("""
        ### Manageable Risk Factors
        - High blood pressure
        - Smoking
        - Diabetes
        - High cholesterol
        - Physical inactivity
        - Obesity
        - Unhealthy diet
        - Excessive alcohol consumption
        - Atrial fibrillation
        - Sleep apnea
        """)
    
    with risk_col2:
        st.markdown("""
        ### Non-Manageable Risk Factors
        - Age (risk increases with age)
        - Gender (stroke risk differs between genders)
        - Family history of stroke
        - Race and ethnicity (higher risks for certain groups)
        - Previous stroke or TIA
        - Certain genetic disorders
        """)
    
    st.markdown("## Prevention Strategies")
    st.markdown("""
    ### Lifestyle Changes
    - **Diet**: Follow a Mediterranean or DASH diet rich in fruits, vegetables, whole grains, and lean proteins
    - **Exercise**: Engage in regular physical activity (at least 150 minutes of moderate exercise per week)
    - **Smoking**: Quit smoking and avoid secondhand smoke
    - **Alcohol**: Limit alcohol consumption
    
    ### Medical Management
    - **Blood Pressure**: Control high blood pressure through medication and lifestyle changes
    - **Cholesterol**: Maintain healthy cholesterol levels
    - **Diabetes**: Manage blood sugar levels effectively
    - **Medications**: Take prescribed medications, such as blood thinners for atrial fibrillation
    
    ### Regular Screening
    - Get regular health check-ups to monitor blood pressure, cholesterol, and blood sugar
    - Discuss stroke risk with healthcare providers
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
<p>© 2025 Advanced Stroke Risk Predictor | Developed for Healthcare Professionals</p>
<p class='disclaimer'>This tool is intended for educational and screening purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
</div>
""", unsafe_allow_html=True)
