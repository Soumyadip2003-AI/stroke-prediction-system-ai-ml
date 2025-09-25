#!/usr/bin/env python3
"""
Add 10K More Data Script
Adds 10,000 more synthetic data points to healthcare-dataset-stroke-data.csv
"""

import pandas as pd
import numpy as np
import random

def add_10k_data():
    """Add 10,000 more synthetic data points to the dataset"""
    print("📊 Loading current dataset...")
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    print(f"Current dataset shape: {df.shape}")
    print(f"Current stroke distribution: {df['stroke'].value_counts()}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create 10,000 more synthetic data points
    print("🔧 Generating 10,000 additional synthetic data points...")
    
    n_synthetic = 10000  # Number of additional synthetic samples
    
    synthetic_data = []
    
    for i in range(n_synthetic):
        # Generate synthetic sample
        sample = {}
        
        # Gender (0: Male, 1: Female, 2: Other)
        sample['gender'] = np.random.choice(['Male', 'Female', 'Other'], p=[0.5, 0.45, 0.05])
        
        # Age (realistic distribution with more variety)
        age_group = np.random.choice(['young', 'middle', 'elderly', 'very_elderly'], 
                                   p=[0.3, 0.4, 0.2, 0.1])
        
        if age_group == 'young':
            sample['age'] = np.random.normal(28, 6)
        elif age_group == 'middle':
            sample['age'] = np.random.normal(48, 12)
        elif age_group == 'elderly':
            sample['age'] = np.random.normal(72, 8)
        else:  # very_elderly
            sample['age'] = np.random.normal(82, 4)
        
        sample['age'] = max(0, min(100, sample['age']))  # Clamp to realistic range
        
        # Hypertension (age-dependent with more complexity)
        base_hypertension_prob = 0.03 + (sample['age'] - 20) * 0.01
        if sample['age'] > 65:
            base_hypertension_prob += 0.25
        elif sample['age'] > 50:
            base_hypertension_prob += 0.15
        sample['hypertension'] = 1 if np.random.random() < base_hypertension_prob else 0
        
        # Heart disease (age, hypertension, and gender dependent)
        heart_disease_base = 0.01 + (sample['age'] - 20) * 0.004
        heart_disease_base += sample['hypertension'] * 0.2
        if sample['gender'] == 'Male':
            heart_disease_base += 0.08
        elif sample['gender'] == 'Female':
            heart_disease_base += 0.03
        sample['heart_disease'] = 1 if np.random.random() < heart_disease_base else 0
        
        # Ever married (age-dependent with cultural factors)
        if sample['age'] < 18:
            sample['ever_married'] = 'No'
        elif sample['age'] > 35:
            sample['ever_married'] = np.random.choice(['Yes', 'No'], p=[0.9, 0.1])
        elif sample['age'] > 28:
            sample['ever_married'] = np.random.choice(['Yes', 'No'], p=[0.7, 0.3])
        elif sample['age'] > 22:
            sample['ever_married'] = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
        else:
            sample['ever_married'] = np.random.choice(['Yes', 'No'], p=[0.2, 0.8])
        
        # Work type (age and gender dependent)
        if sample['age'] < 18:
            sample['work_type'] = 'children'
        elif sample['age'] > 75:
            sample['work_type'] = np.random.choice(['Private', 'Self-employed', 'Govt_job'], 
                                                 p=[0.2, 0.5, 0.3])
        elif sample['age'] > 65:
            sample['work_type'] = np.random.choice(['Private', 'Self-employed', 'Govt_job', 'Never_worked'], 
                                                 p=[0.3, 0.4, 0.2, 0.1])
        elif sample['age'] > 50:
            sample['work_type'] = np.random.choice(['Private', 'Self-employed', 'Govt_job', 'Never_worked'], 
                                                 p=[0.45, 0.3, 0.15, 0.1])
        else:
            sample['work_type'] = np.random.choice(['Private', 'Self-employed', 'Govt_job', 'Never_worked'], 
                                                 p=[0.5, 0.25, 0.15, 0.1])
        
        # Residence type (with correlation to work type and age)
        if sample['work_type'] == 'Govt_job':
            sample['Residence_type'] = np.random.choice(['Urban', 'Rural'], p=[0.85, 0.15])
        elif sample['work_type'] == 'Self-employed':
            sample['Residence_type'] = np.random.choice(['Urban', 'Rural'], p=[0.35, 0.65])
        elif sample['age'] > 60:
            sample['Residence_type'] = np.random.choice(['Urban', 'Rural'], p=[0.4, 0.6])
        else:
            sample['Residence_type'] = np.random.choice(['Urban', 'Rural'], p=[0.65, 0.35])
        
        # Average glucose level (age, BMI, and lifestyle dependent)
        base_glucose = 70 + (sample['age'] - 20) * 1.2
        if sample['age'] > 60:
            base_glucose += 15
        elif sample['age'] > 40:
            base_glucose += 8
        sample['avg_glucose_level'] = max(50, np.random.normal(base_glucose, 30))
        
        # BMI (age, gender, and lifestyle dependent)
        if sample['age'] < 25:
            base_bmi = 21 + np.random.normal(0, 3)
        elif sample['age'] < 35:
            base_bmi = 23 + np.random.normal(0, 4)
        elif sample['age'] < 50:
            base_bmi = 25 + np.random.normal(0, 5)
        elif sample['age'] < 65:
            base_bmi = 27 + np.random.normal(0, 5)
        else:
            base_bmi = 28 + np.random.normal(0, 4)
        
        # Gender adjustment for BMI
        if sample['gender'] == 'Female':
            base_bmi -= 1.5
        elif sample['gender'] == 'Other':
            base_bmi += 0.5
        
        sample['bmi'] = max(15, min(50, base_bmi))
        
        # Smoking status (age, gender, and work type dependent)
        if sample['age'] < 18:
            sample['smoking_status'] = 'never smoked'
        else:
            if sample['gender'] == 'Male':
                if sample['work_type'] == 'Self-employed':
                    sample['smoking_status'] = np.random.choice(['never smoked', 'formerly smoked', 'smokes'], 
                                                              p=[0.25, 0.3, 0.45])
                elif sample['work_type'] == 'Private':
                    sample['smoking_status'] = np.random.choice(['never smoked', 'formerly smoked', 'smokes'], 
                                                              p=[0.35, 0.35, 0.3])
                else:
                    sample['smoking_status'] = np.random.choice(['never smoked', 'formerly smoked', 'smokes'], 
                                                              p=[0.4, 0.3, 0.3])
            elif sample['gender'] == 'Female':
                if sample['work_type'] == 'Self-employed':
                    sample['smoking_status'] = np.random.choice(['never smoked', 'formerly smoked', 'smokes'], 
                                                              p=[0.45, 0.35, 0.2])
                elif sample['work_type'] == 'Private':
                    sample['smoking_status'] = np.random.choice(['never smoked', 'formerly smoked', 'smokes'], 
                                                              p=[0.6, 0.25, 0.15])
                else:
                    sample['smoking_status'] = np.random.choice(['never smoked', 'formerly smoked', 'smokes'], 
                                                              p=[0.7, 0.2, 0.1])
            else:  # Other gender
                sample['smoking_status'] = np.random.choice(['never smoked', 'formerly smoked', 'smokes'], 
                                                          p=[0.5, 0.3, 0.2])
        
        # Determine stroke based on comprehensive risk factors
        stroke_risk = 0
        
        # Age factor (non-linear with more granularity)
        if sample['age'] > 85:
            stroke_risk += 0.5
        elif sample['age'] > 75:
            stroke_risk += 0.4
        elif sample['age'] > 65:
            stroke_risk += 0.3
        elif sample['age'] > 55:
            stroke_risk += 0.2
        elif sample['age'] > 45:
            stroke_risk += 0.1
        elif sample['age'] > 35:
            stroke_risk += 0.05
        else:
            stroke_risk += 0.01
        
        # BMI factor (non-linear)
        if sample['bmi'] > 40:
            stroke_risk += 0.35
        elif sample['bmi'] > 35:
            stroke_risk += 0.25
        elif sample['bmi'] > 30:
            stroke_risk += 0.15
        elif sample['bmi'] > 25:
            stroke_risk += 0.08
        elif sample['bmi'] < 18.5:
            stroke_risk += 0.05
        
        # Glucose factor (non-linear)
        if sample['avg_glucose_level'] > 250:
            stroke_risk += 0.5
        elif sample['avg_glucose_level'] > 200:
            stroke_risk += 0.35
        elif sample['avg_glucose_level'] > 126:
            stroke_risk += 0.2
        elif sample['avg_glucose_level'] > 100:
            stroke_risk += 0.08
        
        # Medical conditions (interactive effects)
        stroke_risk += sample['hypertension'] * 0.2
        stroke_risk += sample['heart_disease'] * 0.3
        
        # Combined medical conditions (synergistic effect)
        if sample['hypertension'] == 1 and sample['heart_disease'] == 1:
            stroke_risk += 0.25
        elif sample['hypertension'] == 1 or sample['heart_disease'] == 1:
            stroke_risk += 0.1
        
        # Smoking factor (interactive with age and gender)
        if sample['smoking_status'] == 'smokes':
            if sample['age'] > 60:
                stroke_risk += 0.35
            elif sample['age'] > 40:
                stroke_risk += 0.25
            else:
                stroke_risk += 0.15
        elif sample['smoking_status'] == 'formerly smoked':
            if sample['age'] > 60:
                stroke_risk += 0.2
            elif sample['age'] > 40:
                stroke_risk += 0.12
            else:
                stroke_risk += 0.05
        
        # Work type factor
        if sample['work_type'] == 'Never_worked':
            stroke_risk += 0.12
        elif sample['work_type'] == 'Self-employed':
            stroke_risk += 0.08
        elif sample['work_type'] == 'children':
            stroke_risk += 0.02
        
        # Gender factor
        if sample['gender'] == 'Male':
            stroke_risk += 0.08
        elif sample['gender'] == 'Other':
            stroke_risk += 0.03
        
        # Residence factor
        if sample['Residence_type'] == 'Rural':
            stroke_risk += 0.03
        
        # Marriage factor
        if sample['ever_married'] == 'No' and sample['age'] > 30:
            stroke_risk += 0.05
        
        # Add some randomness
        stroke_risk += np.random.normal(0, 0.1)
        
        # Determine stroke outcome with threshold
        sample['stroke'] = 1 if stroke_risk > 0.35 else 0
        
        # Add ID
        sample['id'] = len(df) + i + 1
        
        synthetic_data.append(sample)
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_data)
    
    print(f"Additional synthetic dataset shape: {synthetic_df.shape}")
    print(f"Additional synthetic stroke distribution: {synthetic_df['stroke'].value_counts()}")
    
    # Combine with existing data
    enhanced_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    print(f"Final enhanced dataset shape: {enhanced_df.shape}")
    print(f"Final stroke distribution: {enhanced_df['stroke'].value_counts()}")
    
    # Save back to original file
    enhanced_df.to_csv('healthcare-dataset-stroke-data.csv', index=False)
    print("✅ Dataset enhanced with 10,000 additional data points!")
    
    return enhanced_df

def main():
    """Main function"""
    print("🚀 Adding 10,000 More Data Points")
    print("=" * 50)
    
    # Add more data
    enhanced_df = add_10k_data()
    
    print("\n✅ Dataset enhancement complete!")
    print(f"📊 Final dataset shape: {enhanced_df.shape}")
    print(f"📈 Final stroke distribution: {enhanced_df['stroke'].value_counts()}")
    print(f"📊 Stroke percentage: {enhanced_df['stroke'].mean()*100:.2f}%")
    
    return enhanced_df

if __name__ == "__main__":
    enhanced_df = main()

