import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Handle missing values
data['bmi'].fillna(data['bmi'].mean(), inplace=True)
data.drop('id', axis=1, inplace=True)

# One-hot encoding
data = pd.get_dummies(data)

# Save feature columns
feature_columns = list(data.columns)
feature_columns.remove("stroke")

# Define features and target
X = data[feature_columns]
y = data['stroke']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model, scaler, and feature columns
joblib.dump(model, 'stroke_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print("Model, scaler, and feature columns saved successfully.")
