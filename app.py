import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Replace zeros with NaN in specific columns
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    df[col] = df[col].replace(0, np.nan)

# Fill missing with mean
df.fillna(df.mean(), inplace=True)

# Separate features/target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---- Streamlit App ----
st.title("Diabetes Prediction App")

st.write(f"Model Accuracy: **{accuracy*100:.2f}%**")

# Input form
st.header("Enter Patient Data:")
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 800, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 10, 100, 30)

# Predict button
if st.button("Predict"):
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)
    prob = model.predict_proba(user_scaled)[0][prediction[0]]

    if prediction[0] == 1:
        st.error(f"Prediction: Diabetic (Probability: {prob:.2f})")
    else:
        st.success(f"Prediction: Not Diabetic (Probability: {prob:.2f})")

# ---- Dynamic Visualization Section ----
st.header("Model Performance Visualizations")

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
labels = ['Not Diabetic', 'Diabetic']
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels, ax=ax_cm)
ax_cm.set_xlabel('Predicted label')
ax_cm.set_ylabel('True label')
ax_cm.set_title('Confusion Matrix - Diabetes Prediction')
st.pyplot(fig_cm)

# Feature importance
st.subheader("Feature Importance")
importances = model.feature_importances_
feature_names = X.columns
fig_fi, ax_fi = plt.subplots()
sns.barplot(x=importances, y=feature_names, ax=ax_fi, color='orange')
ax_fi.set_title('Feature Importance - Random Forest')
st.pyplot(fig_fi)
