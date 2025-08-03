
Diabetes Prediction App

This project is a user-friendly web application built using Streamlit, designed to predict whether a patient is likely to have diabetes based on basic medical inputs. It demonstrates an end-to-end machine learning workflow and makes health data accessible through interactive technology.

Project Overview

Diabetes is a major global health challenge, and early detection is crucial. This app uses historical data to build a prediction model that can assist in identifying potential diabetes cases with a simple form.

- Goal: Predict diabetes status (Yes/No) using medical data inputs.
- Model Used: Random Forest Classifier
- Input Data: Patient features like Glucose level, Blood Pressure, BMI, Age, etc.

Live Demo
Use the app here: [Streamlit Live App](YOUR_STREAMLIT_LINK_HERE)

Dataset Summary
- Total Records: 768 patients
- Features:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
  - Outcome (Target: 1=Diabetic, 0=Not Diabetic)

Machine Learning Workflow
1. Data Cleaning:
   - Replaced 0s in some columns (e.g., Glucose) with NaN.
   - Filled missing values with column averages.

2. Feature Scaling:
   - StandardScaler used to normalize input features.

3. Model Training:
   - RandomForestClassifier with 100 trees.
   - 80% Training, 20% Testing split.

4. Model Evaluation:
   - Accuracy: 75%
   - Precision (Diabetic): 68%
   - Recall (Diabetic): 58%
   - F1-Score (Diabetic): 63%

Prediction Results Example
| Input          | Value            |
| -------------- | ---------------- |
| Glucose        | 140              |
| Blood Pressure | 70               |
| BMI            | 32.0             |
| Age            | 30               |
| Result         | Not Diabetic     |
| Probability    | 0.73             |

Conclusions
- The model achieves solid accuracy (75%) for a beginner project.
- Prediction is more reliable for non-diabetic cases.
- Additional data or advanced models (e.g., XGBoost) could improve accuracy.
- This app is a strong base for health-related AI solutions.

How to Run Locally
1. Clone the repo:
https://github.com/YOUR_USERNAME/diabetes_prediction_project.git

2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run app.py

Technologies Used
- Python
- Pandas, NumPy, Scikit-learn
- Streamlit

Author
Mohammed Majdoub
2nd Year Info Systems & Data Science
University of Haifa

Future Improvements
- Model tuning and boosting for higher accuracy
- Add model comparison (e.g., XGBoost, Logistic Regression)
- Visualization of feature importance
- Store patient results securely (privacy-focused)

License
This project is open source. You are free to use and modify it.

"Turning data into insight is the first step to turning insight into action."
