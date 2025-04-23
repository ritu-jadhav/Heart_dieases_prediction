import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart (1).csv")  # This uses your uploaded file

df = load_data()

# Set features and target
target_column = 'target'
features_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X = df.drop(target_column, axis=1)
y = df[target_column]

# Standardize numerical features
scaler = StandardScaler()
X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

# Train a logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.write("Fill out the form to predict the risk of heart disease.")

with st.form("heart_form"):
    age = st.slider("Age", 25, 80, 50)
    sex = st.selectbox("Sex", [0,1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol", 100, 600, 245)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_df = pd.DataFrame(user_input, columns=X.columns)
    input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of heart disease. (Probability: {prob:.2f})")
