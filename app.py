# Gender -> Female 1 Male 0
# Churn -> Yes 1 No 0
# Scalar is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the x Index(['Age', 'MonthlyCharges', 'Tenure', 'Gender'], dtype='object')

import streamlit as st
import numpy as np
import joblib

# Load the trained scaler and model
scalar = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Streamlit UI
st.title("📊 Customer Churn Prediction App")
st.divider()
st.write("This app predicts whether a customer will churn based on their details.")
st.divider()

# Input fields
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30, step=1)
monthly_charges = st.number_input("Enter Monthly Charges", min_value=0, max_value=10000, value=70, step=1)
tenure = st.number_input("Enter Tenure (in months)", min_value=0, max_value=100, value=12, step=1)
gender = st.selectbox("Select Gender", ["Male", "Female"])

st.divider()

# Prediction button
if st.button("Predict"):
    gender_selected = 1 if gender == "Female" else 0

    # Prepare input in correct order
    x = np.array([[age, monthly_charges, tenure, gender_selected]])

    # Scale input
    x_scaled = scalar.transform(x)

    # Predict churn
    prediction = model.predict(x_scaled)[0]

    # Try to get probability (not all models support predict_proba)
    try:
        prob = model.predict_proba(x_scaled)[0][1] * 100  # Probability of churn (Yes)
    except AttributeError:
        prob = None

    # Display result
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to CHURN.")
        if prob is not None:
            st.write(f"**Churn Probability:** {prob:.2f}%")
    else:
        st.success(f"🟢 The customer is likely to STAY.")
        if prob is not None:
            st.write(f"**Retention Probability:** {100 - prob:.2f}%")

    st.balloons()

else:
    st.info("Click on the **Predict** button to get the result.")
