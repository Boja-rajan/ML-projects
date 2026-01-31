
import joblib
import numpy as np
import os
import streamlit as st
st.write("Streamlit is working!")   
# --- Page Configuration ---
st.set_page_config(page_title="ML Predictor", layout="centered")

# --- Model Loading ---
# Using a decorator to cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    model_path = 'linear.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file '{model_path}' not found! Please upload it to your GitHub repo.")
        return None

model = load_model()

# --- UI Layout ---
st.title("🚀 Machine Learning Predictor")
st.write("Enter the required value below to get a prediction from the trained model.")

st.divider()

# Create input fields
# If your model requires multiple features, add more st.number_input fields here
val = st.number_input("Enter your input value:", value=0.0, help="Type the numerical value for prediction.")

if st.button("Predict", type="primary"):
    if model is not None:
        try:
            # Reshape input for a single prediction: [[value]]
            prediction = model.predict([[val]])
            
            # Display results
            st.subheader("Results:")
            st.success(f"The predicted result is: **{prediction[0]}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Prediction unavailable: Model not loaded.")

st.divider()
st.caption("Built with Streamlit • Hosted on GitHub")

