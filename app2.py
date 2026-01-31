import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Insurance Cost Predictor")
st.markdown("### Predict your medical insurance charges instantly")

# -------------------------
# Load & Train Model
# -------------------------
@st.cache_data
def train_model():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)

    df_encoded = pd.get_dummies(
        df,
        columns=["sex", "smoker", "region"],
        drop_first=True
    )

    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns

model, feature_columns = train_model()

# -------------------------
# Sidebar Inputs (8 inputs)
# -------------------------
st.sidebar.header("🧾 Enter Your Details")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.radio("Sex", ["male", "female"])

height = st.sidebar.slider("Height (cm)", 140, 210, 170)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)

children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.radio("Smoker", ["yes", "no"])

region = st.sidebar.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

activity = st.sidebar.selectbox(
    "Physical Activity Level",
    ["Low", "Moderate", "High"]
)

# -------------------------
# Feature Engineering
# -------------------------
bmi = round(weight / ((height / 100) ** 2), 2)

st.info(f"📊 **Calculated BMI:** `{bmi}`")

# -------------------------
# Prepare Input Data
# -------------------------
input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
}

input_df = pd.DataFrame([input_dict])

# Ensure column order matches training
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -------------------------
# Prediction
# -------------------------
if st.button("🔮 Predict Insurance Cost"):
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.success("### ✅ Prediction Result")
    st.markdown(
        f"""
        ## 💵 **Estimated Insurance Cost**
        ### **${prediction:,.2f}**
        """
    )

    st.caption("⚠️ This is a machine learning estimate, not a guaranteed quote.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn")



