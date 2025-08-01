import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/rf_model.pkl")

# App title
st.title("🏠 Residential Energy Analytics")
st.write("Predict daily energy consumption based on home parameters.")

# Sidebar for inputs
st.sidebar.header("Enter Home Details")
square_ft = st.sidebar.slider("Square Footage", 500, 5000, 1500)
occupants = st.sidebar.slider("Number of Occupants", 1, 10, 3)
avg_temp = st.sidebar.slider("Avg Indoor Temperature (°C)", 15, 35, 24)
ac_usage = st.sidebar.slider("AC Usage (hrs/day)", 0, 24, 6)
heater_usage = st.sidebar.slider("Heater Usage (hrs/day)", 0, 24, 2)

# Create input DataFrame
input_data = pd.DataFrame({
    "square_ft": [square_ft],
    "occupants": [occupants],
    "avg_temp": [avg_temp],
    "ac_usage": [ac_usage],
    "heater_usage": [heater_usage]
})

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("🔌 Predicted Daily Energy Consumption:")
    st.success(f"{prediction:.2f} kWh")

# Optional: Sample Dataset Viewer
with st.expander("📂 View Sample Data"):
    st.write(pd.read_csv("data/energy_data.csv"))
