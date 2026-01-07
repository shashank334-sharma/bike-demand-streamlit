import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("🚲 Bike Demand Prediction App")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.CSV")   
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

X = df[["temp", "atemp", "hum", "windspeed"]]
y = df["cnt"]

# -----------------------------
# Train Model
# -----------------------------
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=150, random_state=42))
])

model.fit(X, y)

# -----------------------------
# Model Accuracy
# -----------------------------
y_pred = model.predict(X)
accuracy = r2_score(y, y_pred)
st.info(f"📊 Model Accuracy (R² Score): {accuracy:.2f}")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Values")

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temp", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

input_data = pd.DataFrame({
    "temp": [temp],
    "atemp": [atemp],
    "hum": [hum],
    "windspeed": [windspeed]
})

st.subheader("Selected Inputs")
st.write(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Predicted Bike Count: {int(prediction)}")

