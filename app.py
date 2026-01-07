import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("🚲 Bike Demand Prediction App")

# -----------------------------
# Features & Target (GLOBAL)
# -----------------------------
FEATURES = ["season", "yr", "mnth", "hr", "temp", "atemp", "hum", "windspeed"]
TARGET = "cnt"

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")   # make sure filename matches GitHub exactly

    # Replace ? with NaN
    df.replace("?", np.nan, inplace=True)

    # Convert all features + target to numeric
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values
    df.dropna(subset=FEATURES + [TARGET], inplace=True)

    return df

# ✅ Load dataset
df = load_data()

# -----------------------------
# Prepare Data
# -----------------------------
X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# Train Model
# -----------------------------
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=250, random_state=42))
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

# Season input
season = st.sidebar.selectbox(
    "Season",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Spring 🌸",
        2: "Summer ☀️",
        3: "Fall 🍂",
        4: "Winter ❄️"
    }[x]
)

# Time inputs
yr = st.sidebar.selectbox(
    "Year",
    [0, 1],
    format_func=lambda x: "2011" if x == 0 else "2012"
)
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 12)

# Weather inputs
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temp", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

# Input dataframe
input_data = pd.DataFrame({
    "season": [season],
    "yr": [yr],
    "mnth": [mnth],
    "hr": [hr],
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
