import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("🚲 Bike Demand Prediction App")

# -----------------------------
# Load Data (Bullet-proof)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Required columns
    FEATURES = ["season", "yr", "mnth", "hr", "temp", "atemp", "hum", "windspeed"]
    TARGET = "cnt"

    # Keep only required columns
    df = df[FEATURES + [TARGET]]

    # Convert everything to numeric safely
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values with median (safe for ML)
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df

# -----------------------------
# Load Dataset
# -----------------------------
df = load_data()

FEATURES = ["season", "yr", "mnth", "hr", "temp", "atemp", "hum", "windspeed"]
TARGET = "cnt"

from sklearn.model_selection import train_test_split

X = df[FEATURES].values.astype(float)
y = df[TARGET].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)



# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Values")

season = st.sidebar.selectbox(
    "Season",
    [1, 2, 3, 4],
    format_func=lambda x: {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}[x]
)

yr = st.sidebar.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 12)

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temp", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

input_data = np.array([[season, yr, mnth, hr, temp, atemp, hum, windspeed]], dtype=float)

st.subheader("Selected Inputs")
st.write(pd.DataFrame(input_data, columns=FEATURES))

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Predicted Bike Count: {int(prediction)}")


