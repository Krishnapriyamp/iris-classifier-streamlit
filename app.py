# app.py - Streamlit Iris predictor
import streamlit as st
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Iris Species Predictor", layout="centered")

st.title("Iris Flower Species Predictor 🌸")
st.write("Enter measurements (in cm) and click Predict — model trained with scikit-learn")

# try to load model & scaler from same folder
model_path = Path("iris_model.joblib")
scaler_path = Path("scaler.joblib")

if not model_path.exists() or not scaler_path.exists():
    st.warning("Model or scaler not found. Make sure iris_model.joblib and scaler.joblib are in this folder.")
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, step=0.1)
    sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, step=0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.35, step=0.1)
    petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.3, step=0.1)

    if st.button("Predict"):
        X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        proba = model.predict_proba(Xs)[0]
        names = ["setosa", "versicolor", "virginica"]
        st.success(f"Predicted species: **{names[pred]}**")
        st.write("Probabilities:")
        for n, p in zip(names, proba):
            st.write(f"- {n}: {p:.2f}")
