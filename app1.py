import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load model
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🍷 Wine Quality Prediction")

# All 13 input sliders
alcohol = st.slider("Alcohol", 10.0, 15.0, 12.0)
malic_acid = st.slider("Malic Acid", 0.0, 6.0, 2.0)
ash = st.slider("Ash", 1.0, 3.5, 2.0)
alcalinity_of_ash = st.slider("Alcalinity of Ash", 10.0, 30.0, 18.0)
magnesium = st.slider("Magnesium", 70, 162, 100)
total_phenols = st.slider("Total Phenols", 0.5, 4.0, 2.0)
flavanoids = st.slider("Flavanoids", 0.0, 5.0, 2.0)
nonflavanoid_phenols = st.slider("Nonflavanoid Phenols", 0.0, 1.0, 0.5)
proanthocyanins = st.slider("Proanthocyanins", 0.0, 4.0, 1.0)
color_intensity = st.slider("Color Intensity", 1.0, 13.0, 5.0)
hue = st.slider("Hue", 0.4, 1.8, 1.0)
od280 = st.slider("OD280/OD315 of Diluted Wines", 0.5, 4.0, 2.0)
proline = st.slider("Proline", 300, 1700, 1000)

# Combine all inputs
features = np.array([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
                      total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
                      color_intensity, hue, od280, proline]])

# 1. Sidebar for inputs
st.sidebar.header("Input Features")
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 12.5)
malic_acid = st.sidebar.slider("Malic Acid", 0.5, 5.0, 2.5)
# ... Add other features

features = np.array([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
                      total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
                      color_intensity, hue, od280, proline]])  # Make sure order matches training!

# 2. Predict on button click
if st.button("Predict"):
    prediction = model.predict(features)

    # 3. Show prediction result
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", str(prediction[0]))
    with col2:
        st.metric("Confidence", f"{np.max(model.predict_proba(features)) * 100:.2f}%")

    # 4. Optional: Feature importance
    if st.checkbox("Show Feature Importance"):
        import matplotlib.pyplot as plt

        importances = model.feature_importances_
        feature_names = ["alcohol", "malic_acid", "ash, alcalinity_of_ash", "magnesium",
                      "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                      "color_intensity", "hue", "od280", "proline"]  # Put all 13 here

        fig, ax = plt.subplots()
        ax.barh(feature_names, importances)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

# Make prediction
prediction = model.predict(features)

st.success(f"🍇 Predicted Wine Class: {prediction[0]}")

# Plot feature importance
if st.checkbox("Show Feature Importance"):
    importances = model.feature_importances_
    feature_names = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                     "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                     "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"]

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)