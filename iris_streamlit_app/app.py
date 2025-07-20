
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🌼 Iris Flower Species Predictor")
st.write("Input flower dimensions and get a prediction of its species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)
probability = model.predict_proba(features)

species = ["Setosa", "Versicolor", "Virginica"]

st.subheader("Prediction")
st.success(f"The predicted species is: **{species[prediction[0]]}**")

st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(probability, columns=species)
st.bar_chart(proba_df.T)

if st.checkbox("Show Iris dataset scatter plot"):
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", ax=ax)
    st.pyplot(fig)
