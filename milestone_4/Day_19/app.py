import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and model
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
model = joblib.load("model.pkl")

# App Title
st.title("🌸 Iris Flower Classifier")
st.markdown("This app predicts the type of Iris flower based on input features and also allows dataset exploration.")

# Sidebar navigation
mode = st.sidebar.radio("Choose Mode:", ["🔍 Data Exploration", "📊 Prediction"])

if mode == "🔍 Data Exploration":
    st.subheader("Dataset Overview")
    st.dataframe(X.head())

    # Histogram
    feature = st.selectbox("Select feature for histogram", X.columns)
    fig, ax = plt.subplots()
    sns.histplot(X[feature], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Scatter plot
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("X-axis", X.columns, index=0)
    y_axis = st.selectbox("Y-axis", X.columns, index=1)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X[x_axis], y=X[y_axis], hue=y.map(lambda i: iris.target_names[i]), ax=ax)
    st.pyplot(fig)

elif mode == "📊 Prediction":
    st.subheader("Enter Flower Measurements")

    # Input widgets
    sepal_length = st.slider("Sepal length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()))
    sepal_width  = st.slider("Sepal width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()))
    petal_length = st.slider("Petal length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()))
    petal_width  = st.slider("Petal width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()))

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.success(f"🌼 Predicted Species: **{iris.target_names[prediction]}**")
        st.write("Prediction Probabilities:")
        for species, p in zip(iris.target_names, proba):
            st.write(f"- {species}: {p:.2f}")
