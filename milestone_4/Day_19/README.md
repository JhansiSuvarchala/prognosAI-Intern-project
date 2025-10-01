🌸 Iris Flower Classifier – Streamlit ML Web App
📌 Project Overview

This project is a Streamlit web application that allows users to:

Explore the Iris dataset with histograms and scatter plots.

Predict the species of an Iris flower based on user-provided measurements using a trained Random Forest model.

The app demonstrates an end-to-end machine learning workflow:

Dataset loading

Model training and saving (train_model.py)

Interactive prediction & data exploration (app.py)

🚀 How to Run
1️⃣ Setup Environment
pip install -r requirements.txt

2️⃣ Train the Model
python train_model.py

This will generate model.pkl in your project folder.

3️⃣ Run the Streamlit App
python -m streamlit run app.py

Streamlit will open a local server, usually at:
👉 http://localhost:8501

📂 Project Structure
streamlit_ml_project/
│── app.py # Streamlit web app
│── train_model.py # Model training script
│── model.pkl # Trained ML model (generated after training)
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── screenshots/ # Screenshots of the app

📊 Features

✅ Data Exploration

Histogram of selected feature

Scatter plot between two features

✅ Prediction Mode

User input via sliders

Predict Iris species

Display prediction probabilities

📸 Screenshots
📋 Dataset Overview

🔍 Data Exploration Mode

📊 Prediction Mode

📝 Evaluation Criteria

Correctness of ML model training & prediction

Functional and interactive Streamlit UI

Data exploration features included

Code readability and organization

✨ Author: Jhansi Suvarchala Koduru
📅 Date: September 2025
