# Auto-Insurance-Price-Predictor

## 📌 Project Overview
This project is an end-to-end Machine Learning application that predicts the **Monthly Premium Auto** insurance cost for customers. It analyzes risk factors such as vehicle class, location, coverage type, and employment status to estimate policy costs accurately.

## 🚀 Features
- **Data Processing:** Cleaned and encoded a dataset of 9,100+ records using `LabelEncoder`.
- **Machine Learning:** Implemented a **Random Forest Regressor** achieving **97% Accuracy (R² Score)**.
- **Interactive Web App:** Deployed a user-friendly frontend using **Streamlit** for real-time predictions.
- **Model Persistence:** Saved the trained model and encoders using `joblib` for instant inference.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Joblib
- **Deployment:** Streamlit

## 📊 Model Performance
- **R² Score:** 0.97 (High Accuracy)
- **Mean Absolute Error (MAE):** ~$4.28

## 🔧 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Auto-Insurance-Price-Predictor.git](https://github.com/your-username/Auto-Insurance-Price-Predictor.git)
## Install dependencies:
   pip install pandas scikit-learn streamlit joblib

## Run The App
   streamlit run app.py
