# House Price Prediction Using Machine Learning

## Project Overview
This project predicts house prices using Machine Learning. It uses property features such as area, bedrooms, bathrooms, parking, and furnishing status to estimate house prices.

## Project Domain
Machine Learning & Data Science

## Project Type
Supervised Machine Learning – Regression

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Features
- User Registration and Login
- Interactive Dashboard
- Exploratory Data Analysis (EDA)
- House Price Prediction
- Prediction History
- Model Performance Analysis
- User Profile Management

## Project Structure
```text
House-Price-Prediction/
│
├── app.py
├── auth.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── Housing.csv
│
├── models/
│   └── house_price_model.pkl
│
└── notebooks/
    └── HousePricePrediction.ipynb
```

## Installation and Setup

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install the required libraries:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit application:

```bash
streamlit run app.py
```

## Machine Learning Model
The project uses Linear Regression for house price prediction.

## Project Objective
To build an interactive Machine Learning application that estimates house prices based on property characteristics.
