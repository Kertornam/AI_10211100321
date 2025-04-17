import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def run_regression():
    st.title("🏠 Regression: Predict California Housing Prices")

    # Load dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target * 100000

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Feature selection
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']
    target = 'Price'
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"**Mean Absolute Error:** ${mae:,.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Plot
    st.subheader("🔍 Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted House Prices")
    st.pyplot(fig)

    # Prediction from custom input
    st.subheader("🎯 Predict New House Price")
    medinc = st.number_input("Median Income (MedInc)", min_value=0.0, value=5.0)
    house_age = st.slider("House Age", 1, 50, 20)
    averooms = st.number_input("Average Rooms", min_value=1.0, value=5.0)
    aveoccup = st.number_input("Average Occupants", min_value=1.0, value=3.0)

    input_data = pd.DataFrame([[medinc, house_age, averooms, aveoccup]], columns=features)
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Predicted House Price: **${prediction:,.2f}**")
