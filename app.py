import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # Support Vector Regression
import streamlit as st

# Fetch COVID-19 data
url = "https://disease.sh/v3/covid-19/countries/uk"
r = requests.get(url)
if r.status_code != 200:
    st.error("Failed to fetch data. Please try again later.")
else:
    data = r.json()

    # Extract relevant fields
    covid_data = {
        "cases": data["cases"],
        "todayCases": data["todayCases"],
        "deaths": data["deaths"],
        "todayDeaths": data["todayDeaths"],
        "recovered": data["recovered"],
        "active": data["active"],
        "critical": data["critical"],
        "casesPerMillion": data["casesPerOneMillion"],
        "deathsPerMillion": data["deathsPerOneMillion"],
    }

    # Convert to Pandas DataFrame
    df = pd.DataFrame([covid_data])
    st.write("COVID-19 Data for the UK:")
    st.write(df)

    # Visualization: Bar chart of COVID data
    labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
    values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("COVID-19 Data for UK")
    st.pyplot(plt)

    # Generate synthetic historical data for cases and deaths over the last 30 days
    np.random.seed(42)
    historical_cases = np.random.randint(30000, 70000, size=30)  # Random data for cases
    historical_deaths = np.random.randint(500, 2000, size=30)  # Random data for deaths

    df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
    df_historical["day"] = range(1, 31)

    # Display the first few rows of historical data
    st.write("Historical Data (Cases & Deaths) for the last 30 days:")
    st.write(df_historical.head())

    # Split data into features (X) and target (y)
    X = df_historical[["day"]]
    y = df_historical["cases"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the SVM model (SVR)
    svr = SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train)

    # Predict the next day's cases (Day 31)
    next_day = np.array([[31]])
    predicted_cases = svr.predict(next_day)
    st.write(f"Predicted cases for Day 31 using SVM: {int(predicted_cases[0])}")

    # Streamlit interface for user input
    st.title("COVID-19 Cases Prediction for the UK")
    st.write("Predicting COVID-19 cases for the next day based on historical data.")

    # User Input for prediction
    day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

    if st.button("Predict"):
        prediction = svr.predict([[day_input]])
        st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
