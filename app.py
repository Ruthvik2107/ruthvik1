import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # Support Vector Machine (SVR) for regression

# Step 1: Fetch data from the API
url = "https://disease.sh/v3/covid-19/countries/japan"
r = requests.get(url)
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

# Step 2: Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
print(df)

# Step 3: Plot bar chart for current COVID-19 data
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]
plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for JAPAN")
plt.show()

# Step 4: Generate random historical data for last 30 days (for illustration)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)
df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)
print(df_historical.head())

# Step 5: Prepare data for training and testing
X = df_historical[["day"]]  # Features (days)
y = df_historical["cases"]  # Target (cases)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize and train Support Vector Regression (SVR) model
model = SVR(kernel='rbf')  # Radial basis function kernel for non-linear regression
model.fit(X_train, y_train)

# Step 7: Predict next day's cases using the trained SVR model
next_day = np.array([[31]])  # Predict for day 31
predicted_cases = model.predict(next_day)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

# Step 8: Streamlit UI for user interaction
st.title("COVID-19 Cases Prediction in JAPAN")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input: Day number for prediction
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
