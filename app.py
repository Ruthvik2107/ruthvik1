import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Fetch COVID-19 data for the UK
url = "https://disease.sh/v3/covid-19/countries/uk"
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

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])

# Generate random historical data for demonstration
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Feature and target variables for prediction models
X = df_historical[["day"]]
y = df_historical["cases"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
linear_model = LinearRegression()
svm_model = SVR(kernel='linear')

# Train both models
linear_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Streamlit app
st.title("COVID-19 Cases Prediction in UK")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# Display the current COVID-19 stats for the UK
st.write("COVID-19 Stats for UK:")
st.write(f"Total Cases: {data['cases']}")
st.write(f"Active Cases: {data['active']}")
st.write(f"Recovered: {data['recovered']}")
st.write(f"Deaths: {data['deaths']}")

# Plotting the historical data (cases over the days)
plt.figure(figsize=(10, 6))
plt.plot(df_historical["day"], df_historical["cases"], label="Cases", color='blue', marker='o')
plt.xlabel("Day")
plt.ylabel("Number of Cases")
plt.title("COVID-19 Cases Over the Last 30 Days in UK")
plt.grid(True)
plt.legend()
st.pyplot(plt)  # Display the plot in Streamlit

# User Input for day
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

# Predict using both models
if st.button("Predict"):
    # Linear regression prediction
    linear_prediction = linear_model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input} using Linear Regression: {int(linear_prediction[0])}")
    
    # SVM regression prediction
    svm_prediction = svm_model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input} using SVM (Linear Kernel): {int(svm_prediction[0])}")
