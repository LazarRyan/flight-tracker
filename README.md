# Italy 2025 - Tanner & Jill Tie The Knot
## This repository contains a Streamlit application designed to predict flight prices and provide visual insights for an upcoming trip to Italy for Tanner and Jill's wedding on September 10th, 2025. The app also includes a countdown timer to the wedding date, providing users with a constant reminder of how many days are left until the big day.

### Table of Contents
Overview
Features
Installation
Usage
Detailed Code Explanation
Imports and Page Configuration
Data Loading and Cleaning
Model Training and Prediction
Visualizations
Countdown Timer
Main Function
Data Requirements
Customization
License
Acknowledgements
Overview

The "Italy 2025 - Tanner & Jill Tie The Knot" app serves as a tool to track and predict flight prices leading up to a specific event. It is built using Streamlit, a powerful framework for creating web apps using Python. The app allows users to upload historical flight data, train a machine learning model, and predict future flight prices. Additionally, it visualizes the data through interactive charts and includes a countdown timer to the event.

Features
Countdown Timer: A real-time countdown showing the days left until September 10th, 2025.
Data Upload and Cleaning: Upload CSV files containing historical flight data, which the app will automatically clean and prepare for analysis.
Model Training: Train a machine learning model (RandomForestRegressor) on the historical data to predict future flight prices.
Flight Price Predictions: Predict flight prices for future dates based on the trained model.
Visualizations: Interactive charts for visualizing historical flight prices and future predictions.
Installation
To run this app locally, follow these steps:

Clone the Repository:

bash
git clone https://github.com/yourusername/italy-2025-tanner-jill.git
cd italy-2025-tanner-jill
Create a Virtual Environment:

bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit App:

bash
streamlit run app.py
Usage
Once the app is running, you can interact with the following features:

Countdown Timer: Shows the days remaining until the wedding on September 10th, 2025.
Upload Historical Data: Upload a CSV file containing historical flight data.
Train Model: Optionally train a machine learning model on the historical data.
Predict Future Prices: Use the trained model to predict future flight prices.
Visualize Data: View interactive charts that display historical and predicted flight prices.

# Detailed Code Explanation

# Imports and Page Configuration
python
Copy code
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Italy 2025 - Tanner & Jill Tie The Knot")
Imports: The app uses several Python libraries, including Streamlit for the web interface, pandas for data manipulation, numpy for numerical operations, JSON for data parsing, scikit-learn for machine learning, joblib for model saving/loading, matplotlib for plotting, and datetime for time-based operations.
Page Configuration: The st.set_page_config() function sets the title of the browser tab to "Italy 2025 - Tanner & Jill Tie The Knot".
Data Loading and Cleaning
python
Copy code
def load_data(filepath):
    df = pd.read_csv(filepath)
    st.write("Loaded data columns:", df.columns.tolist())  # Print the column names for debugging
    return df

def extract_price(df):
    # Check for the correct column name
    price_column = 'price' if 'price' in df.columns else 'Price' if 'Price' in df.columns else None
    if not price_column:
        st.error("The 'price' column is not found in the dataset.")
        return df
    
    def extract_price_value(price_data):
        try:
            if isinstance(price_data, str):
                price_dict = json.loads(price_data.replace("'", "\""))
                return float(price_dict.get('total', np.nan))
            else:
                return float(price_data)
        except (TypeError, json.JSONDecodeError, KeyError):
            st.warning(f"Error processing price data: {price_data}")
            return np.nan
    
    df['Price'] = df[price_column].apply(extract_price_value)
    return df

def clean_data(df):
    df = extract_price(df)
    if 'Price' not in df.columns:
        st.error("The 'Price' column could not be created. Cleaning process stopped.")
        return df
    
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'], errors='coerce')
    q1, q3 = df['Price'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
    df.loc[:, 'DepartureDate'] = df['DepartureDate'].ffill()
    
    return df

Data Loading: The load_data() function reads a CSV file into a pandas DataFrame and displays the column names in the Streamlit interface for debugging.
Price Extraction: The extract_price() function extracts price data from a nested JSON structure in the DataFrame and handles any potential errors.
Data Cleaning: The clean_data() function removes outliers, converts dates to datetime objects, and fills missing values, ensuring that the data is ready for analysis.

# Model Training and Prediction
python
def preprocess_data(df):
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
    df['DayOfWeek'] = df['DepartureDate'].dt.dayofweek
    df['Month'] = df['DepartureDate'].dt.month
    return df

def train_model(df):
    X = df[['DayOfWeek', 'Month']]
    y = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future_prices(model, future_df):
    future_df['DayOfWeek'] = future_df['DepartureDate'].dt.dayofweek
    future_df['Month'] = future_df['DepartureDate'].dt.month
    X_future = future_df[['DayOfWeek', 'Month']]
    future_df['PredictedPrice'] = model.predict(X_future)
    return future_df

Data Preprocessing: The preprocess_data() function adds features like the day of the week and month to the DataFrame, which are used in model training.
Model Training: The train_model() function trains a RandomForestRegressor model using the preprocessed data, predicting prices based on the day of the week and month.
Price Prediction: The predict_future_prices() function uses the trained model to predict future flight prices based on the selected dates.

# Visualizations
python
def plot_historical_prices(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['DepartureDate'], df['Price'], marker='o')
    plt.title('Historical Flight Prices')
    plt.xlabel('Departure Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    st.pyplot(plt)

def plot_future_prices(future_df):
    plt.figure(figsize=(10, 5))
    plt.plot(future_df['DepartureDate'], future_df['PredictedPrice'], marker='o', color='orange')
    plt.title('Predicted Future Flight Prices')
    plt.xlabel('Departure Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True)
    st.pyplot(plt)

### Historical Prices Plot: The plot_historical_prices() function generates a line chart showing the historical flight prices over time.
### Future Prices Plot: The plot_future_prices() function generates a line chart displaying the predicted future flight prices.

# Countdown Timer
python
def display_countdown(target_date):
    today = datetime.today()
    days_left = (target_date - today).days
    st.metric(label="Days until September 10, 2025", value=days_left)
Countdown Timer: The display_countdown() function calculates the number of days remaining until September 10, 2025, and displays it in the Streamlit app using a metric widget.
Main Function
python
Copy code
def main():
    st.title("Italy 2025 - Tanner & Jill Tie The Knot")

    # Countdown to September 10, 2025
    target_date = datetime(2025, 9, 10)
    display_countdown(target_date)

    historical_csv = st.sidebar.text_input("Path to Historical Data CSV", value="flight_prices.csv")
    train_model_option = st.sidebar.checkbox("Train Model", value=True)
    predict_future_option = st.sidebar.checkbox("Predict Future Prices", value=True)

    if train_model_option:
        st.write("Loading and cleaning historical data...")
        df = load_data(historical_csv)
        df_clean = clean_data(df)
        if 'Price' in df_clean.columns:
            save_clean_data(df_clean, 'cleaned_flight_prices.csv')
            st.success("Data cleaned and saved to cleaned_flight_prices.csv")
            
            st.write("Training model...")
            df = preprocess_data(df_clean)
            model = train_model(df)
            joblib.dump(model, 'flight_price_model.pkl')
            st.success("Model trained and saved to flight_price_model.pkl")

            st.write("Visualizing historical flight prices...")
            plot_historical_prices(df_clean)

    if predict_future_option:
        st.write("Predicting future prices...")
        model = joblib.load('flight_price_model.pkl')
        future_dates = pd.date_range(start='2024-12-02', end='2025-09-05')
        future_df = pd.DataFrame(future_dates, columns=['DepartureDate'])
        future_df = predict_future_prices(model, future_df)
        future_df.to_csv('future_prices.csv', index=False)
        st.success("Future prices predicted and saved to future_prices.csv")

        st.write("Visualizing future flight price predictions...")
        plot_future_prices(future_df)

        st.write("Future Price Predictions")
        st.dataframe(future_df)

if __name__ == "__main__":
    main()

Main Function: The main() function orchestrates the entire app. It sets up the title, handles user input for uploading data and training the model, displays visualizations, and runs the countdown timer.
Data Requirements

# The app expects a CSV file with historical flight data. The data should include at least the following columns:

DepartureDate: The date of the flight.
Price: The price of the flight, which can be in a nested JSON format if needed.
Customization
Page Title: Modify the st.set_page_config() function to change the page title.
Prediction Date Range: Adjust the future_dates variable in the main() function to change the range of future dates for which prices are predicted.
Visualizations: You can add or modify the charts to include additional insights or different data visualizations.
License

# This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
Thanks to the Streamlit team for providing an easy-to-use framework for building web applications.
Special thanks to all contributors and supporters who helped make this project possible.
