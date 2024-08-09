import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Function to load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    st.write("Loaded data columns:", df.columns.tolist())  # Print the column names for debugging
    return df

# Function to extract nested dictionary values
def extract_price(df):
    # Check for the correct column name
    price_column = None
    if 'price' in df.columns:
        price_column = 'price'
    elif 'Price' in df.columns:
        price_column = 'Price'
    else:
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

# Function to clean data
def clean_data(df):
    df = extract_price(df)
    
    if 'Price' not in df.columns:
        st.error("The 'Price' column could not be created. Cleaning process stopped.")
        return df
    
    # Convert DepartureDate to datetime
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'], errors='coerce')
    
    # Remove outliers in Price
    q1 = df['Price'].quantile(0.25)
    q3 = df['Price'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
    
    # Fill missing values in DepartureDate using .loc
    df.loc[:, 'DepartureDate'] = df['DepartureDate'].ffill()
    
    return df

# Function to preprocess data
def preprocess_data(df):
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
    df['DayOfWeek'] = df['DepartureDate'].dt.dayofweek
    df['Month'] = df['DepartureDate'].dt.month
    return df

# Function to train model
def train_model(df):
    X = df[['DayOfWeek', 'Month']]
    y = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Function to predict future prices
def predict_future_prices(model, future_df):
    future_df['DayOfWeek'] = future_df['DepartureDate'].dt.dayofweek
    future_df['Month'] = future_df['DepartureDate'].dt.month
    X_future = future_df[['DayOfWeek', 'Month']]
    future_df['PredictedPrice'] = model.predict(X_future)
    return future_df

# Function to save cleaned data
def save_clean_data(df, filepath):
    df.to_csv(filepath, index=False)

# Function to plot historical prices
def plot_historical_prices(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['DepartureDate'], df['Price'], marker='o')
    plt.title('Historical Flight Prices')
    plt.xlabel('Departure Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    st.pyplot(plt)

# Function to plot future price predictions
def plot_future_prices(future_df):
    plt.figure(figsize=(10, 5))
    plt.plot(future_df['DepartureDate'], future_df['PredictedPrice'], marker='o', color='orange')
    plt.title('Predicted Future Flight Prices')
    plt.xlabel('Departure Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True)
    st.pyplot(plt)

# Function to display countdown
def display_countdown(target_date):
    today = datetime.today()
    days_left = (target_date - today).days
    st.metric(label="Days until September 10, 2025", value=days_left)

# Streamlit app
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

