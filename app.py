import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    st.write("Loaded data columns:", df.columns.tolist())
    return df

# Function to clean and preprocess data
def preprocess_data(df):
    df['DaysToFlight'] = (df['FlightDate'] - df['Date']).dt.days
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    return df

# Function to train model
def train_model(df):
    X = df[['DaysToFlight', 'DayOfWeek', 'Month']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Function to predict future prices
def predict_prices(model, start_date, end_date, flight_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'Date': date_range})
    future_df['DaysToFlight'] = (flight_date - future_df['Date']).dt.days
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['Month'] = future_df['Date'].dt.month
    
    X_future = future_df[['DaysToFlight', 'DayOfWeek', 'Month']]
    future_df['PredictedPrice'] = model.predict(X_future)
    return future_df

# Function to find best days to buy
def find_best_days_to_buy(future_df, n=5):
    return future_df.nsmallest(n, 'PredictedPrice')

# Function to plot price predictions
def plot_price_predictions(future_df, best_days):
    plt.figure(figsize=(12, 6))
    plt.plot(future_df['Date'], future_df['PredictedPrice'], marker='', linewidth=2)
    plt.scatter(best_days['Date'], best_days['PredictedPrice'], color='red', s=50, zorder=5)
    plt.title('Predicted Flight Prices and Best Days to Buy')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    for _, row in best_days.iterrows():
        plt.annotate(f"${row['PredictedPrice']:.2f}\n{row['Date'].strftime('%Y-%m-%d')}", 
                     (row['Date'], row['PredictedPrice']),
                     textcoords="offset points", xytext=(0,10), ha='center')
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Flight Price Predictor")

    # File uploader
    uploaded_file = st.file_uploader("Upload your historical flight data CSV", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = preprocess_data(df)
        st.success("Data loaded and preprocessed successfully!")

        # Train model
        model = train_model(df)
        st.write("Model trained successfully!")

        # User inputs
        flight_date = st.date_input("Select your flight date", min_value=datetime.now().date() + timedelta(days=1))
        prediction_start = st.date_input("Start date for price prediction", value=datetime.now().date())
        prediction_end = st.date_input("End date for price prediction", value=flight_date - timedelta(days=1))

        if st.button("Predict Prices"):
            future_df = predict_prices(model, prediction_start, prediction_end, flight_date)
            best_days = find_best_days_to_buy(future_df)

            st.subheader("Price Prediction Chart")
            plot_price_predictions(future_df, best_days)

            st.subheader("Best Days to Buy Tickets")
            st.dataframe(best_days[['Date', 'PredictedPrice']].set_index('Date'))

            st.subheader("All Predicted Prices")
            st.dataframe(future_df[['Date', 'PredictedPrice']].set_index('Date'))

if __name__ == "__main__":
    main()
