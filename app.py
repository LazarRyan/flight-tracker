import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ... (keep the existing load_data, extract_price, and clean_data functions) ...

# Function to preprocess data
def preprocess_data(df):
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
    df['DaysBeforeDeparture'] = (df['DepartureDate'] - df['DepartureDate'].min()).dt.days
    df['DayOfWeek'] = df['DepartureDate'].dt.dayofweek
    df['Month'] = df['DepartureDate'].dt.month
    return df

# Function to train model
def train_model(df):
    X = df[['DaysBeforeDeparture', 'DayOfWeek', 'Month']]
    y = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Function to predict future prices
def predict_future_prices(model, target_date, days_to_predict):
    future_dates = pd.date_range(end=target_date, periods=days_to_predict)
    future_df = pd.DataFrame({'DepartureDate': future_dates})
    future_df['DaysBeforeDeparture'] = (target_date - future_df['DepartureDate']).dt.days
    future_df['DayOfWeek'] = future_df['DepartureDate'].dt.dayofweek
    future_df['Month'] = future_df['DepartureDate'].dt.month
    
    X_future = future_df[['DaysBeforeDeparture', 'DayOfWeek', 'Month']]
    future_df['PredictedPrice'] = model.predict(X_future)
    return future_df

# Function to find best days to buy
def find_best_days_to_buy(future_df, n=5):
    best_days = future_df.nsmallest(n, 'PredictedPrice')
    return best_days

# Function to plot price predictions and best days
def plot_price_predictions(future_df, best_days):
    plt.figure(figsize=(12, 6))
    plt.plot(future_df['DepartureDate'], future_df['PredictedPrice'], marker='o', alpha=0.5)
    plt.scatter(best_days['DepartureDate'], best_days['PredictedPrice'], color='red', s=100, zorder=5)
    plt.title('Predicted Flight Prices and Best Days to Buy')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True)
    for _, row in best_days.iterrows():
        plt.annotate(f"${row['PredictedPrice']:.2f}", 
                     (row['DepartureDate'], row['PredictedPrice']),
                     textcoords="offset points", xytext=(0,10), ha='center')
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Flight Price Predictor")

    historical_csv = st.sidebar.text_input("Path to Historical Data CSV", value="flight_prices.csv")
    target_date = st.sidebar.date_input("Select target flight date", value=datetime(2025, 9, 10))
    days_to_predict = st.sidebar.slider("Days to predict before target date", 30, 365, 180)

    st.write("Loading and cleaning historical data...")
    df = load_data(historical_csv)
    df_clean = clean_data(df)

    if 'Price' in df_clean.columns:
        st.success("Data cleaned successfully")
        
        st.write("Training model...")
        df = preprocess_data(df_clean)
        model = train_model(df)

        st.write("Predicting future prices...")
        future_df = predict_future_prices(model, target_date, days_to_predict)
        
        st.write("Finding best days to buy...")
        best_days = find_best_days_to_buy(future_df)
        
        st.write("Visualizing price predictions and best days to buy...")
        plot_price_predictions(future_df, best_days)

        st.write("Best Days to Buy Tickets:")
        st.dataframe(best_days[['DepartureDate', 'PredictedPrice']])

if __name__ == "__main__":
    main()
