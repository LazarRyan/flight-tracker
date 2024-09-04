import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import os

# Set page config
st.set_page_config(page_title="Flight Price Predictor", layout="wide")

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    
    def extract_price(price_data):
        try:
            price_dict = json.loads(price_data.replace("'", "\""))
            return float(price_dict['price']['total'])
        except:
            return np.nan
    
    def extract_departure(itinerary_data):
        try:
            itinerary_dict = json.loads(itinerary_data.replace("'", "\""))
            return itinerary_dict[0]['segments'][0]['departure']['at']
        except:
            return np.nan
    
    df['price'] = df['price_details'].apply(extract_price)
    df['departure'] = df['itineraries'].apply(extract_departure)
    df['departure'] = pd.to_datetime(df['departure'])
    
    df = df.dropna(subset=['price', 'departure'])
    df = df[(df['price'] > 0) & (df['departure'] > '2023-01-01')]
    
    return df[['departure', 'price']]

# Function to engineer features
def engineer_features(df):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['days_to_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

# Function to train model
def train_model(df):
    X = df[['day_of_week', 'month', 'days_to_flight', 'is_weekend']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    return model, train_mae, test_mae

# Function to predict prices
def predict_prices(model, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'departure': date_range})
    future_df = engineer_features(future_df)
    
    X_future = future_df[['day_of_week', 'month', 'days_to_flight', 'is_weekend']]
    future_df['predicted_price'] = model.predict(X_future)
    
    return future_df

# Function to plot prices
def plot_prices(df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df['departure'], df['predicted_price'], marker='o')
    plt.title(title)
    plt.xlabel('Departure Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True)
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title("Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    # Sidebar for user inputs
    st.sidebar.header("User Input")
    data_file = st.sidebar.file_uploader("Upload your flight data CSV", type="csv")
    target_date = st.sidebar.date_input("Target Flight Date", value=datetime(2025, 9, 10))
    
    if data_file is not None:
        df = load_and_preprocess_data(data_file)
        
        if df is not None and not df.empty:
            st.write("Data loaded successfully!")
            st.write(f"Number of records: {len(df)}")
            st.write("Sample data:")
            st.write(df.head())
            
            df = engineer_features(df)
            model, train_mae, test_mae = train_model(df)
            
            st.write(f"Model trained. Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}")
            
            # Predict prices for the next year
            start_date = datetime.now().date()
            end_date = target_date + timedelta(days=30)  # Predict up to a month after the target date
            future_prices = predict_prices(model, start_date, end_date)
            
            st.subheader("Predicted Prices")
            plot_prices(future_prices, "Predicted Flight Prices")
            
            # Find best days to buy
            best_days = future_prices.nsmallest(5, 'predicted_price')
            st.subheader("Best Days to Buy Tickets")
            st.write(best_days[['departure', 'predicted_price']])
            
            # Countdown to target date
            days_left = (target_date - datetime.now().date()).days
            st.metric(label=f"Days until {target_date}", value=days_left)
        else:
            st.error("Failed to load or process the data. Please check your CSV file.")
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
