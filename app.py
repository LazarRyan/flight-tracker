import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from amadeus import Client, ResponseError

# Amadeus API configuration (you'll need to set these up in your Streamlit secrets)
AMADEUS_CLIENT_ID = st.secrets["AMADEUS_CLIENT_ID"]
AMADEUS_CLIENT_SECRET = st.secrets["AMADEUS_CLIENT_SECRET"]

amadeus = Client(
    client_id=AMADEUS_CLIENT_ID,
    client_secret=AMADEUS_CLIENT_SECRET
)

# Function to load existing data
def load_data(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame(columns=['Date', 'DepartureDate', 'Price'])

# Function to get flight offers
def get_flight_offers(origin, destination, departure_date):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date.strftime("%Y-%m-%d"),
            adults=1
        )
        return response.data
    except ResponseError as error:
        st.error(f"An error occurred: {error}")
        return []

# Function to extract price from flight offers
def extract_price(offers):
    if offers:
        return min(float(offer['price']['total']) for offer in offers)
    return None

# Function to collect new price data
def collect_new_data(origin, destination, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    today = datetime.now().date()
    new_data = []
    for departure_date in date_range:
        offers = get_flight_offers(origin, destination, departure_date)
        price = extract_price(offers)
        if price:
            new_data.append({'Date': today, 'DepartureDate': departure_date, 'Price': price})
    return pd.DataFrame(new_data)

# Function to update historical data
def update_historical_data(existing_data, new_data):
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data = updated_data.drop_duplicates(subset=['Date', 'DepartureDate'], keep='last')
    return updated_data

# Function to preprocess data
def preprocess_data(df):
    df['DayOfWeek'] = df['DepartureDate'].dt.dayofweek
    df['Month'] = df['DepartureDate'].dt.month
    df['DaysToFlight'] = (df['DepartureDate'] - df['Date']).dt.days
    return df

# Function to train model
def train_model(df):
    X = df[['DayOfWeek', 'Month', 'DaysToFlight']]
    y = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Function to predict future prices
def predict_future_prices(model, future_df):
    X_future = future_df[['DayOfWeek', 'Month', 'DaysToFlight']]
    future_df['PredictedPrice'] = model.predict(X_future)
    return future_df

# Function to find best days to buy
def find_best_days_to_buy(future_df, n=5):
    return future_df.nsmallest(n, 'PredictedPrice')

# Function to plot price predictions
def plot_price_predictions(future_df, best_days):
    plt.figure(figsize=(12, 6))
    plt.plot(future_df['DepartureDate'], future_df['PredictedPrice'], marker='', linewidth=2)
    plt.scatter(best_days['DepartureDate'], best_days['PredictedPrice'], color='red', s=50, zorder=5)
    plt.title('Predicted Flight Prices and Best Days to Buy')
    plt.xlabel('Departure Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    for _, row in best_days.iterrows():
        plt.annotate(f"${row['PredictedPrice']:.2f}\n{row['DepartureDate'].strftime('%Y-%m-%d')}", 
                     (row['DepartureDate'], row['PredictedPrice']),
                     textcoords="offset points", xytext=(0,10), ha='center')
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Flight Price Predictor")

    # User inputs
    origin = st.text_input("Origin Airport Code", "SFO")
    destination = st.text_input("Destination Airport Code", "JFK")
    target_date = st.date_input("Target Flight Date", value=datetime.now().date() + timedelta(days=90))
    
    historical_data_path = "historical_flight_data.csv"
    
    if st.button("Update Data and Predict"):
        # Load existing data
        existing_data = load_data(historical_data_path)
        
        # Collect new data (simulate monthly collection)
        today = datetime.now().date()
        last_collection = existing_data['Date'].max().date() if not existing_data.empty else today - timedelta(days=30)
        if (today - last_collection).days >= 30:
            st.write("Collecting new data...")
            new_data = collect_new_data(origin, destination, today, target_date)
            
            # Update historical data
            updated_data = update_historical_data(existing_data, new_data)
            updated_data.to_csv(historical_data_path, index=False)
            st.success("Historical data updated and saved.")
        else:
            updated_data = existing_data
            st.info("Using existing historical data (last updated less than 30 days ago).")
        
        # Preprocess and train model
        st.write("Training model...")
        df = preprocess_data(updated_data)
        model = train_model(df)
        
        # Predict future prices
        st.write("Predicting future prices...")
        future_dates = pd.date_range(start=today, end=target_date)
        future_df = pd.DataFrame({'Date': [today]*len(future_dates), 'DepartureDate': future_dates})
        future_df = preprocess_data(future_df)
        future_df = predict_future_prices(model, future_df)
        best_days = find_best_days_to_buy(future_df)
        
        # Visualize results
        st.subheader("Price Prediction Chart")
        plot_price_predictions(future_df, best_days)
        
        st.subheader("Best Days to Buy Tickets")
        st.dataframe(best_days[['DepartureDate', 'PredictedPrice']].set_index('DepartureDate'))
        
        st.subheader("All Predicted Prices")
        st.dataframe(future_df[['DepartureDate', 'PredictedPrice']].set_index('DepartureDate'))

if __name__ == "__main__":
    main()
