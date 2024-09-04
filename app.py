import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from amadeus import Client, ResponseError

# Amadeus API configuration
AMADEUS_CLIENT_ID = st.secrets["AMADEUS_CLIENT_ID"]
AMADEUS_CLIENT_SECRET = st.secrets["AMADEUS_CLIENT_SECRET"]

amadeus = Client(
    client_id=AMADEUS_CLIENT_ID,
    client_secret=AMADEUS_CLIENT_SECRET
)

def load_and_preprocess_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        # Read CSV with header
        df = pd.read_csv(filepath, parse_dates=['DepartureDate'])
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'DepartureDate': 'departure',
            'Price': 'price',
            'Itineraries': 'itineraries',
            'ValidatingAirlineCodes': 'carriers',
            'TravelerPricings': 'price_details'
        })
        
        # Convert price to float, replacing any non-numeric values with NaN
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        def extract_price(price_data):
            try:
                price_dict = json.loads(price_data.replace("'", "\""))
                return float(price_dict[0]['price']['total'])
            except:
                return np.nan
        
        def extract_departure(itinerary_data):
            try:
                itinerary_dict = json.loads(itinerary_data.replace("'", "\""))
                return itinerary_dict[0]['segments'][0]['departure']['at']
            except:
                return np.nan
        
        # Only process these columns if the price column is empty or has issues
        if df['price'].isnull().all() or df['price'].max() == 'Price':
            df['price'] = df['price_details'].apply(extract_price)
            df['departure'] = df['itineraries'].apply(extract_departure)
        
        df['departure'] = pd.to_datetime(df['departure'])
        
        df = df.dropna(subset=['price', 'departure'])
        df = df[(df['price'] > 0) & (df['departure'] > '2023-01-01')]
        
        return df[['departure', 'price']]
    except Exception as e:
        st.error(f"Error loading data from {filepath}: {str(e)}")
        return pd.DataFrame()

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
        st.error(f"Error fetching data from Amadeus API: {error}")
        return []

def process_and_combine_data(api_data, existing_data):
    new_data = []
    for offer in api_data:
        price = float(offer['price']['total'])
        departure = offer['itineraries'][0]['segments'][0]['departure']['at']
        new_data.append({'departure': departure, 'price': price})
    
    new_df = pd.DataFrame(new_data)
    new_df['departure'] = pd.to_datetime(new_df['departure'])
    
    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['departure'], keep='last')
    combined_df = combined_df.sort_values('departure')
    
    return combined_df

def engineer_features(df):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['days_to_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

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

def predict_prices(model, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'departure': date_range})
    future_df = engineer_features(future_df)
    
    X_future = future_df[['day_of_week', 'month', 'days_to_flight', 'is_weekend']]
    future_df['predicted_price'] = model.predict(X_future)
    
    return future_df

def plot_prices(df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df['departure'], df['predicted_price'], marker='o')
    plt.title(title)
    plt.xlabel('Departure Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True)
    st.pyplot(plt)

def main():
    st.title("Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    st.sidebar.header("User Input")
    origin = st.sidebar.text_input("Origin Airport Code", "JFK")
    destination = st.sidebar.text_input("Destination Airport Code", "FCO")
    target_date = st.sidebar.date_input("Target Flight Date", value=datetime(2025, 9, 10))
    
    existing_data = load_and_preprocess_data("flight_prices.csv")
    
    if existing_data.empty:
        st.warning("No existing data found. Attempting to fetch data from API.")
    else:
        st.success(f"Loaded {len(existing_data)} records from existing data.")
    
    api_data = get_flight_offers(origin, destination, target_date)
    
    if api_data:
        st.success("Successfully fetched new data from Amadeus API")
        combined_data = process_and_combine_data(api_data, existing_data)
        combined_data.to_csv("flight_prices.csv", index=False)
        st.success("Updated data saved to flight_prices.csv")
    else:
        st.warning("No new data fetched from API. Using existing data.")
        combined_data = existing_data
    
    if not combined_data.empty:
        st.write(f"Total records for analysis: {len(combined_data)}")
        st.write("Sample data:")
        st.write(combined_data.head())
        
        df = engineer_features(combined_data)
        model, train_mae, test_mae = train_model(df)
        
        st.write(f"Model trained. Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}")
        
        start_date = datetime.now().date()
        end_date = target_date + timedelta(days=30)
        future_prices = predict_prices(model, start_date, end_date)
        
        st.subheader("Predicted Prices")
        plot_prices(future_prices, "Predicted Flight Prices")
        
        best_days = future_prices.nsmallest(5, 'predicted_price')
        st.subheader("Best Days to Buy Tickets")
        st.write(best_days[['departure', 'predicted_price']])
        
        days_left = (target_date - datetime.now().date()).days
        st.metric(label=f"Days until {target_date}", value=days_left)
    else:
        st.error("No data available for prediction. Please check your data source or try again later.")

if __name__ == "__main__":
    main()
