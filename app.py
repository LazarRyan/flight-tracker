import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

# Function to load data
def load_data(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
        return df
    return pd.DataFrame(columns=['Date', 'DepartureDate', 'Price', 'Origin', 'Destination'])

# Function to get flight offers with error handling
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
        st.warning(f"An error occurred while fetching data: {error}")
        return []

# Function to extract price from flight offers
def extract_price(offers):
    if offers:
        return min(float(offer['price']['total']) for offer in offers)
    return None

# Function to collect new price data with fallback
def collect_new_data(origin, destination, start_date, end_date, existing_data):
    date_range = pd.date_range(start=start_date, end=end_date)
    today = datetime.now().date()
    new_data = []
    api_error_count = 0
    
    for departure_date in date_range:
        offers = get_flight_offers(origin, destination, departure_date)
        price = extract_price(offers)
        
        if price is None:
            api_error_count += 1
            if api_error_count > 5:  # If more than 5 consecutive errors, use existing data
                st.warning("Too many API errors. Using existing data for predictions.")
                return existing_data
            
            # Try to find a price for this date in existing data
            existing_price = existing_data.loc[
                (existing_data['DepartureDate'] == departure_date) & 
                (existing_data['Origin'] == origin) & 
                (existing_data['Destination'] == destination), 'Price'
            ].mean()
            if not np.isnan(existing_price):
                price = existing_price
            else:
                continue  # Skip this date if no price is available
        else:
            api_error_count = 0  # Reset error count on successful API call
        
        new_data.append({
            'Date': today,
            'DepartureDate': departure_date,
            'Price': price,
            'Origin': origin,
            'Destination': destination
        })
    
    return pd.DataFrame(new_data)

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

# Function to display countdown
def display_countdown(target_date):
    today = datetime.now().date()
    days_left = (target_date - today).days
    st.metric(label=f"Days until {target_date.strftime('%B %d, %Y')}", value=days_left)

# Streamlit app
def main():
    st.title("Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    # User inputs
    origin = st.text_input("Origin Airport Code", "SFO")
    destination = st.text_input("Destination Airport Code", "FCO")  # Rome, Italy
    target_date = st.date_input("Target Flight Date", value=datetime(2025, 9, 10).date())
    
    display_countdown(target_date)
    
    historical_data_path = f"historical_flight_data_{origin}_{destination}.csv"
    
    if st.button("Update Data and Predict"):
        # Load existing data
        existing_data = load_data(historical_data_path)
        
        # Check if we need to update the data
        last_update = existing_data['Date'].max() if not existing_data.empty else datetime.min.date()
        today = datetime.now().date()
        
        if (today - last_update).days >= 30 or existing_data.empty:
            st.write("Collecting new data...")
            new_data = collect_new_data(origin, destination, today, target_date, existing_data)
            
            if not new_data.empty and not new_data.equals(existing_data):
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                updated_data = updated_data.drop_duplicates(subset=['Date', 'DepartureDate', 'Origin', 'Destination'], keep='last')
                updated_data.to_csv(historical_data_path, index=False)
                st.success("Historical data updated and saved.")
            else:
                st.info("No new data collected or API errors occurred. Using existing data.")
                updated_data = existing_data
        else:
            st.info("Using existing historical data (last updated less than 30 days ago).")
            updated_data = existing_data
        
        if not updated_data.empty:
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
        else:
            st.error("No data available for predictions. Please try again later.")

if __name__ == "__main__":
    main()
