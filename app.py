import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
from amadeus import Client, ResponseError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Amadeus API configuration
AMADEUS_CLIENT_ID = st.secrets["AMADEUS_CLIENT_ID"]
AMADEUS_CLIENT_SECRET = st.secrets["AMADEUS_CLIENT_SECRET"]

amadeus = Client(
    client_id=AMADEUS_CLIENT_ID,
    client_secret=AMADEUS_CLIENT_SECRET
)

def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def safe_json_loads(x):
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse JSON: {x}")
        return None

def safe_eval(x):
    if not isinstance(x, str):
        return x
    try:
        return eval(x)
    except:
        logging.warning(f"Failed to eval: {x}")
        return None

def load_data(filepath):
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, header=None, names=['date', 'price', 'itineraries', 'carriers', 'price_details'])
            
            # Convert date string to datetime more explicitly
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            
            # Safely convert price to float
            df['price'] = df['price'].apply(safe_float)
            
            # Safely parse JSON and list strings
            df['itineraries'] = df['itineraries'].apply(safe_json_loads)
            df['carriers'] = df['carriers'].apply(safe_eval)
            df['price_details'] = df['price_details'].apply(safe_json_loads)
            
            # Extract departure datetime from itineraries
            def extract_departure(itinerary):
                if not isinstance(itinerary, list) or not itinerary:
                    return pd.NaT
                try:
                    return datetime.fromisoformat(itinerary[0]['segments'][0]['departure']['at'])
                except (IndexError, KeyError, ValueError, TypeError):
                    logging.warning(f"Failed to extract departure from: {itinerary}")
                    return pd.NaT

            df['departure'] = df['itineraries'].apply(extract_departure)
            
            # Remove rows with NaN prices or departures
            df = df.dropna(subset=['price', 'departure'])
            
            logging.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logging.exception(f"Error loading data from {filepath}: {str(e)}")
            return pd.DataFrame(columns=['date', 'price', 'itineraries', 'carriers', 'price_details', 'departure'])
    else:
        logging.warning(f"File not found: {filepath}")
        return pd.DataFrame(columns=['date', 'price', 'itineraries', 'carriers', 'price_details', 'departure'])

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
        logging.warning(f"An error occurred while fetching data: {error}")
        return []

def extract_price(offers):
    if offers:
        return min(float(offer['price']['total']) for offer in offers)
    return None

def collect_new_data(origin, destination, start_date, end_date, existing_data):
    date_range = pd.date_range(start=start_date, end=end_date)
    new_data = []
    api_call_count = 0
    
    for departure_date in date_range:
        if api_call_count >= 2:
            logging.info("API call limit reached. Using existing data for remaining predictions.")
            break
        
        # Check if we already have data for this date
        if not existing_data.empty and departure_date.date() in existing_data['date'].dt.date.values:
            continue
        
        offers = get_flight_offers(origin, destination, departure_date)
        api_call_count += 1
        
        if offers:
            price = extract_price(offers)
            new_data.append({
                'date': departure_date.date(),
                'price': price,
                'itineraries': json.dumps(offers[0]['itineraries']),
                'carriers': str(offers[0]['validatingAirlineCodes']),
                'price_details': json.dumps(offers[0]['travelerPricings']),
                'departure': datetime.fromisoformat(offers[0]['itineraries'][0]['segments'][0]['departure']['at'])
            })
    
    new_df = pd.DataFrame(new_data)
    return pd.concat([existing_data, new_df], ignore_index=True).sort_values('date')

def preprocess_data(df):
    df['DayOfWeek'] = df['departure'].dt.dayofweek
    df['Month'] = df['departure'].dt.month
    df['DaysToFlight'] = (df['departure'] - datetime.now()).dt.days
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsHoliday'] = ((df['Month'] == 12) & (df['departure'].dt.day >= 20)).astype(int)
    return df

def train_model(df):
    X = df[['DayOfWeek', 'Month', 'DaysToFlight', 'IsWeekend', 'IsHoliday']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    logging.info(f"Train MAE: ${train_mae:.2f}")
    logging.info(f"Test MAE: ${test_mae:.2f}")
    
    return model

def predict_future_prices(model, future_df):
    X_future = future_df[['DayOfWeek', 'Month', 'DaysToFlight', 'IsWeekend', 'IsHoliday']]
    future_df['PredictedPrice'] = model.predict(X_future)
    return future_df

def find_best_days_to_buy(future_df, n=5):
    return future_df.nsmallest(n, 'PredictedPrice')

def plot_price_predictions(future_df, best_days):
    plt.figure(figsize=(12, 6))
    plt.plot(future_df['departure'], future_df['PredictedPrice'], marker='', linewidth=2)
    plt.scatter(best_days['departure'], best_days['PredictedPrice'], color='red', s=50, zorder=5)
    plt.title('Predicted Flight Prices and Best Days to Buy')
    plt.xlabel('Departure Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    for _, row in best_days.iterrows():
        plt.annotate(f"${row['PredictedPrice']:.2f}\n{row['departure'].strftime('%Y-%m-%d')}", 
                     (row['departure'], row['PredictedPrice']),
                     textcoords="offset points", xytext=(0,10), ha='center')
    st.pyplot(plt)

def display_countdown(target_date):
    today = datetime.now().date()
    days_left = (target_date - today).days
    st.metric(label=f"Days until {target_date.strftime('%B %d, %Y')}", value=days_left)

def main():
    st.title("Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    # User inputs
    origin = st.text_input("Origin Airport Code", "JFK")
    destination = st.text_input("Destination Airport Code", "FCO")  # Rome, Italy
    target_date = st.date_input("Target Flight Date", value=datetime(2025, 9, 10).date())
    
    display_countdown(target_date)
    
    historical_data_path = "flight_prices.csv"
    
    if st.button("Update Data and Predict"):
        try:
            # Load existing data
            existing_data = load_data(historical_data_path)
            
            if existing_data.empty:
                st.warning(f"No existing data found in {historical_data_path}. Will attempt to collect new data.")
            
            # Collect new data (limited to 2 API calls)
            st.write("Collecting new data...")
            updated_data = collect_new_data(origin, destination, datetime.now().date(), target_date, existing_data)
            
            # Save updated data only if new data was added
            if len(updated_data) > len(existing_data):
                updated_data.to_csv(historical_data_path, index=False, header=False)
                st.success(f"New data added and saved to {historical_data_path}.")
            else:
                st.info("No new data added. Using existing historical data.")
            
            if not updated_data.empty and not updated_data['price'].isnull().all():
                # Preprocess and train model
                st.write("Training model...")
                df = preprocess_data(updated_data)
                model = train_model(df)
                
                # Predict future prices
                st.write("Predicting future prices...")
                future_dates = pd.date_range(start=datetime.now(), end=target_date)
                future_df = pd.DataFrame({'departure': future_dates})
                future_df = preprocess_data(future_df)
                future_df = predict_future_prices(model, future_df)
                best_days = find_best_days_to_buy(future_df)
                
                # Visualize results
                st.subheader("Price Prediction Chart")
                plot_price_predictions(future_df, best_days)
                
                st.subheader("Best Days to Buy Tickets")
                st.dataframe(best_days[['departure', 'PredictedPrice']].set_index('departure'))
                
                st.subheader("All Predicted Prices")
                st.dataframe(future_df[['departure', 'PredictedPrice']].set_index('departure'))
            else:
                st.error("No valid price data available for predictions. Please check your data source.")
        except Exception as e:
            logging.exception("An error occurred:")
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check the logs for more details.")

if __name__ == "__main__":
    main()
