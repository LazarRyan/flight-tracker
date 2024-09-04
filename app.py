import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Amadeus API credentials
API_KEY = 'your_api_key_here'
API_SECRET = 'your_api_secret_here'

def get_access_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        st.error("Failed to obtain access token")
        return None

def get_flight_offers(origin, destination, date):
    access_token = get_access_token()
    if not access_token:
        return None

    url = f"https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date,
        "adults": 1,
        "nonStop": True,
        "currencyCode": "USD"
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API request failed with status code {response.status_code}")
        return None

def process_and_combine_data(api_data, existing_data):
    new_data = []
    for offer in api_data['data']:
        price = float(offer['price']['total'])
        departure_time = offer['itineraries'][0]['segments'][0]['departure']['at']
        arrival_time = offer['itineraries'][0]['segments'][-1]['arrival']['at']
        origin = offer['itineraries'][0]['segments'][0]['departure']['iataCode']
        destination = offer['itineraries'][0]['segments'][-1]['arrival']['iataCode']
        airline = offer['validatingAirlineCodes'][0]
        
        departure_datetime = datetime.fromisoformat(departure_time.replace('Z', '+00:00'))
        arrival_datetime = datetime.fromisoformat(arrival_time.replace('Z', '+00:00'))
        duration = (arrival_datetime - departure_datetime).total_seconds() / 3600
        
        new_data.append({
            'Origin': origin,
            'Destination': destination,
            'Airline': airline,
            'Price': price,
            'Duration': duration,
            'Departure': departure_datetime,
            'DayOfWeek': departure_datetime.weekday(),
            'Month': departure_datetime.month
        })
    
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    return combined_df

def load_and_preprocess_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Departure'] = pd.to_datetime(df['Departure'])
        return df
    else:
        return pd.DataFrame(columns=['Origin', 'Destination', 'Airline', 'Price', 'Duration', 'Departure', 'DayOfWeek', 'Month'])

def should_call_api(origin, destination):
    cache_file = "api_calls.json"
    today = datetime.now().date().isoformat()
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            api_calls = json.load(f)
    else:
        api_calls = {}
    
    if today not in api_calls:
        api_calls[today] = []
    
    route = f"{origin}-{destination}"
    if route not in api_calls[today]:
        return True
    return False

def update_api_call_record(origin, destination):
    cache_file = "api_calls.json"
    today = datetime.now().date().isoformat()
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            api_calls = json.load(f)
    else:
        api_calls = {}
    
    if today not in api_calls:
        api_calls[today] = []
    
    route = f"{origin}-{destination}"
    api_calls[today].append(route)
    
    with open(cache_file, "w") as f:
        json.dump(api_calls, f)

def train_model(data):
    le = LabelEncoder()
    data['Origin_encoded'] = le.fit_transform(data['Origin'])
    data['Destination_encoded'] = le.fit_transform(data['Destination'])
    data['Airline_encoded'] = le.fit_transform(data['Airline'])

    X = data[['Origin_encoded', 'Destination_encoded', 'Airline_encoded', 'Duration', 'DayOfWeek', 'Month']]
    y = data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, le

def predict_price(model, le, origin, destination, airline, duration, departure_date):
    input_data = pd.DataFrame({
        'Origin_encoded': [le.transform([origin])[0]],
        'Destination_encoded': [le.transform([destination])[0]],
        'Airline_encoded': [le.transform([airline])[0]],
        'Duration': [duration],
        'DayOfWeek': [departure_date.weekday()],
        'Month': [departure_date.month]
    })

    predicted_price = model.predict(input_data)[0]
    return predicted_price

def main():
    st.title("✈️ Flight Price Predictor")

    col1, col2 = st.columns(2)

    with col1:
        origin = st.text_input("Origin (IATA code)", "JFK")
    with col2:
        destination = st.text_input("Destination (IATA code)", "LAX")

    col3, col4 = st.columns(2)

    with col3:
        airline = st.selectbox("Airline", ["AA", "DL", "UA", "B6", "WN"])
    with col4:
        duration = st.number_input("Flight Duration (hours)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)

    target_date = st.date_input("Departure Date", min_value=datetime.now().date() + timedelta(days=1))

    if st.button("🔍 Predict Prices"):
        with st.spinner("Loading data and making predictions..."):
            existing_data = load_and_preprocess_data("flight_prices.csv")
            
            if should_call_api(origin, destination):
                api_data = get_flight_offers(origin, destination, target_date.isoformat())
                if api_data:
                    st.success("✅ Successfully fetched new data from Amadeus API")
                    combined_data = process_and_combine_data(api_data, existing_data)
                    combined_data.to_csv("flight_prices.csv", index=False)
                    st.success("💾 Updated data saved to flight_prices.csv")
                    update_api_call_record(origin, destination)
                else:
                    st.warning("⚠️ No new data fetched from API. Using existing data.")
                    combined_data = existing_data
            else:
                st.info("ℹ️ Using cached data. This route has already been queried today.")
                combined_data = existing_data
            
            model, le = train_model(combined_data)
            
            predicted_price = predict_price(model, le, origin, destination, airline, duration, target_date)
            
            st.subheader("Predicted Price")
            st.markdown(f"<h1 style='text-align: center; color: green;'>${predicted_price:.2f}</h1>", unsafe_allow_html=True)
            
            st.subheader("Historical Price Distribution")
            fig, ax = plt.subplots()
            combined_data['Price'].hist(bins=30, ax=ax)
            ax.set_xlabel("Price (USD)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
