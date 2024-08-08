import requests
import pandas as pd
from datetime import datetime, timedelta, date
from tqdm import tqdm
import os
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Fetch Data
def get_access_token(client_id, client_secret):
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, headers=headers, data=data)
    response_json = response.json()
    return response_json["access_token"]

def fetch_flight_data(access_token, date, origin, destination, max_results=50):
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date,
        "max": max_results,
        "currencyCode": "USD",
        "nonStop": 'false',
        "adults": 1
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {date}: {response.json()}")
        return None

def save_to_csv(data, filename):
    if data:
        df = pd.DataFrame(data)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)

def fetch_data():
    CLIENT_ID = 'tDt4CHIGVt6qkRmKU2gCRNvjyV7AjOI9'
    CLIENT_SECRET = 'P0oq4zGh7IkSibbG'
    ORIGIN = "JFK"
    DESTINATION = "FCO"

    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)

    start_date = date.today()
    end_date = start_date + timedelta(days=30)

    all_data = []

    for single_date in tqdm(pd.date_range(start_date, end_date)):
        date_str = single_date.strftime("%Y-%m-%d")
        data = fetch_flight_data(access_token, date_str, ORIGIN, DESTINATION)
        if data:
            for offer in data["data"]:
                offer_data = {
                    "Date": date_str,
                    "Price": offer["price"]["total"],
                    "Origin": ORIGIN,
                    "Destination": DESTINATION
                }
                all_data.append(offer_data)

    # Ensure that the dataframe has the correct columns
    df = pd.DataFrame(all_data, columns=["Date", "Price", "Origin", "Destination"])
    df.to_csv("flight_prices.csv", index=False)

# Load Data
def load_data():
    df = pd.read_csv('flight_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.to_csv('flight_prices_clean.csv', index=False)

# Clean Data
def clean_data():
    df = pd.read_csv('flight_prices_clean.csv')
    df = df.drop_duplicates()
    df.to_csv('flight_prices_clean.csv', index=False)

# Train and Predict
def train_model():
    df = pd.read_csv('flight_prices_clean.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['DaysFromNow'] = (df['Date'] - datetime.now()).dt.days
    X = df[['DaysFromNow']]
    y = df['Price']
    
    model = RandomForestRegressor()
    model.fit(X, y)
    
    joblib.dump(model, 'flight_price_model.pkl')

    future_dates = pd.date_range(start='2024-12-02', end='2025-09-05')
    future_df = pd.DataFrame(future_dates, columns=['Date'])
    future_df['DaysFromNow'] = (future_df['Date'] - datetime.now()).dt.days
    future_df['PredictedPrice'] = model.predict(future_df[['DaysFromNow']])
    future_df.to_csv('future_prices.csv', index=False)

# Streamlit App
def load_data_app(csv_file):
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_model_app(model_file):
    return joblib.load(model_file)

def main():
    st.title("Flight Price Predictor")

    with st.spinner("Fetching and processing data..."):
        fetch_data()
        load_data()
        clean_data()
        train_model()

    historical_csv = 'flight_prices_clean.csv'
    future_csv = 'future_prices.csv'
    model_file = 'flight_price_model.pkl'

    historical_df = load_data_app(historical_csv)
    future_df = load_data_app(future_csv)
    model = load_model_app(model_file)

    st.write("### Historical Flight Prices")
    st.dataframe(historical_df)

    st.write("### Future Flight Price Predictions")
    st.dataframe(future_df)

    fig, ax = plt.subplots()
    ax.plot(historical_df['Date'], historical_df['Price'], label='Historical Prices')
    ax.plot(future_df['Date'], future_df['PredictedPrice'], label='Predicted Prices', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    st.pyplot(fig)

    target_date = datetime(2025, 9, 10)
    current_date = datetime.now()
    countdown = (target_date - current_date).days
    st.write(f"### Countdown to September 10, 2025: {countdown} days")

if __name__ == "__main__":
    main()

