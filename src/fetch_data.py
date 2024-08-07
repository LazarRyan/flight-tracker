import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import streamlit as st
from tqdm import tqdm

# Amadeus API credentials
CLIENT_ID = 'feUUYGgSzyYDRMQqkpXDCyjCa2cd8LCH'
CLIENT_SECRET = 'wSOEoiXhghON9wea'

# Get access token
def get_access_token(client_id, client_secret):
    url = 'https://test.api.amadeus.com/v1/security/oauth2/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    response = requests.post(url, headers=headers, data=data)
    response_json = response.json()
    return response_json['access_token']

# Fetch flight data from Amadeus API
def fetch_flight_data(access_token, origin, destination, departure_date):
    url = 'https://test.api.amadeus.com/v2/shopping/flight-offers'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        'originLocationCode': origin,
        'destinationLocationCode': destination,
        'departureDate': departure_date,
        'adults': 1
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Save data to CSV
def save_to_csv(data, filename):
    if data:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    else:
        print("No data to save.")

# Main function to fetch data and save to CSV
if __name__ == "__main__":
    origin = 'JFK'  # Origin location code
    destination = 'FCO'  # Destination location code
    start_date = datetime(2024, 8, 7)  # Start date for fetching data
    end_date = datetime(2024, 12, 1)  # End date for fetching data

    # Get access token
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)

    all_data = []
    date_range = pd.date_range(start_date, end_date)

    # Fetch data for each date in the range
    for single_date in tqdm(date_range, desc="Fetching flight data"):
        departure_date = single_date.strftime('%Y-%m-%d')
        data = fetch_flight_data(access_token, origin, destination, departure_date)
        if data and 'data' in data:
            for flight in data['data']:
                flight_info = {
                    'Airline': flight['validatingAirlineCodes'][0],
                    'Price': float(flight['price']['total']),
                    'Departure': flight['itineraries'][0]['segments'][0]['departure']['at'],
                    'Arrival': flight['itineraries'][0]['segments'][0]['arrival']['at'],
                    'DepartureDate': departure_date,
                    'DayOfWeek': single_date.weekday(),
                    'Month': single_date.month
                }
                all_data.append(flight_info)
        else:
            print(f"No data found for {departure_date}")

        time.sleep(1)  # Sleep to avoid hitting API rate limits

    # Save data to CSV
    save_to_csv(all_data, 'flight_prices.csv')
    print("Data saved to flight_prices.csv")

