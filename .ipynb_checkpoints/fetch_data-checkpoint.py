import requests
import pandas as pd
import time
from tqdm import tqdm
from datetime import datetime, timedelta

CLIENT_ID = 'tDt4CHIGVt6qkRmKU2gCRNvjyV7AjOI9'
CLIENT_SECRET = 'P0oq4zGh7IkSibbG'
TOKEN_URL = 'https://test.api.amadeus.com/v1/security/oauth2/token'
API_URL = 'https://test.api.amadeus.com/v2/shopping/flight-offers'

# Function to get access token
def get_access_token():
    response = requests.post(TOKEN_URL, data={
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    })
    response_data = response.json()
    access_token = response_data['access_token']
    expires_in = response_data['expires_in']
    expiry_time = time.time() + expires_in
    return access_token, expiry_time

# Function to refresh token if expired
def ensure_valid_token(token_info):
    access_token, expiry_time = token_info
    if time.time() > expiry_time:
        access_token, expiry_time = get_access_token()
    return access_token, expiry_time

# Save data to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    print(df.head())  # Print the first few rows to inspect the structure
    df.to_csv(filename, index=False)

# Fetch flight data
def fetch_flight_data(access_token, origin, destination, departure_date, adults=1):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        'originLocationCode': origin,
        'destinationLocationCode': destination,
        'departureDate': departure_date,
        'adults': adults,
        'nonStop': 'false',
        'max': 50  # Limit to 50 flights per day
    }
    response = requests.get(API_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        for flight in data['data']:
            flight['DepartureDate'] = departure_date
        return data
    else:
        print(f"Error fetching data for {departure_date}: {response.json()}")
        return None

def main():
    origin = 'JFK'
    destination = 'FCO'
    start_date = datetime.today().strftime('%Y-%m-%d')
    end_date = (datetime.today() + timedelta(days=60)).strftime('%Y-%m-%d')  # Collect data for the next 60 days

    # Generate list of dates
    dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d').tolist()

    # Initial token fetch
    token_info = get_access_token()

    all_data = []

    for date in tqdm(dates):
        # Ensure we have a valid token
        access_token, expiry_time = ensure_valid_token(token_info)
        
        # Fetch flight data
        data = fetch_flight_data(access_token, origin, destination, date)
        if data:
            all_data.extend(data['data'])
        else:
            # Handle rate limit error
            if response.status_code == 429:
                print("Rate limit exceeded, waiting for 60 seconds before retrying...")
                time.sleep(60)
                access_token, expiry_time = ensure_valid_token(token_info)
                data = fetch_flight_data(access_token, origin, destination, date)
                if data:
                    all_data.extend(data['data'])

    # Save data to CSV
    save_to_csv(all_data, 'flight_prices.csv')

if __name__ == "__main__":
    main()

