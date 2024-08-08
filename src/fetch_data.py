import requests
import pandas as pd
import datetime
from tqdm import tqdm
import os

# Function to get access token
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

# Function to fetch flight data
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
        "nonStop": 'false',  # Ensuring this is passed as a string 'false'
        "adults": 1
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {date}: {response.json()}")
        return None

# Function to save data to CSV
def save_to_csv(data, filename):
    if data:
        df = pd.DataFrame(data)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)

# Main function
def main():
    CLIENT_ID = 'tDt4CHIGVt6qkRmKU2gCRNvjyV7AjOI9'
    CLIENT_SECRET = 'P0oq4zGh7IkSibbG'
    ORIGIN = "JFK"  # Change as needed
    DESTINATION = "FCO"  # Change as needed

    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    
    start_date = datetime.date.today()
    end_date = start_date + datetime.timedelta(days=60)
    date_range = pd.date_range(start_date, end_date)

    all_data = []

    for date in tqdm(date_range):
        date_str = date.strftime("%Y-%m-%d")
        data = fetch_flight_data(access_token, date_str, ORIGIN, DESTINATION)
        if data and "data" in data:
            for flight in data["data"]:
                flight_info = {
                    "DepartureDate": date_str,
                    "Price": float(flight["price"]["total"]),  # Extracting and converting price to float
                    "Itineraries": flight["itineraries"],
                    "ValidatingAirlineCodes": flight["validatingAirlineCodes"],
                    "TravelerPricings": flight["travelerPricings"]
                }
                all_data.append(flight_info)
    
    save_to_csv(all_data, "flight_prices.csv")

if __name__ == "__main__":
    main()

