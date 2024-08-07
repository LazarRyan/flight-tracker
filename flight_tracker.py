import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import joblib
import streamlit as st
import time

# Function to fetch data
def fetch_data():
    def get_access_token(client_id, client_secret):
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        payload = f"grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(url, data=payload, headers=headers)
        return response.json().get("access_token")

    def fetch_flight_data(access_token, origin, destination, departure_date):
        url = f"https://test.api.amadeus.com/v2/shopping/flight-offers?originLocationCode={origin}&destinationLocationCode={destination}&departureDate={departure_date}&adults=1&nonStop=false&max=250"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(url, headers=headers)
        return response.json()

    def save_to_csv(data, filename):
        if data and 'data' in data:
            df = pd.json_normalize(data['data'])
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print(f"No data found for {filename}")

    client_id = 'your_client_id'
    client_secret = 'your_client_secret'
    access_token = get_access_token(client_id, client_secret)

    origin = 'EWR'
    destination = 'FCO'
    dates = ['2024-08-07', '2024-09-07', '2024-10-07', '2024-11-07']

    for date in dates:
        data = fetch_flight_data(access_token, origin, destination, date)
        save_to_csv(data, f"flights_{date}.csv")

# Function to load data
def load_data():
    def preprocess_data(df):
        df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
        df['DayOfWeek'] = df['DepartureDate'].dt.dayofweek
        df['Month'] = df['DepartureDate'].dt.month
        df = df[df['Price'] > 0]  # Remove rows with non-positive prices
        return df

    historical_files = [f"flights_{date}.csv" for date in ['2024-08-07', '2024-09-07', '2024-10-07', '2024-11-07']]
    df_list = [pd.read_csv(file) for file in historical_files]
    historical_df = pd.concat(df_list, ignore_index=True)
    historical_df = preprocess_data(historical_df)
    historical_df.to_csv("historical_flights.csv", index=False)

    future_df = pd.DataFrame({
        'DepartureDate': pd.date_range(start='2024-08-08', end='2025-09-05', freq='MS'),
        'DayOfWeek': pd.date_range(start='2024-08-08', end='2025-09-05', freq='MS').dayofweek,
        'Month': pd.date_range(start='2024-08-08', end='2025-09-05', freq='MS').month
    })
    future_df.to_csv("future_flights.csv", index=False)

# Function to train model and predict
def train_predict():
    def train_model(df):
        from sklearn.ensemble import RandomForestRegressor
        X = df[['DayOfWeek', 'Month']]
        y = df['Price']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def preprocess_data(df):
        df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
        df['DayOfWeek'] = df['DepartureDate'].dt.dayofweek
        df['Month'] = df['DepartureDate'].dt.month
        df = df[df['Price'] > 0]  # Remove rows with non-positive prices
        return df

    def predict_future_prices(model, df):
        X_future = df[['DayOfWeek', 'Month']]
        df['PredictedPrice'] = model.predict(X_future)
        return df

    historical_df = pd.read_csv("historical_flights.csv")
    future_df = pd.read_csv("future_flights.csv")

    historical_df = preprocess_data(historical_df)
    future_df = preprocess_data(future_df)

    model = train_model(historical_df)
    future_prices_df = predict_future_prices(model, future_df)

    future_prices_df.to_csv("predicted_flight_prices.csv", index=False)

# Streamlit app to visualize data
def run_app():
    st.title("Flight Price Tracker and Predictor")

    countdown_date = datetime(2025, 9, 5)
    now = datetime.now()
    countdown = (countdown_date - now).days
    st.write(f"Days until September 5, 2025: {countdown}")

    st.header("Predicted Flight Prices")
    df = pd.read_csv("predicted_flight_prices.csv")
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])

    fig, ax = plt.subplots()
    ax.plot(df['DepartureDate'], df['PredictedPrice'], label='Predicted Prices', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('Predicted Flight Prices')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

# Main function to run all steps
def main():
    st.sidebar.title("Flight Price Tracker Steps")
    st.sidebar.write("Running all steps sequentially:")
    
    with st.sidebar:
        with st.spinner("Fetching data..."):
            fetch_data()
            st.success("Data fetched successfully!")
    
    with st.sidebar:
        with st.spinner("Loading data..."):
            load_data()
            st.success("Data loaded successfully!")
    
    with st.sidebar:
        with st.spinner("Training model and predicting prices..."):
            train_predict()
            st.success("Model trained and prices predicted successfully!")
    
    with st.spinner("Running the app..."):
        run_app()
        st.success("App running successfully!")

if __name__ == "__main__":
    main()

