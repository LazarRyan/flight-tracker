import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime, timedelta
from amadeus import Client, ResponseError
from google.cloud import storage
from io import StringIO
from google.oauth2 import service_account
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page config
st.set_page_config(page_title="Flight Price Predictor", layout="wide")

# Initialize clients
@st.cache_resource
def initialize_clients():
    amadeus = Client(
        client_id=st.secrets["AMADEUS_CLIENT_ID"],
        client_secret=st.secrets["AMADEUS_CLIENT_SECRET"]
    )
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(st.secrets["gcs_bucket_name"])
    return amadeus, bucket

amadeus, bucket = initialize_clients()

def get_data_filename(origin, destination):
    return f"flight_prices_{origin}_{destination}.csv"

def load_data_from_gcs(origin, destination):
    filename = get_data_filename(origin, destination)
    blob = bucket.blob(filename)
    df = pd.DataFrame()

    if blob.exists():
        try:
            content = blob.download_as_text()
            df = pd.read_csv(StringIO(content))
            df['departure'] = pd.to_datetime(df['departure'])
            logging.info(f"Loaded {len(df)} records for {origin} to {destination}")
        except Exception as e:
            logging.warning(f"Error loading data for {origin} to {destination}: {str(e)}")

    return df

def save_data_to_gcs(df, origin, destination):
    filename = get_data_filename(origin, destination)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob = bucket.blob(filename)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    logging.info(f"Saved {len(df)} records for {origin} to {destination}")

def fetch_and_process_data(origin, destination, start_date, end_date):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=start_date,
            adults=1
        )
        data = response.data
        if not data:
            st.warning("No flight offers found for the specified route and dates.")
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'departure': offer['itineraries'][0]['segments'][0]['departure']['at'],
            'price': float(offer['price']['total']),
            'origin': origin,
            'destination': destination
        } for offer in data])
        df['departure'] = pd.to_datetime(df['departure'])
        return df
    except ResponseError as error:
        st.error(f"Error fetching data from Amadeus API: {error}")
        logging.error(f"Error fetching data from Amadeus API: {error}")
        if error.response.status_code == 500:
            st.warning("The Amadeus server encountered an internal error. This is not an issue with our application. Please try again later.")
        else:
            st.warning(f"An unexpected error occurred with status code: {error.response.status_code}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching data: {str(e)}")
        logging.error(f"Unexpected error in fetch_and_process_data: {str(e)}")
        return pd.DataFrame()

def engineer_features(df):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['day'] = df['departure'].dt.day
    df['days_until_flight'] = (df['departure'] - datetime.now()).dt.days
    return df

def train_model(df):
    features = ['day_of_week', 'month', 'day', 'days_until_flight']
    X = df[features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    return model, train_mae, test_mae

def predict_prices(model, start_date, end_date, origin, destination, origins, destinations):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_data = pd.DataFrame({'departure': date_range})
    future_data['day_of_week'] = future_data['departure'].dt.dayofweek
    future_data['month'] = future_data['departure'].dt.month
    future_data['day'] = future_data['departure'].dt.day
    future_data['days_until_flight'] = (future_data['departure'] - datetime.now()).dt.days
    future_data['origin'] = origin
    future_data['destination'] = destination
    
    features = ['day_of_week', 'month', 'day', 'days_until_flight']
    future_data['predicted_price'] = model.predict(future_data[features])
    
    return future_data

def plot_prices(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['departure'], y=df['predicted_price'], mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price ($)')
    return fig

def validate_input(origin, destination, outbound_date):
    if not origin or not destination:
        st.error("Please enter both origin and destination airport codes.")
        return False
    if outbound_date <= datetime.now().date():
        st.error("Please select a future date for your outbound flight.")
        return False
    return True

def main():
    st.title("✈️ Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    col1, col2 = st.columns(2)

    with col1:
        origin = st.text_input("🛫 Origin Airport Code", "").upper()
        outbound_date = st.date_input("🗓️ Outbound Flight Date", value=datetime(2025, 9, 10))
    with col2:
        destination = st.text_input("🛬 Destination Airport Code", "").upper()

    if st.button("🔍 Predict Prices"):
        if not validate_input(origin, destination, outbound_date):
            return

        with st.spinner("Loading data and making predictions..."):
            try:
                existing_data = load_data_from_gcs(origin, destination)

                if not existing_data.empty:
                    st.success(f"Using existing data for {origin} to {destination}")
                    st.info(f"Total records: {len(existing_data)}")
                else:
                    st.info(f"No existing data found for {origin} to {destination}. Will fetch new data.")

                st.info("Fetching new data from API...")
                new_data = fetch_and_process_data(origin, destination, outbound_date, outbound_date)
                if not new_data.empty:
                    existing_data = pd.concat([existing_data, new_data], ignore_index=True)
                    existing_data = existing_data.sort_values('departure').drop_duplicates(subset=['departure', 'origin', 'destination'], keep='last')
                    save_data_to_gcs(existing_data, origin, destination)
                    st.success(f"Data updated successfully. Total records: {len(existing_data)}")
                else:
                    st.warning("Unable to fetch new data from API. Proceeding with existing data.")

                if existing_data.empty:
                    st.error("No data available for prediction. Please try again later or with a different route.")
                    return

                st.success(f"Analyzing {len(existing_data)} records for your route.")

                with st.expander("View Sample Data"):
                    st.dataframe(existing_data.head())

                df = engineer_features(existing_data)
                model, train_mae, test_mae = train_model(df)

                logging.info(f"Model trained. Estimated price accuracy: ±${test_mae:.2f} (based on test data)")

                future_prices = predict_prices(model, datetime.now().date(), outbound_date, origin, destination, 
                                               existing_data['origin'].unique(), existing_data['destination'].unique())

                st.subheader("📈 Predicted Prices")
                fig = plot_prices(future_prices, f"Predicted Prices ({origin} to {destination})")
                st.plotly_chart(fig, use_container_width=True)

                best_days = future_prices.nsmallest(5, 'predicted_price')
                st.subheader("💰 Best Days to Book")
                st.table(best_days[['departure', 'predicted_price']].set_index('departure').rename(columns={'predicted_price': 'Predicted Price ($)'}))

                avg_price = future_prices['predicted_price'].mean()
                st.metric(label="💵 Average Predicted Price", value=f"${avg_price:.2f}")

                price_range = future_prices['predicted_price'].max() - future_prices['predicted_price'].min()
                st.metric(label="📊 Price Range", value=f"${price_range:.2f}")

                st.info(f"Predictions shown are for flights from today until {outbound_date}.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logging.error(f"Unexpected error in main function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
