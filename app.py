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
    filename = f"flight_prices_{origin}_{destination}.csv"
    blob = bucket.blob(filename)
    df = pd.DataFrame()
    last_modified = None

    if blob.exists():
        try:
            content = blob.download_as_text()
            df = pd.read_csv(StringIO(content))
            df['departure'] = pd.to_datetime(df['departure'])
            last_modified = blob.updated
            logging.info(f"Loaded {len(df)} records for {origin} to {destination}")
        except Exception as e:
            logging.warning(f"Error loading data for {origin} to {destination}: {str(e)}")
    else:
        logging.info(f"No existing data file found for {origin} to {destination}")

    return df, last_modified

def save_data_to_gcs(df, origin, destination):
    filename = get_data_filename(origin, destination)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob = bucket.blob(filename)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    logging.info(f"Saved {len(df)} records for {origin} to {destination}")

def get_last_api_call_time(origin, destination):
    filename = f"last_api_call_{origin}_{destination}.txt"
    blob = bucket.blob(filename)
    if blob.exists():
        return datetime.fromisoformat(blob.download_as_text().strip())
    return None

def update_api_call_time(origin, destination):
    filename = f"last_api_call_{origin}_{destination}.txt"
    blob = bucket.blob(filename)
    blob.upload_from_string(datetime.now().isoformat())

def should_call_api(origin, destination):
    last_call_time = get_last_api_call_time(origin, destination)
    if last_call_time is None:
        return True
    time_since_last_call = datetime.now() - last_call_time
    return time_since_last_call.total_seconds() >= 12 * 3600  # 12 hours in seconds

def get_flight_offers(origin, destination, departure_date):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date.strftime("%Y-%m-%d"),
            adults=1,
            max=5
        )
        return response.data
    except ResponseError as error:
        logging.error(f"Error fetching flight data: {error}")
        return []

def fetch_and_process_data(origin, destination, start_date, end_date):
    all_data = []
    current_date = start_date
    progress_bar = st.progress(0)
    total_days = (end_date - start_date).days

    for i in range(total_days):
        offers = get_flight_offers(origin, destination, current_date)
        for offer in offers:
            outbound = offer['itineraries'][0]
            price = float(offer['price']['total'])
            departure = outbound['segments'][0]['departure']['at']
            all_data.append({
                'departure': departure,
                'price': price,
                'origin': origin,
                'destination': destination
            })
        current_date += timedelta(days=1)
        progress_bar.progress((i + 1) / total_days)

    new_df = pd.DataFrame(all_data)
    new_df['departure'] = pd.to_datetime(new_df['departure'])
    return new_df

def engineer_features(df):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['days_to_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df = pd.get_dummies(df, columns=['origin', 'destination'], prefix=['origin', 'dest'])
    return df

def train_model(df):
    X = df.drop(['departure', 'price'], axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    return model, train_mae, test_mae

def predict_prices(model, start_date, end_date, origin, destination, all_origins, all_destinations):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'departure': date_range})
    future_df['origin'] = origin
    future_df['destination'] = destination
    future_df = engineer_features(future_df)
    
    for o in all_origins:
        if f'origin_{o}' not in future_df.columns:
            future_df[f'origin_{o}'] = 0
    for d in all_destinations:
        if f'dest_{d}' not in future_df.columns:
            future_df[f'dest_{d}'] = 0
    
    X_future = future_df.drop(['departure'], axis=1)
    future_df['predicted_price'] = model.predict(X_future)

    return future_df

def plot_prices(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['departure'], y=df['predicted_price'],
                             mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title=title, xaxis_title='Departure Date',
                      yaxis_title='Predicted Price (USD)')
    return fig

def validate_input(origin, destination, outbound_date):
    if len(origin) != 3 or len(destination) != 3:
        st.error("Origin and destination must be 3-letter IATA airport codes.")
        return False
    if outbound_date <= datetime.now().date():
        st.error("Outbound date must be in the future.")
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
                existing_data, last_modified = load_data_from_gcs(origin, destination)

                if not existing_data.empty and last_modified is not None:
                    most_recent_data = existing_data['departure'].max()
                    oldest_data = existing_data['departure'].min()
                    
                    st.success(f"Using existing data (last updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')})")
                    st.info(f"Data range: {oldest_data.date()} to {most_recent_data.date()}")
                    st.info(f"Total records: {len(existing_data)}")
                else:
                    st.info(f"No existing data found for {origin} to {destination}. Will fetch new data.")

                api_calls_made = 0
                while should_call_api(origin, destination) and api_calls_made < 2:
                    st.info(f"Fetching new data from API (Call {api_calls_made + 1}/2)...")
                    new_data = fetch_and_process_data(origin, destination, datetime.now().date(), outbound_date)
                    if not new_data.empty:
                        existing_data = pd.concat([existing_data, new_data], ignore_index=True)
                        existing_data = existing_data.sort_values('departure').drop_duplicates(subset=['departure', 'origin', 'destination'], keep='last')
                        save_data_to_gcs(existing_data, origin, destination)
                        update_api_call_time(origin, destination)
                        api_calls_made += 1
                        st.success(f"Data updated successfully. Total records: {len(existing_data)}")
                    else:
                        st.warning(f"No new data fetched from API on call {api_calls_made + 1}.")
                        break

                if existing_data.empty:
                    st.error("No data available for prediction. Please try again with a different route or check your data source.")
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
