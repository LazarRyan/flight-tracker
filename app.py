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

# Custom CSS for improved visual security
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

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

def get_data_filename(origin, destination, trip_type):
    return f"flight_prices_{origin}_{destination}_{trip_type}.csv"

def get_api_call_filename(origin, destination):
    return f"api_calls_{origin}_{destination}.txt"

def load_data_from_gcs(origin, destination, trip_type):
    filename = get_data_filename(origin, destination, trip_type)
    blob = bucket.blob(filename)
    df = pd.DataFrame()

    try:
        content = blob.download_as_text()
        df = pd.read_csv(StringIO(content))
        df['departure'] = pd.to_datetime(df['departure'])
        if trip_type == "round-trip" and 'return' in df.columns:
            df['return'] = pd.to_datetime(df['return'])
        logging.info(f"Loaded {len(df)} records for {origin} to {destination} ({trip_type})")
    except Exception as e:
        logging.warning(f"No specific data file found for {origin} to {destination} ({trip_type}). Error: {str(e)}")
        
        # Try to load from the general flight_prices.csv
        general_blob = bucket.blob("flight_prices.csv")
        try:
            content = general_blob.download_as_text()
            general_df = pd.read_csv(StringIO(content))
            general_df['departure'] = pd.to_datetime(general_df['departure'])
            
            if 'origin' in general_df.columns and 'destination' in general_df.columns:
                df = general_df[(general_df['origin'] == origin) & (general_df['destination'] == destination)]
                
                if trip_type == "round-trip" and 'return' in general_df.columns:
                    df = df[df['return'].notna()]
                    df['return'] = pd.to_datetime(df['return'])
                elif trip_type == "one-way" and 'return' in general_df.columns:
                    df = df[df['return'].isna()]
                
                logging.info(f"Loaded {len(df)} records from general data for {origin} to {destination} ({trip_type})")
            else:
                logging.warning("Origin and destination data not available in the general dataset.")
        except Exception as e:
            logging.error(f"Failed to load general flight data. Error: {str(e)}")

    if df.empty:
        logging.warning(f"No existing data found for {origin} to {destination} ({trip_type})")
    
    return df

def save_data_to_gcs(new_df, origin, destination, trip_type):
    filename = get_data_filename(origin, destination, trip_type)
    existing_df = load_data_from_gcs(origin, destination, trip_type)
    
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    if trip_type == "round-trip":
        combined_df = combined_df.sort_values('departure').drop_duplicates(subset=['departure', 'return', 'origin', 'destination'], keep='last')
    else:
        combined_df = combined_df.sort_values('departure').drop_duplicates(subset=['departure', 'origin', 'destination'], keep='last')
    
    csv_buffer = StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    blob = bucket.blob(filename)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    
    logging.info(f"Saved {len(combined_df)} records for {origin} to {destination} ({trip_type})")

def should_call_api(origin, destination):
    filename = get_api_call_filename(origin, destination)
    blob = bucket.blob(filename)
    if blob.exists():
        content = blob.download_as_text().strip().split('\n')
        current_date = datetime.now().date().isoformat()
        calls_today = sum(1 for call in content if call.startswith(current_date))
        return calls_today < 2
    return True

def update_api_call_time(origin, destination):
    filename = get_api_call_filename(origin, destination)
    blob = bucket.blob(filename)
    current_date = datetime.now().date().isoformat()
    if blob.exists():
        content = blob.download_as_text().strip().split('\n')
        content = [line for line in content if line.startswith(current_date)]
    else:
        content = []
    content.append(f"{current_date}_{datetime.now().isoformat()}")
    blob.upload_from_string('\n'.join(content))

@st.cache_data
def get_flight_offers(origin, destination, departure_date, return_date=None):
    try:
        if return_date:
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin,
                destinationLocationCode=destination,
                departureDate=departure_date.strftime("%Y-%m-%d"),
                returnDate=return_date.strftime("%Y-%m-%d"),
                adults=1,
                max=5
            )
        else:
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

def fetch_data_for_months(origin, destination, start_date, end_date, trip_type):
    all_data = []
    current_date = start_date
    progress_bar = st.progress(0)
    total_days = (end_date - start_date).days

    for i in range(total_days):
        if trip_type == "round-trip":
            return_date = current_date + timedelta(days=7)  # Assume 7-day trips for round-trip
            offers = get_flight_offers(origin, destination, current_date, return_date)
        else:
            offers = get_flight_offers(origin, destination, current_date)
        all_data.extend(offers)
        current_date += timedelta(days=1)
        progress_bar.progress((i + 1) / total_days)

    return all_data

def process_and_combine_data(api_data, existing_data, origin, destination, trip_type):
    new_data = []
    for offer in api_data:
        outbound = offer['itineraries'][0]
        if trip_type == "round-trip":
            inbound = offer['itineraries'][1]
            new_data.append({
                'departure': outbound['segments'][0]['departure']['at'],
                'return': inbound['segments'][0]['departure']['at'],
                'price': float(offer['price']['total']),
                'origin': origin,
                'destination': destination
            })
        else:
            new_data.append({
                'departure': outbound['segments'][0]['departure']['at'],
                'price': float(offer['price']['total']),
                'origin': origin,
                'destination': destination
            })
    
    new_df = pd.DataFrame(new_data)
    new_df['departure'] = pd.to_datetime(new_df['departure'])
    if trip_type == "round-trip":
        new_df['return'] = pd.to_datetime(new_df['return'])

    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    if trip_type == "round-trip":
        combined_df = combined_df.sort_values('departure').drop_duplicates(subset=['departure', 'return', 'origin', 'destination'], keep='last')
    else:
        combined_df = combined_df.sort_values('departure').drop_duplicates(subset=['departure', 'origin', 'destination'], keep='last')
    
    cutoff_date = datetime.now() - timedelta(days=365)
    combined_df = combined_df[
        (combined_df['departure'] >= cutoff_date) |
        (combined_df['departure'] >= datetime.now())
    ]
    return combined_df.sort_values('departure')

def engineer_features(df, trip_type):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['days_to_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    if trip_type == "round-trip":
        df['trip_duration'] = (df['return'] - df['departure']).dt.days
        df['return_day_of_week'] = df['return'].dt.dayofweek
        df['return_month'] = df['return'].dt.month
    df = pd.get_dummies(df, columns=['origin', 'destination'], prefix=['origin', 'dest'])
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(['departure', 'price'] + (['return'] if 'return' in df.columns else []), axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    return model, train_mae, test_mae

def predict_prices(model, start_date, end_date, origin, destination, all_origins, all_destinations, trip_type):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'departure': date_range})
    if trip_type == "round-trip":
        future_df['return'] = future_df['departure'] + timedelta(days=7)  # Assume 7-day trips
    future_df['origin'] = origin
    future_df['destination'] = destination
    future_df = engineer_features(future_df, trip_type)
    
    for o in all_origins:
        if f'origin_{o}' not in future_df.columns:
            future_df[f'origin_{o}'] = 0
    for d in all_destinations:
        if f'dest_{d}' not in future_df.columns:
            future_df[f'dest_{d}'] = 0
    
    X_future = future_df.drop(['departure'] + (['return'] if trip_type == "round-trip" else []), axis=1)
    future_df['predicted_price'] = model.predict(X_future)

    return future_df

def plot_prices(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['departure'], y=df['predicted_price'],
                             mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title=title, xaxis_title='Departure Date',
                      yaxis_title='Predicted Price (USD)')
    return fig

def validate_input(origin, destination, outbound_date, return_date=None):
    if len(origin) != 3 or len(destination) != 3:
        st.error("Origin and destination must be 3-letter IATA airport codes.")
        return False
    if outbound_date <= datetime.now().date():
        st.error("Outbound date must be in the future.")
        return False
    if return_date and return_date <= outbound_date:
        st.error("Return date must be after the outbound date.")
        return False
    return True

def main():
    st.title("✈️ Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    trip_type = st.radio("Trip Type", ["one-way", "round-trip"])

    col1, col2 = st.columns(2)

    with col1:
        origin = st.text_input("🛫 Origin Airport Code", "").upper()
        outbound_date = st.date_input("🗓️ Outbound Flight Date", value=datetime(2025, 9, 10))
    with col2:
        destination = st.text_input("🛬 Destination Airport Code", "").upper()
        if trip_type == "round-trip":
            return_date = st.date_input("🗓️ Return Flight Date", value=datetime(2025, 9, 17))

    if st.button("🔍 Predict Prices"):
        if not validate_input(origin, destination, outbound_date, return_date if trip_type == "round-trip" else None):
            return

        with st.spinner("Loading data and making predictions..."):
            existing_data = load_data_from_gcs(origin, destination, trip_type)

            if existing_data.empty:
                st.info("No existing data found. Fetching new data from API.")
            else:
                st.success(f"Loaded existing records for analysis.")

            # Outbound flight API call
            api_calls_made = 0
            while should_call_api(origin, destination) and api_calls_made < 2:
                with st.spinner(f"Fetching new data for outbound flights (Call {api_calls_made + 1}/2)..."):
                    api_data = fetch_data_for_months(origin, destination, outbound_date, outbound_date + timedelta(days=30), "one-way")
                if api_data:
                    new_data = process_and_combine_data(api_data, existing_data, origin, destination, "one-way")
                    save_data_to_gcs(new_data, origin, destination, "one-way")
                                    update_api_call_time(origin, destination)
                    existing_data = new_data
                    api_calls_made += 1
                    logging.info(f"Successfully fetched and processed new outbound flight data (Call {api_calls_made}/2).")
                else:
                    logging.warning(f"No new data fetched from API for outbound flights on call {api_calls_made + 1}.")
                    break

            # Return flight API call (for round-trips)
            if trip_type == "round-trip":
                api_calls_made = 0
                while should_call_api(destination, origin) and api_calls_made < 2:
                    with st.spinner(f"Fetching new data for return flights (Call {api_calls_made + 1}/2)..."):
                        return_api_data = fetch_data_for_months(destination, origin, return_date, return_date + timedelta(days=30), "one-way")
                    if return_api_data:
                        return_new_data = process_and_combine_data(return_api_data, existing_data, destination, origin, "one-way")
                        save_data_to_gcs(return_new_data, destination, origin, "one-way")
                        update_api_call_time(destination, origin)
                        existing_data = pd.concat([existing_data, return_new_data], ignore_index=True)
                        api_calls_made += 1
                        logging.info(f"Successfully fetched and processed new return flight data (Call {api_calls_made}/2).")
                    else:
                        logging.warning(f"No new data fetched from API for return flights on call {api_calls_made + 1}.")
                        break

            if not existing_data.empty:
                logging.info(f"Total records for analysis: {len(existing_data)}")

                with st.expander("View Sample Data"):
                    st.dataframe(existing_data.head())

                df = engineer_features(existing_data, trip_type)
                model, train_mae, test_mae = train_model(df)

                logging.info(f"Model trained. Estimated price accuracy: ±${test_mae:.2f} (based on test data)")

                if trip_type == "round-trip":
                    future_prices = predict_prices(model, outbound_date, return_date, origin, destination, existing_data['origin'].unique(), existing_data['destination'].unique(), trip_type)
                else:
                    future_prices = predict_prices(model, outbound_date, outbound_date + timedelta(days=30), origin, destination, existing_data['origin'].unique(), existing_data['destination'].unique(), trip_type)

                st.subheader(f"📈 Predicted {trip_type.capitalize()} Prices")
                fig = plot_prices(future_prices, f"Predicted {trip_type.capitalize()} Prices ({origin} to {destination})")
                st.plotly_chart(fig, use_container_width=True)

                best_days = future_prices.nsmallest(5, 'predicted_price')
                st.subheader(f"💰 Best Days to Book {trip_type.capitalize()}")
                if trip_type == "round-trip":
                    st.table(best_days[['departure', 'return', 'predicted_price']].set_index('departure').rename(columns={'predicted_price': 'Predicted Price ($)', 'return': 'Return Date'}))
                else:
                    st.table(best_days[['departure', 'predicted_price']].set_index('departure').rename(columns={'predicted_price': 'Predicted Price ($)'}))

                avg_price = future_prices['predicted_price'].mean()
                st.metric(label=f"💵 Average Predicted {trip_type.capitalize()} Price", value=f"${avg_price:.2f}")

                price_range = future_prices['predicted_price'].max() - future_prices['predicted_price'].min()
                st.metric(label=f"📊 {trip_type.capitalize()} Price Range", value=f"${price_range:.2f}")

            else:
                st.error("No data available for prediction. Please try again with a different route or check your data source.")
                logging.error(f"No data available for prediction for {origin} to {destination}.")

if __name__ == "__main__":
    main()
