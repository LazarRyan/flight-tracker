import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime, timedelta
from amadeus import Client, ResponseError
import random
from google.cloud import storage
from io import StringIO
from google.oauth2 import service_account

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
    # Amadeus API configuration
    amadeus = Client(
        client_id=st.secrets["AMADEUS_CLIENT_ID"],
        client_secret=st.secrets["AMADEUS_CLIENT_SECRET"]
    )

    # Google Cloud Storage configuration
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
    try:
        content = blob.download_as_text()
        df = pd.read_csv(StringIO(content))
        df['departure'] = pd.to_datetime(df['departure'])
        return df
    except Exception:
        return pd.DataFrame(columns=['departure', 'price', 'origin', 'destination'])

def save_data_to_gcs(df, origin, destination):
    filename = get_data_filename(origin, destination)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob = bucket.blob(filename)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

def should_call_api(origin, destination):
    blob = bucket.blob(f"api_calls_{origin}_{destination}.txt")
    if blob.exists():
        content = blob.download_as_text().strip().split('\n')
        current_date = datetime.now().date().isoformat()
        calls_today = sum(1 for call in content if call.startswith(current_date))
        return calls_today < 2
    return True

def update_api_call_time(origin, destination):
    blob = bucket.blob(f"api_calls_{origin}_{destination}.txt")
    current_date = datetime.now().date().isoformat()
    if blob.exists():
        content = blob.download_as_text().strip().split('\n')
        # Keep only today's calls
        content = [line for line in content if line.startswith(current_date)]
    else:
        content = []
    
    content.append(f"{current_date}_{datetime.now().isoformat()}")
    blob.upload_from_string('\n'.join(content))


@st.cache_data
def get_flight_offers(origin, destination, year, month):
    start_date = max(datetime(year, month, 1), datetime.now())
    end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    
    all_offers = []
    days_in_month = (end_date - start_date).days + 1
    random_days = random.sample(range(days_in_month), min(3, days_in_month))

    for day in random_days:
        current_date = start_date + timedelta(days=day)
        try:
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin,
                destinationLocationCode=destination,
                departureDate=current_date.strftime("%Y-%m-%d"),
                adults=1,
                max=1
            )
            all_offers.extend(response.data)
        except ResponseError:
            pass

    return all_offers

def fetch_data_for_months(origin, destination, num_months=12):
    all_data = []
    current_date = datetime.now()
    progress_bar = st.progress(0)
    for i in range(num_months):
        month_data = get_flight_offers(origin, destination, current_date.year, current_date.month)
        all_data.extend(month_data)
        current_date = (current_date + timedelta(days=32)).replace(day=1)
        progress_bar.progress((i + 1) / num_months)
    return all_data

def process_and_combine_data(api_data, existing_data, origin, destination):
    new_data = [
        {'departure': offer['itineraries'][0]['segments'][0]['departure']['at'],
         'price': float(offer['price']['total']),
         'origin': origin,
         'destination': destination}
        for offer in api_data
    ]
    new_df = pd.DataFrame(new_data)
    new_df['departure'] = pd.to_datetime(new_df['departure'])

    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    combined_df = combined_df.sort_values('departure').drop_duplicates(subset=['departure', 'origin', 'destination'], keep='last')
    
    cutoff_date = datetime.now() - timedelta(days=365)
    combined_df = combined_df[
        (combined_df['departure'] >= cutoff_date) |
        (combined_df['departure'] >= datetime.now())
    ]
    return combined_df.sort_values('departure')

def engineer_features(df):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['days_to_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df = pd.get_dummies(df, columns=['origin', 'destination'], prefix=['origin', 'dest'])
    return df

@st.cache_resource
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
    future_df = pd.DataFrame({'departure': date_range, 'origin': origin, 'destination': destination})
    future_df = engineer_features(future_df)
    
    for o in all_origins:
        if f'origin_{o}' not in future_df.columns:
            future_df[f'origin_{o}'] = 0
    for d in all_destinations:
        if f'dest_{d}' not in future_df.columns:
            future_df[f'dest_{d}'] = 0
    
    X_future = future_df.drop('departure', axis=1)
    future_df['predicted_price'] = model.predict(X_future)

    return future_df

def plot_prices(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['departure'], y=df['predicted_price'],
                             mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title=title, xaxis_title='Departure Date',
                      yaxis_title='Predicted Price (USD)')
    return fig

def validate_input(origin, destination, target_date):
    if len(origin) != 3 or len(destination) != 3:
        st.error("Origin and destination must be 3-letter IATA airport codes.")
        return False
    if target_date <= datetime.now().date():
        st.error("Target date must be in the future.")
        return False
    return True

def main():
    st.title("✈️ Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    col1, col2, col3 = st.columns(3)

    with col1:
        origin = st.text_input("🛫 Origin Airport Code", "").upper()
    with col2:
        destination = st.text_input("🛬 Destination Airport Code", "").upper()
    with col3:
        target_date = st.date_input("🗓️ Target Flight Date", value=datetime(2025, 9, 10))

    if st.button("🔍 Predict Prices"):
        if not validate_input(origin, destination, target_date):
            return

        with st.spinner("Loading data and making predictions..."):
            existing_data = load_data_from_gcs(origin, destination)

            if existing_data.empty:
                st.info(f"⚠️ No existing data found for {origin} to {destination}. Fetching new data from API.")
            else:
                st.success(f"✅ Loaded {len(existing_data)} existing records for {origin} to {destination}.")

            api_calls_made = 0
            while should_call_api(origin, destination) and api_calls_made < 2:
                with st.spinner(f"Fetching new data from Amadeus API (Call {api_calls_made + 1}/2). This may take a few minutes..."):
                    api_data = fetch_data_for_months(origin, destination)
                if api_data:
                    new_data = process_and_combine_data(api_data, existing_data, origin, destination)
                    save_data_to_gcs(new_data, origin, destination)
                    update_api_call_time(origin, destination)
                    existing_data = new_data
                    api_calls_made += 1
                    st.success(f"✅ Successfully fetched and processed new data (Call {api_calls_made}/2).")
                else:
                    st.warning(f"⚠️ No new data fetched from API on call {api_calls_made + 1}.")
                    break

            if api_calls_made == 0:
                st.info(f"ℹ️ Using existing data for {origin} to {destination}. API call limit reached for this route today.")

            if not existing_data.empty:
                st.success(f"📊 Total records for analysis: {len(existing_data)}")

                with st.expander("View Sample Data"):
                    st.dataframe(existing_data.head())

                df = engineer_features(existing_data)
                model, train_mae, test_mae = train_model(df)

                st.info(f"🤖 Model trained. Estimated price accuracy: ±${test_mae:.2f} (based on test data)")

                start_date = datetime.now().date()
                end_date = target_date + timedelta(days=30)
                all_origins = existing_data['origin'].unique()
                all_destinations = existing_data['destination'].unique()
                future_prices = predict_prices(model, start_date, end_date, origin, destination, all_origins, all_destinations)

                st.subheader("📈 Predicted Prices")
                fig = plot_prices(future_prices, f"Predicted Flight Prices ({origin} to {destination})")
                st.plotly_chart(fig, use_container_width=True)

                best_days = future_prices.nsmallest(5, 'predicted_price')
                st.subheader("💰 Best Days to Buy Tickets")
                st.table(best_days[['departure', 'predicted_price']].set_index('departure').rename(columns={'predicted_price': 'Predicted Price ($)'}))

                days_until_target = (target_date - datetime.now().date()).days
                st.metric(label=f"⏳ Days until target date ({target_date})", value=days_until_target)

                avg_price = future_prices['predicted_price'].mean()
                st.metric(label="💵 Average Predicted Price", value=f"${avg_price:.2f}")

                price_range = future_prices['predicted_price'].max() - future_prices['predicted_price'].min()
                st.metric(label="📊 Price Range", value=f"${price_range:.2f}")

            else:
                st.error(f"❌ No data available for prediction for {origin} to {destination}. Please try again with a different route or check your data source.")

if __name__ == "__main__":
    main()
