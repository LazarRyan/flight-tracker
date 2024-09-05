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

def load_data_from_gcs():
    blob = bucket.blob("flight_prices.csv")
    try:
        content = blob.download_as_text()
        df = pd.read_csv(StringIO(content))
        df['departure'] = pd.to_datetime(df['departure'])
        return df
    except Exception:
        return pd.DataFrame(columns=['departure', 'price'])

def save_data_to_gcs(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob = bucket.blob("flight_prices.csv")
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

def should_call_api(origin, destination):
    blob = bucket.blob(f"last_api_call_{origin}_{destination}.txt")
    if blob.exists():
        last_call = datetime.fromisoformat(blob.download_as_text().strip())
        return datetime.now() - last_call >= timedelta(hours=24)
    return True

def update_api_call_time(origin, destination):
    blob = bucket.blob(f"last_api_call_{origin}_{destination}.txt")
    blob.upload_from_string(datetime.now().isoformat())

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

def process_and_combine_data(api_data, existing_data):
    new_data = [
        {'departure': offer['itineraries'][0]['segments'][0]['departure']['at'],
         'price': float(offer['price']['total'])}
        for offer in api_data
    ]
    new_df = pd.DataFrame(new_data)
    new_df['departure'] = pd.to_datetime(new_df['departure'])

    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    combined_df = combined_df.sort_values('departure').drop_duplicates(subset=['departure'], keep='last')
    
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
    return df

@st.cache_resource
def train_model(df):
    X = df[['day_of_week', 'month', 'days_to_flight', 'is_weekend']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    return model, train_mae, test_mae

def predict_prices(model, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'departure': date_range})
    future_df = engineer_features(future_df)

    X_future = future_df[['day_of_week', 'month', 'days_to_flight', 'is_weekend']]
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
        origin = st.text_input("🛫 Origin Airport Code", "EWR").upper()
    with col2:
        destination = st.text_input("🛬 Destination Airport Code", "FCO").upper()
    with col3:
        target_date = st.date_input("🗓️ Target Flight Date", value=datetime(2025, 9, 10))

    if st.button("🔍 Predict Prices"):
        if not validate_input(origin, destination, target_date):
            return

        with st.spinner("Loading data and making predictions..."):
            existing_data = load_data_from_gcs()

            if existing_data.empty:
                st.info("⚠️ No existing data found. Fetching new data from API.")
            else:
                st.success(f"✅ Loaded {len(existing_data)} records from existing data in GCS.")

            if should_call_api(origin, destination):
                with st.spinner("Fetching new data from Amadeus API. This may take a few minutes..."):
                    api_data = fetch_data_for_months(origin, destination)
                if api_data:
                    combined_data = process_and_combine_data(api_data, existing_data)
                    save_data_to_gcs(combined_data)
                    update_api_call_time(origin, destination)
                else:
                    st.warning("⚠️ No new data fetched from API. Using existing data.")
                    combined_data = existing_data
            else:
                st.info(f"ℹ️ Using cached data for {origin} to {destination}. API call limit reached for this route today.")
                combined_data = existing_data

            if not combined_data.empty:
                st.success(f"📊 Total records for analysis: {len(combined_data)}")

                df = engineer_features(combined_data)
                model, train_mae, test_mae = train_model(df)

                st.info(f"🤖 Model trained. Estimated price accuracy: ±${test_mae:.2f} (based on test data)")

                start_date = datetime.now().date()
                end_date = target_date + timedelta(days=30)
                future_prices = predict_prices(model, start_date, end_date)

                st.subheader("📈 Predicted Prices")
                st.plotly_chart(plot_prices(future_prices, "Predicted Flight Prices"), use_container_width=True)

                best_days = future_prices.nsmallest(5, 'predicted_price')
                st.subheader("💰 Best Days to Buy Tickets")
                st.table(best_days[['departure', 'predicted_price']].set_index('departure').rename(columns={'predicted_price': 'Predicted Price'}))

                days_left = (target_date - datetime.now().date()).days
                st.metric(label=f"⏳ Days until {target_date}", value=days_left)
            else:
                st.error("❌ No data available for prediction. Please try again with a different date or check your data source.")

if __name__ == "__main__":
    main()
