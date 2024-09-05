import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from amadeus import Client, ResponseError
import time
import random

# Set page config
st.set_page_config(page_title="Flight Price Predictor", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1517479149777-5f3b1511d5ad?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
    }
    .Widget>label {
        color: white;
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        color: #4F8BF9;
        border-radius: 50px;
        height: 3em;
        width: 100%;
    }
    .stTextInput>div>div>input {
        color: #4F8BF9;
    }
    .css-145kmo2 {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Amadeus API configuration
try:
    AMADEUS_CLIENT_ID = st.secrets["AMADEUS_CLIENT_ID"]
    AMADEUS_CLIENT_SECRET = st.secrets["AMADEUS_CLIENT_SECRET"]
    amadeus = Client(client_id=AMADEUS_CLIENT_ID, client_secret=AMADEUS_CLIENT_SECRET)
except Exception as e:
    st.error(f"Error initializing Amadeus client: {e}")
    st.stop()

def format_price(price):
    return f"${price:,.2f}"

def load_and_preprocess_data(filepath):
    if not os.path.exists(filepath):
        st.warning(f"File not found: {filepath}. Starting with empty dataset.")
        return pd.DataFrame(columns=['departure', 'price'])

    try:
        df = pd.read_csv(filepath)
        if 'DepartureDate' in df.columns:
            df = df.rename(columns={'DepartureDate': 'departure', 'Price': 'price'})
        df['departure'] = pd.to_datetime(df['departure'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price', 'departure'])
        df = df[(df['price'] > 0) & (df['departure'] > '2023-01-01')]
        return df[['departure', 'price']]
    except Exception as e:
        st.error(f"Error loading data from {filepath}: {str(e)}")
        return pd.DataFrame(columns=['departure', 'price'])

def should_call_api(origin, destination):
    cache_file = f"last_api_call_{origin}_{destination}.txt"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            last_call = datetime.fromisoformat(f.read().strip())
        if datetime.now() - last_call < timedelta(hours=24):
            return False
    return True

def update_api_call_time(origin, destination):
    cache_file = f"last_api_call_{origin}_{destination}.txt"
    with open(cache_file, "w") as f:
        f.write(datetime.now().isoformat())

def get_flight_offers(origin, destination, year, month):
    start_date = datetime(year, month, 1)
    if start_date < datetime.now():
        start_date = datetime.now()
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
        except ResponseError as error:
            st.error(f"Error fetching data for {current_date.date()}: {error}")
    
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
    st.success(f"✅ Fetched data for {num_months} months")
    return all_data

def process_and_combine_data(api_data, existing_data):
    new_data = []
    for offer in api_data:
        price = float(offer['price']['total'])
        departure = offer['itineraries'][0]['segments'][0]['departure']['at']
        new_data.append({'departure': departure, 'price': price})
    
    new_df = pd.DataFrame(new_data)
    new_df['departure'] = pd.to_datetime(new_df['departure'])
    
    st.info(f"New data fetched: {len(new_df)} records")
    
    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    st.info(f"Combined data (before processing): {len(combined_df)} records")
    
    combined_df = combined_df.sort_values('departure').drop_duplicates(subset=['departure'], keep='last')
    st.info(f"After removing duplicates: {len(combined_df)} records")
    
    combined_df = combined_df.sort_values('departure')
    
    cutoff_date = datetime.now() - timedelta(days=365)
    combined_df = combined_df[
        (combined_df['departure'] >= cutoff_date) | 
        (combined_df['departure'] >= datetime.now())
    ]
    st.info(f"Final combined data: {len(combined_df)} records")
    
    return combined_df

def engineer_features(df):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['days_to_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

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
    future_df['predicted price'] = model.predict(X_future)
    future_df['formatted price'] = future_df['predicted price'].apply(format_price)
    
    return future_df

def plot_prices(df, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['departure'], df['predicted price'], marker='o', color='#4F8BF9')
    ax.set_title(title, color='black', fontsize=16)
    ax.set_xlabel('Departure Date', color='black')
    ax.set_ylabel('Predicted Price (USD)', color='black')
    ax.grid(True, color='gray', linestyle='--', alpha=0.7)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
    ax.set_ylim(bottom=0)
    ax.tick_params(colors='black', which='both')
    fig.autofmt_xdate()
    plt.tight_layout()
    
    st.pyplot(fig)

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
            existing_data = load_and_preprocess_data("flight_prices.csv")
            
            if existing_data.empty:
                st.warning("⚠️ No existing data found. Fetching new data from API.")
            else:
                st.success(f"✅ Loaded {len(existing_data)} records from existing data.")
            
            if should_call_api(origin, destination):
                st.info("Fetching new data from Amadeus API. This may take a few minutes...")
                api_data = fetch_data_for_months(origin, destination)
                if api_data:
                    st.success(f"✅ Successfully fetched new data from Amadeus API")
                    combined_data = process_and_combine_data(api_data, existing_data)
                    
                    combined_data.to_csv("flight_prices.csv", index=False)
                    st.success(f"💾 Updated data saved to flight_prices.csv ({len(combined_data)} records)")
                    
                    verified_data = load_and_preprocess_data("flight_prices.csv")
                    st.info(f"Verified saved data: {len(verified_data)} records")
                    
                    update_api_call_time(origin, destination)
                else:
                    st.warning("⚠️ No new data fetched from API. Using existing data.")
                    combined_data = existing_data
            else:
                st.info(f"ℹ️ Using cached data for {origin} to {destination}. API call limit reached for this route today.")
                combined_data = existing_data
            
            if not combined_data.empty:
                st.write(f"📊 Total records for analysis: {len(combined_data)}")
                
                with st.expander("View Sample Data"):
                    st.dataframe(combined_data.head())
                
                df = engineer_features(combined_data)
                model, train_mae, test_mae = train_model(df)
                
                st.info(f"🤖 Model trained. Estimated price accuracy: ±${test_mae:.2f} (based on test data)")
                
                start_date = datetime.now().date()
                end_date = target_date + timedelta(days=30)
                future_prices = predict_prices(model, start_date, end_date)
                
                st.subheader("📈 Predicted Prices")
                with st.container():
                    col1, col2, col3 = st.columns([1,3,1])
                    with col2:
                        plot_prices(future_prices, "Predicted Flight Prices")
                
                best_days = future_prices.nsmallest(5, 'predicted price')
                st.subheader("💰 Best Days to Buy Tickets")
                st.table(best_days[['departure', 'formatted price']].rename(columns={'formatted price': 'predicted price'}).set_index('departure'))
                
                days_left = (target_date - datetime.now().date()).days
                st.metric(label=f"⏳ Days until {target_date}", value=days_left)
            else:
                st.error("❌ No data available for prediction. Please try again with a different date or check your data source.")
                st.stop()

if __name__ == "__main__":
    main()
