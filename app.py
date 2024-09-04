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
AMADEUS_CLIENT_ID = st.secrets["AMADEUS_CLIENT_ID"]
AMADEUS_CLIENT_SECRET = st.secrets["AMADEUS_CLIENT_SECRET"]

amadeus = Client(
    client_id=AMADEUS_CLIENT_ID,
    client_secret=AMADEUS_CLIENT_SECRET
)

def format_price(price):
    return f"${price:,.2f}"

def load_and_preprocess_data(master_filepath, route_filepath):
    master_df = pd.DataFrame(columns=['departure', 'price', 'origin', 'destination'])
    route_df = pd.DataFrame(columns=['departure', 'price'])

    if os.path.exists(master_filepath):
        master_df = pd.read_csv(master_filepath, parse_dates=['departure'])
    else:
        st.warning(f"Master file not found: {master_filepath}. Starting with an empty master dataset.")

    if os.path.exists(route_filepath):
        route_df = pd.read_csv(route_filepath, parse_dates=['departure'])
    else:
        st.info(f"Route-specific file not found: {route_filepath}. Starting with an empty route dataset.")

    # Ensure data types and remove invalid entries
    for df in [master_df, route_df]:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price', 'departure'])
        df = df[(df['price'] > 0) & (df['departure'] > '2023-01-01')]

    return master_df, route_df

def should_call_api(origin, destination):
    cache_file = "api_calls.json"
    today = datetime.now().date().isoformat()
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            api_calls = json.load(f)
    else:
        api_calls = {}
    
    route_key = f"{origin}-{destination}"
    
    if today in api_calls and route_key in api_calls[today]:
        return False
    return True

def update_api_call_time(origin, destination):
    cache_file = "api_calls.json"
    today = datetime.now().date().isoformat()
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            api_calls = json.load(f)
    else:
        api_calls = {}
    
    if today not in api_calls:
        api_calls[today] = {}
    
    route_key = f"{origin}-{destination}"
    api_calls[today][route_key] = datetime.now().isoformat()
    
    with open(cache_file, "w") as f:
        json.dump(api_calls, f)

def get_flight_offers(origin, destination, departure_date):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date.strftime("%Y-%m-%d"),
            adults=1
        )
        return response.data
    except ResponseError as error:
        st.error(f"Error fetching data from Amadeus API: {error}")
        return []
    except Exception as e:
        st.error(f"Unexpected error when calling Amadeus API: {str(e)}")
        return []

def process_and_combine_data(api_data, master_df, route_df, origin, destination):
    new_data = []
    for offer in api_data:
        price = float(offer['price']['total'])
        departure = offer['itineraries'][0]['segments'][0]['departure']['at']
        new_data.append({'departure': departure, 'price': price, 'origin': origin, 'destination': destination})
    
    new_df = pd.DataFrame(new_data)
    new_df['departure'] = pd.to_datetime(new_df['departure'])
    
    # Update master DataFrame
    master_df = pd.concat([master_df, new_df], ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['departure', 'origin', 'destination'], keep='last')
    master_df = master_df.sort_values('departure')
    
    # Update route-specific DataFrame
    route_new_df = new_df[['departure', 'price']]
    route_df = pd.concat([route_df, route_new_df], ignore_index=True)
    route_df = route_df.drop_duplicates(subset=['departure'], keep='last')
    route_df = route_df.sort_values('departure')
    
    return master_df, route_df

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

def main():
    st.title("✈️ Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        origin = st.text_input("🛫 Origin Airport Code", "JFK")
    with col2:
        destination = st.text_input("🛬 Destination Airport Code", "FCO")
    with col3:
        target_date = st.date_input("🗓️ Target Flight Date", value=datetime(2025, 9, 10))
    
    if st.button("🔍 Predict Prices"):
        with st.spinner("Loading data and making predictions..."):
            master_csv_filename = "flight_prices_master.csv"
            route_csv_filename = f"flight_prices_{origin}_{destination}.csv"
            master_data, route_data = load_and_preprocess_data(master_csv_filename, route_csv_filename)
            
            if route_data.empty:
                st.warning("⚠️ No existing data found for this route. Attempting to fetch data from API.")
            else:
                st.success(f"✅ Loaded {len(route_data)} records for this route from existing data.")
            
            if should_call_api(origin, destination):
                api_data = get_flight_offers(origin, destination, target_date)
                if api_data:
                    st.success("✅ Successfully fetched new data from Amadeus API")
                    master_data, route_data = process_and_combine_data(api_data, master_data, route_data, origin, destination)
                    master_data.to_csv(master_csv_filename, index=False)
                    route_data.to_csv(route_csv_filename, index=False)
                    st.success(f"💾 Updated data saved to {master_csv_filename} and {route_csv_filename}")
                    update_api_call_time(origin, destination)
                else:
                    st.warning("⚠️ No new data fetched from API. Using existing data.")
            else:
                st.info("ℹ️ Using cached data. API call limit reached for this route today.")
            
            if not route_data.empty:
                st.write(f"📊 Total records for analysis: {len(route_data)}")
                
                with st.expander("View Sample Data"):
                    st.dataframe(route_data.head())
                
                df = engineer_features(route_data)
                model, train_mae, test_mae = train_model(df)
                
                st.info(f"🤖 Model trained. Train MAE: {format_price(train_mae)}, Test MAE: {format_price(test_mae)}")
                
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
                st.error("❌ No data available for prediction. Please check your data source or try again later.")

if __name__ == "__main__":
    main()
