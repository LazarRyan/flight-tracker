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

def load_and_preprocess_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, parse_dates=['DepartureDate'])
        df = df.rename(columns={
            'DepartureDate': 'departure',
            'Price': 'price',
            'Itineraries': 'itineraries',
            'ValidatingAirlineCodes': 'carriers',
            'TravelerPricings': 'price_details'
        })
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        def extract_price(price_data):
            try:
                price_dict = json.loads(price_data.replace("'", "\""))
                return float(price_dict[0]['price']['total'])
            except:
                return np.nan
        
        def extract_departure(itinerary_data):
            try:
                itinerary_dict = json.loads(itinerary_data.replace("'", "\""))
                return itinerary_dict[0]['segments'][0]['departure']['at']
            except:
                return np.nan
        
        if df['price'].isnull().all() or df['price'].max() == 'Price':
            df['price'] = df['price_details'].apply(extract_price)
            df['departure'] = df['itineraries'].apply(extract_departure)
        
        df['departure'] = pd.to_datetime(df['departure'])
        df = df.dropna(subset=['price', 'departure'])
        df = df[(df['price'] > 0) & (df['departure'] > '2023-01-01')]
        df['price'] = df['price'].apply(format_price)
        
        return df[['departure', 'price']]
    except Exception as e:
        st.error(f"Error loading data from {filepath}: {str(e)}")
        return pd.DataFrame()

def should_call_api():
    cache_file = "last_api_call.txt"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            last_call = datetime.fromisoformat(f.read().strip())
        if datetime.now() - last_call < timedelta(days=1):
            return False
    return True

def update_api_call_time():
    with open("last_api_call.txt", "w") as f:
        f.write(datetime.now().isoformat())

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

def process_and_combine_data(api_data, existing_data):
    new_data = []
    for offer in api_data:
        price = float(offer['price']['total'])
        departure = offer['itineraries'][0]['segments'][0]['departure']['at']
        new_data.append({'departure': departure, 'price': format_price(price)})
    
    new_df = pd.DataFrame(new_data)
    new_df['departure'] = pd.to_datetime(new_df['departure'])
    
    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['departure'], keep='last')
    combined_df = combined_df.sort_values('departure')
    
    return combined_df

def engineer_features(df):
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
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
    future_df['predicted price'] = future_df['predicted price'].apply(format_price)
    
    return future_df

def plot_prices(df, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    df['numeric_price'] = df['predicted price'].str.replace('$', '').str.replace(',', '').astype(float)
    ax.plot(df['departure'], df['numeric_price'], marker='o', color='#4F8BF9')
    ax.set_title(title, color='black', fontsize=16)
    ax.set_xlabel('Departure Date', color='black')
    ax.set_ylabel('Predicted Price (USD)', color='black')
    ax.grid(True, color='gray', linestyle='--', alpha=0.7)
    
    # Format y-axis ticks to show dollar amounts
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
    
    # Ensure y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    # Adjust tick colors
    ax.tick_params(colors='black', which='both')
    
    # Rotate x-axis labels for better readability
    fig.autofmt_xdate()
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Use Streamlit's native plotting function
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
            existing_data = load_and_preprocess_data("flight_prices.csv")
            
            if existing_data.empty:
                st.warning("⚠️ No existing data found. Attempting to fetch data from API.")
            else:
                st.success(f"✅ Loaded {len(existing_data)} records from existing data.")
            
            if should_call_api():
                api_data = get_flight_offers(origin, destination, target_date)
                if api_data:
                    st.success("✅ Successfully fetched new data from Amadeus API")
                    combined_data = process_and_combine_data(api_data, existing_data)
                    combined_data.to_csv("flight_prices.csv", index=False)
                    st.success("💾 Updated data saved to flight_prices.csv")
                    update_api_call_time()
                else:
                    st.warning("⚠️ No new data fetched from API. Using existing data.")
                    combined_data = existing_data
            else:
                st.info("ℹ️ Using cached data. API call limit reached for today.")
                combined_data = existing_data
            
            if not combined_data.empty:
                st.write(f"📊 Total records for analysis: {len(combined_data)}")
                
                with st.expander("View Sample Data"):
                    st.dataframe(combined_data.head())
                
                df = engineer_features(combined_data)
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
                st.table(best_days[['departure', 'predicted price']].set_index('departure'))
                
                days_left = (target_date - datetime.now().date()).days
                st.metric(label=f"⏳ Days until {target_date}", value=days_left)
            else:
                st.error("❌ No data available for prediction. Please check your data source or try again later.")

if __name__ == "__main__":
    main()
