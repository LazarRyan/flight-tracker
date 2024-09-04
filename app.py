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

def load_and_preprocess_data(filepath, origin, destination):
    st.write(f"Debug: Loading data from {filepath} for route {origin}-{destination}")
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        st.write("Debug: CSV file loaded successfully")
        st.write("Debug: CSV columns:", df.columns.tolist())
        st.write("Debug: First few rows of CSV:")
        st.write(df.head())

        if 'itineraries' not in df.columns:
            st.error("'itineraries' column not found in CSV")
            return pd.DataFrame()

        def extract_route(itinerary_data):
            try:
                itinerary_dict = json.loads(itinerary_data.replace("'", "\""))
                first_segment = itinerary_dict[0]['segments'][0]
                return f"{first_segment['departure']['iataCode']}-{first_segment['arrival']['iataCode']}"
            except Exception as e:
                st.write(f"Debug: Error extracting route: {e}")
                return None

        df['route'] = df['itineraries'].apply(extract_route)
        st.write(f"Debug: Routes extracted. Unique routes: {df['route'].unique().tolist()}")
        
        st.write(f"Debug: Filtering for route: {origin}-{destination}")
        st.write(f"Debug: Rows before filtering: {len(df)}")
        df = df[df['route'] == f"{origin}-{destination}"]
        st.write(f"Debug: Rows after filtering: {len(df)}")

        if 'DepartureDate' in df.columns and 'Price' in df.columns:
            df['departure'] = pd.to_datetime(df['DepartureDate'])
            df['price'] = pd.to_numeric(df['Price'], errors='coerce')
        else:
            st.error("Required columns 'DepartureDate' or 'Price' not found in CSV")
            return pd.DataFrame()

        df = df.dropna(subset=['price', 'departure'])
        df = df[(df['price'] > 0) & (df['departure'] > '2023-01-01')]
        st.write(f"Debug: Final dataframe shape: {df.shape}")
        return df[['departure', 'price']]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def should_call_api(route):
    cache_file = f"last_api_call_{route}.txt"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            last_call = datetime.fromisoformat(f.read().strip())
        if datetime.now().date() == last_call.date():
            return False
    return True

def update_api_call_time(route):
    with open(f"last_api_call_{route}.txt", "w") as f:
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

def process_and_combine_data(api_data, existing_data, origin, destination):
    new_data = []
    for offer in api_data:
        price = float(offer['price']['total'])
        departure = offer['itineraries'][0]['segments'][0]['departure']['at']
        new_data.append({
            'DepartureDate': departure,
            'Price': price,
            'itineraries': json.dumps(offer['itineraries'])
        })
    
    new_df = pd.DataFrame(new_data)
    new_df['DepartureDate'] = pd.to_datetime(new_df['DepartureDate'])
    
    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['DepartureDate'], keep='last')
    combined_df = combined_df.sort_values('DepartureDate')
    
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
    
    route = f"{origin}-{destination}"
    
    if st.button("🔍 Predict Prices"):
        with st.spinner("Loading data and making predictions..."):
            existing_data = load_and_preprocess_data("flight_prices.csv", origin, destination)
            
            if existing_data.empty:
                st.warning(f"⚠️ No existing data found for route {route}. Attempting to fetch data from API.")
            else:
                st.success(f"✅ Loaded {len(existing_data)} records for route {route}.")
            
            if should_call_api(route):
                api_data = get_flight_offers(origin, destination, target_date)
                if api_data:
                    st.success(f"✅ Successfully fetched new data from Amadeus API for route {route}")
                    combined_data = process_and_combine_data(api_data, existing_data, origin, destination)
                    combined_data.to_csv("flight_prices.csv", mode='a', header=not os.path.exists("flight_prices.csv"), index=False)
                    st.success("💾 Updated data saved to flight_prices.csv")
                    update_api_call_time(route)
                else:
                    st.warning(f"⚠️ No new data fetched from API for route {route}. Using existing data.")
                    combined_data = existing_data
            else:
                st.info(f"ℹ️ Using cached data for route {route}. API call limit reached for today.")
                combined_data = existing_data
            
            if not combined_data.empty:
                st.write(f"📊 Total records for analysis: {len(combined_data)}")
                
                with st.expander("View Sample Data"):
                    st.dataframe(combined_data.head())
                
                df = engineer_features(combined_data)
                model, train_mae, test_mae = train_model(df)
                
                st.info(f"🤖 Model trained for route {route}. Train MAE: {format_price(train_mae)}, Test MAE: {format_price(test_mae)}")
                
                start_date = datetime.now().date()
                end_date = target_date + timedelta(days=30)
                future_prices = predict_prices(model, start_date, end_date)
                
                st.subheader("📈 Predicted Prices")
                with st.container():
                    col1, col2, col3 = st.columns([1,3,1])
                    with col2:
                        plot_prices(future_prices, f"Predicted Flight Prices for {route}")
                
                best_days = future_prices.nsmallest(5, 'predicted price')
                st.subheader("💰 Best Days to Buy Tickets")
                st.table(best_days[['departure', 'formatted price']].rename(columns={'formatted price': 'predicted price'}).set_index('departure'))
                
                days_left = (target_date - datetime.now().date()).days
                st.metric(label=f"⏳ Days until {target_date}", value=days_left)
            else:
                st.error(f"❌ No data available for prediction for route {route}. Please check your data source or try again later.")

if __name__ == "__main__":
    main()
