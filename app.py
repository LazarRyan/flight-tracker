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

# Initialize clients (assuming you have this function)
amadeus, bucket = initialize_clients()

def get_data_filename(origin, destination):
    return f"flight_prices_{origin}_{destination}.csv"

def load_data_from_gcs(origin, destination):
    filename = get_data_filename(origin, destination)
    blob = bucket.blob(filename)
    df = pd.DataFrame()

    try:
        content = blob.download_as_text()
        df = pd.read_csv(StringIO(content))
        df['departure'] = pd.to_datetime(df['departure'])
        if 'return' in df.columns:
            df['return'] = pd.to_datetime(df['return'])
        logging.info(f"Loaded {len(df)} records for {origin} to {destination}")
    except Exception as e:
        logging.warning(f"Error loading data for {origin} to {destination}: {str(e)}")

    return df

def engineer_features(df, trip_type, return_date=None):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['days_to_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    if trip_type == "round-trip" and return_date is not None:
        df['return_day_of_week'] = return_date.weekday()
        df['return_month'] = return_date.month
        df['trip_duration'] = (return_date - df['departure'].dt.date).dt.days
    else:
        df['return_day_of_week'] = 0
        df['return_month'] = 0
        df['trip_duration'] = 0
    
    df = pd.get_dummies(df, columns=['origin', 'destination'], prefix=['origin', 'dest'])
    return df

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

def predict_prices(model, end_date, origin, destination, all_origins, all_destinations, trip_type, return_date=None):
    start_date = datetime.now().date()
    date_range = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'departure': date_range})
    future_df['origin'] = origin
    future_df['destination'] = destination
    future_df = engineer_features(future_df, trip_type, return_date)
    
    for o in all_origins:
        if f'origin_{o}' not in future_df.columns:
            future_df[f'origin_{o}'] = 0
    for d in all_destinations:
        if f'dest_{d}' not in future_df.columns:
            future_df[f'dest_{d}'] = 0
    
    X_future = future_df.drop(['departure'], axis=1)
    future_df['predicted_price'] = model.predict(X_future)

    return future_df

def plot_prices(df, title, trip_type):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['departure'], y=df['predicted_price'],
                             mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title=title, xaxis_title='Departure Date',
                      yaxis_title='Predicted Price (USD)')
    if trip_type == "round-trip":
        fig.add_vline(x=df['departure'].iloc[-1], line_dash="dash", line_color="red",
                      annotation_text="Return Date", annotation_position="top right")
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
            existing_data = load_data_from_gcs(origin, destination)

            if existing_data.empty:
                st.error("No existing data found for this route. Please try a different route or check the data source.")
                return

            st.success(f"Loaded {len(existing_data)} existing records for analysis.")

            with st.expander("View Sample Data"):
                st.dataframe(existing_data.head())

            # Engineer features
            df = engineer_features(existing_data, trip_type, return_date if trip_type == "round-trip" else None)

            # Train model
            model, train_mae, test_mae = train_model(df)
            logging.info(f"Model trained. Estimated price accuracy: ±${test_mae:.2f} (based on test data)")

            # Predict future prices
            end_date = return_date if trip_type == "round-trip" else outbound_date
            future_prices = predict_prices(model, end_date, origin, destination, 
                                           existing_data['origin'].unique(), 
                                           existing_data['destination'].unique(), 
                                           trip_type, return_date if trip_type == "round-trip" else None)

            # Display results
            st.subheader(f"📈 Predicted {trip_type.capitalize()} Prices")
            fig = plot_prices(future_prices, f"Predicted {trip_type.capitalize()} Prices ({origin} to {destination})", trip_type)
            st.plotly_chart(fig, use_container_width=True)

            best_days = future_prices.nsmallest(5, 'predicted_price')
            st.subheader(f"💰 Best Days to Book {trip_type.capitalize()}")
            st.table(best_days[['departure', 'predicted_price']].set_index('departure').rename(columns={'predicted_price': 'Predicted Price ($)'}))

            avg_price = future_prices['predicted_price'].mean()
            st.metric(label=f"💵 Average Predicted {trip_type.capitalize()} Price", value=f"${avg_price:.2f}")

            price_range = future_prices['predicted_price'].max() - future_prices['predicted_price'].min()
            st.metric(label=f"📊 {trip_type.capitalize()} Price Range", value=f"${price_range:.2f}")

            if trip_type == "round-trip":
                st.info(f"Predictions shown are for outbound flights from today until {return_date}, assuming a return on {return_date}.")
            else:
                st.info(f"Predictions shown are for flights from today until {outbound_date}.")

if __name__ == "__main__":
    main()
