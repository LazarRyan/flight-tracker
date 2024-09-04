import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.express as px
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
        background: url("https://images.unsplash.com/photo-1569154941061-e231b4725ef1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
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

# ... (keep the existing functions: load_and_preprocess_data, get_flight_offers, process_and_combine_data, engineer_features, train_model, predict_prices)

def plot_prices(df, title):
    fig = px.line(df, x='departure', y='predicted_price', title=title)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

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
            
            api_data = get_flight_offers(origin, destination, target_date)
            
            if api_data:
                st.success("✅ Successfully fetched new data from Amadeus API")
                combined_data = process_and_combine_data(api_data, existing_data)
                combined_data.to_csv("flight_prices.csv", index=False)
                st.success("💾 Updated data saved to flight_prices.csv")
            else:
                st.warning("⚠️ No new data fetched from API. Using existing data.")
                combined_data = existing_data
            
            if not combined_data.empty:
                st.write(f"📊 Total records for analysis: {len(combined_data)}")
                
                with st.expander("View Sample Data"):
                    st.dataframe(combined_data.head())
                
                df = engineer_features(combined_data)
                model, train_mae, test_mae = train_model(df)
                
                st.info(f"🤖 Model trained. Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}")
                
                start_date = datetime.now().date()
                end_date = target_date + timedelta(days=30)
                future_prices = predict_prices(model, start_date, end_date)
                
                st.subheader("📈 Predicted Prices")
                plot_prices(future_prices, "Predicted Flight Prices")
                
                best_days = future_prices.nsmallest(5, 'predicted_price')
                st.subheader("💰 Best Days to Buy Tickets")
                st.table(best_days[['departure', 'predicted_price']].set_index('departure'))
                
                days_left = (target_date - datetime.now().date()).days
                st.metric(label=f"⏳ Days until {target_date}", value=days_left)
            else:
                st.error("❌ No data available for prediction. Please check your data source or try again later.")

if __name__ == "__main__":
    main()
