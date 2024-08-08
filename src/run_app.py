import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import joblib

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
    return df

def load_model(model_file):
    return joblib.load(model_file)

def main():
    st.title("Flight Price Predictor")

    historical_csv = 'flight_prices.csv'
    future_csv = 'future_prices.csv'
    model_file = 'flight_price_model.pkl'

    # Load data
    historical_df = load_data(historical_csv)
    future_df = load_data(future_csv)
    model = load_model(model_file)

    st.write("### Historical Flight Prices")
    st.dataframe(historical_df)

    st.write("### Future Flight Price Predictions")
    st.dataframe(future_df)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(historical_df['DepartureDate'], historical_df['Price'], label='Historical Prices')
    ax.plot(future_df['DepartureDate'], future_df['PredictedPrice'], label='Predicted Prices', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    st.pyplot(fig)

    # Countdown to September 2025
    target_date = datetime(2025, 9, 10)
    current_date = datetime.now()
    countdown = (target_date - current_date).days
    st.write(f"### Countdown to September 10, 2025: {countdown} days")

if __name__ == "__main__":
    main()

