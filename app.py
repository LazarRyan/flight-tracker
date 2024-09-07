import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from amadeus import Client, ResponseError
from google.cloud import storage
from io import StringIO
from google.oauth2 import service_account
import logging
import random
import json
import openai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page config
st.set_page_config(page_title="Flight Price Predictor", layout="wide")

# Initialize clients
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
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    logging.debug(f"OpenAI API key set: {'*' * len(openai.api_key)}")
    return amadeus, bucket

amadeus, bucket = initialize_clients()

def get_data_filename(origin, destination):
    return f"flight_prices_{origin}_{destination}.csv"

def load_data_from_gcs(origin, destination):
    filename = get_data_filename(origin, destination)
    blob = bucket.blob(filename)
    df = pd.DataFrame()

    if blob.exists():
        try:
            content = blob.download_as_text()
            df = pd.read_csv(StringIO(content))
            df['departure'] = pd.to_datetime(df['departure'])
            logging.info(f"Loaded {len(df)} records for {origin} to {destination}")
        except Exception as e:
            logging.warning(f"Error loading data for {origin} to {destination}: {str(e)}")

    return df

def save_data_to_gcs(df, origin, destination):
    filename = get_data_filename(origin, destination)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob = bucket.blob(filename)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    logging.info(f"Saved {len(df)} records for {origin} to {destination}")

def should_call_api(origin, destination):
    api_calls_file = "api_calls.json"
    blob = bucket.blob(api_calls_file)
    now = datetime.now()
    
    try:
        content = blob.download_as_text()
        api_calls = json.loads(content)
    except Exception:
        api_calls = {}

    route_key = f"{origin}-{destination}"
    if route_key in api_calls:
        last_call_time = datetime.fromisoformat(api_calls[route_key])
        if now - last_call_time < timedelta(hours=12):
            return False

    api_calls[route_key] = now.isoformat()
    blob.upload_from_string(json.dumps(api_calls), content_type="application/json")

    return True

def fetch_and_process_data(origin, destination, start_date, end_date):
    all_data = []
    current_date = start_date
    end_date = start_date + relativedelta(months=12)

    while current_date < end_date:
        month_end = current_date + relativedelta(months=1, days=-1)
        sample_dates = [current_date + timedelta(days=random.randint(0, (month_end - current_date).days)) for _ in range(3)]

        for sample_date in sample_dates:
            try:
                response = amadeus.shopping.flight_offers_search.get(
                    originLocationCode=origin,
                    destinationLocationCode=destination,
                    departureDate=sample_date.strftime('%Y-%m-%d'),
                    adults=1
                )
                data = response.data
                if data:
                    flight_data = {
                        'departure': data[0]['itineraries'][0]['segments'][0]['departure']['at'],
                        'price': float(data[0]['price']['total']),
                        'origin': origin,
                        'destination': destination
                    }
                    all_data.append(flight_data)
                    logging.info(f"Fetched data for {origin} to {destination} on {sample_date}")
                else:
                    logging.warning(f"No data found for {origin} to {destination} on {sample_date}")
            except ResponseError as error:
                st.error(f"Error fetching data from Amadeus API: {error}")
                logging.error(f"Error fetching data from Amadeus API: {error}")
            except Exception as e:
                st.error(f"An unexpected error occurred while fetching data: {str(e)}")
                logging.error(f"Unexpected error in fetch_and_process_data: {str(e)}")

        current_date += relativedelta(months=1)

    df = pd.DataFrame(all_data)
    if not df.empty:
        df['departure'] = pd.to_datetime(df['departure'])
    return df

def engineer_features(df):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['day'] = df['departure'].dt.day
    df['days_until_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = ((df['month'] == 12) & (df['day'].isin([24, 25, 31])) | 
                        (df['month'] == 1) & (df['day'] == 1)).astype(int)
    return df

def train_model(df):
    features = ['day_of_week', 'month', 'day', 'days_until_flight', 'is_weekend', 'is_holiday']
    X = df[features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1]
    }
    
    model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5)
    model.fit(X_train, y_train)
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    return model.best_estimator_, train_mae, test_mae

def predict_prices(model, start_date, end_date, origin, destination):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_data = pd.DataFrame({'departure': date_range})
    future_data = engineer_features(future_data)
    future_data['origin'] = origin
    future_data['destination'] = destination
    
    features = ['day_of_week', 'month', 'day', 'days_until_flight', 'is_weekend', 'is_holiday']
    future_data['predicted_price'] = model.predict(future_data[features])
    
    return future_data

def plot_prices(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['departure'], y=df['predicted_price'], mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price ($)')
    return fig

def validate_input(origin, destination, outbound_date):
    if not origin or not destination:
        st.error("Please enter both origin and destination airport codes.")
        return False
    if outbound_date <= datetime.now().date():
        st.error("Please select a future date for your outbound flight.")
        return False
    return True

def get_airport_info(airport_code):
    # This is a placeholder function. You should replace it with a proper airport code database or API.
    airport_info = {
        'FCO': {'city': 'Rome', 'country': 'Italy'},
        'MXP': {'city': 'Milan', 'country': 'Italy'},
        'VCE': {'city': 'Venice', 'country': 'Italy'},
        'NAP': {'city': 'Naples', 'country': 'Italy'},
        # Add more airport codes as needed
    }
    return airport_info.get(airport_code, {'city': airport_code, 'country': 'Italy'})

def get_ai_tourism_advice(destination):
    airport_info = get_airport_info(destination)
    city = airport_info['city']
    country = airport_info['country']
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant providing concise advice about tourist attractions."},
                {"role": "user", "content": f"Give me 3 must-visit attractions in {city}, {country}. Provide a brief one-sentence description for each."}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error in AI tourism advice: {str(e)}")
        logging.error(f"Full error details: {e.__class__.__name__}: {str(e)}")
        raise

def format_best_days_table(df):
    df = df.copy()
    df['departure'] = pd.to_datetime(df['departure']).dt.strftime('%B %d, %Y')
    df['Predicted Price'] = df['predicted_price'].apply(lambda x: f'${x:.2f}')
    df['Weekend'] = df['is_weekend'].map({0: 'No', 1: 'Yes'})
    df['Holiday'] = df['is_holiday'].map({0: 'No', 1: 'Yes'})
    df = df.rename(columns={
        'day_of_week': 'Day of Week',
        'month': 'Month',
        'day': 'Day',
        'days_until_flight': 'Days Until Flight'
    })
    columns_order = ['Day of Week', 'Month', 'Day', 'Days Until Flight', 'Weekend', 'Holiday', 'origin', 'destination', 'Predicted Price']
    return df[columns_order].set_index('departure')

def main():
    st.title("✈️ Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    col1, col2 = st.columns(2)

    with col1:
        origin = st.text_input("🛫 Origin Airport Code", "").upper()
        outbound_date = st.date_input("🗓️ Outbound Flight Date", value=datetime(2025, 9, 10))
    with col2:
        destination = st.text_input("🛬 Destination Airport Code in Italy", "").upper()

    if st.button("🔍 Predict Prices"):
        if not validate_input(origin, destination, outbound_date):
            return

        with st.spinner("Loading data and making predictions..."):
            try:
                existing_data = load_data_from_gcs(origin, destination)

                if not existing_data.empty:
                    st.success(f"Using existing data for {origin} to {destination}")
                    st.info(f"Total records: {len(existing_data)}")
                else:
                    st.info(f"No existing data found for {origin} to {destination}. Will fetch new data.")

                if should_call_api(origin, destination):
                    st.info("Fetching new data from API...")
                    new_data = fetch_and_process_data(origin, destination, datetime.now().date(), outbound_date)
                    if not new_data.empty:
                        existing_data = pd.concat([existing_data, new_data], ignore_index=True)
                        existing_data = existing_data.sort_values('departure').drop_duplicates(subset=['departure', 'origin', 'destination'], keep='last')
                        save_data_to_gcs(existing_data, origin, destination)
                        st.success(f"Data updated successfully. Total records: {len(existing_data)}")
                    else:
                        st.warning("Unable to fetch new data from API. Proceeding with existing data.")
                else:
                    st.info("API call limit reached for this route. Using existing data.")

                if existing_data.empty:
                    st.error("No data available for prediction. Please try again later or with a different route.")
                    return

                st.success(f"Analyzing {len(existing_data)} records for your route.")

                with st.expander("View Sample Data"):
                    st.dataframe(existing_data.head())

                df = engineer_features(existing_data)
                model, train_mae, test_mae = train_model(df)

                logging.info(f"Model trained. Estimated price accuracy: ±${test_mae:.2f} (based on test data)")

                future_prices = predict_prices(model, datetime.now().date(), outbound_date, origin, destination)

                st.subheader("📈 Predicted Prices")
                fig = plot_prices(future_prices, f"Predicted Prices ({origin} to {destination})")
                st.plotly_chart(fig, use_container_width=True)

                best_days = future_prices.nsmallest(5, 'predicted_price')
                st.subheader("💰 Best Days to Book")
                formatted_best_days = format_best_days_table(best_days)
                st.table(formatted_best_days)

                avg_price = future_prices['predicted_price'].mean()
                st.metric(label="💵 Average Predicted Price", value=f"${avg_price:.2f}")

                price_range = future_prices['predicted_price'].max() - future_prices['predicted_price'].min()
                st.metric(label="📊 Price Range", value=f"${price_range:.2f}")

                st.info(f"Predictions shown are for flights from today until {outbound_date}.")

                # AI Tourism Advice
                st.subheader("🏛️ AI Tourism Advice")
                airport_info = get_airport_info(destination)
                city = airport_info['city']
                country = airport_info['country']
                try:
                    advice = get_ai_tourism_advice(destination)
                    st.write(f"Top attractions in {city}, {country}:")
                    st.write(advice)
                except Exception as e:
                    logging.error(f"Error in AI tourism advice: {str(e)}")
                    st.error(f"Sorry, I couldn't retrieve tourism advice for {city}, {country} at the moment. Please try again later.")

                st.info("The AI tourism advice is generated based on the destination airport code or city name. For more accurate results, consider using major airport codes or city names.")

            except Exception as e:
                                st.error(f"An unexpected error occurred: {str(e)}")
                logging.error(f"Unexpected error in main function: {str(e)}", exc_info=True)

    st.markdown("---")
    st.markdown("Developed with ❤️ for Tanner & Jill's wedding")

if __name__ == "__main__":
    main()
