import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
import time
import xboost as xbg
from sklearn.ensemble import GradientBoostingRegressor

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
                        'destination': destination,
                        'query_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    all_data.append(flight_data)
                    logging.info(f"Fetched data for {origin} to {destination} on {sample_date}")
                else:
                    logging.warning(f"No data found for {origin} to {destination} on {sample_date}")
                
                time.sleep(1)  # 1 second delay
            except ResponseError as error:
                st.error(f"Error fetching data from Amadeus API: {error}")
                logging.error(f"Error fetching data from Amadeus API: {error}")
                time.sleep(5)  # Longer delay on error
            except Exception as e:
                st.error(f"An unexpected error occurred while fetching data: {str(e)}")
                logging.error(f"Unexpected error in fetch_and_process_data: {str(e)}")
                time.sleep(5)  # Longer delay on error

        current_date += relativedelta(months=1)

    df = pd.DataFrame(all_data)
    if not df.empty:
        df['departure'] = pd.to_datetime(df['departure'])
        df['query_date'] = pd.to_datetime(df['query_date'])
    return df

def engineer_features(df):
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['day'] = df['departure'].dt.day
    df['days_until_flight'] = (df['departure'] - datetime.now()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = ((df['month'] == 12) & (df['day'].isin([24, 25, 31])) | 
                        (df['month'] == 1) & (df['day'] == 1)).astype(int)
    
    if 'query_date' not in df.columns:
        df['query_date'] = datetime.now()
    else:
        df['query_date'] = pd.to_datetime(df['query_date'])
    
    df['days_until_departure'] = (df['departure'] - df['query_date']).dt.days
    return df

def train_model(df, origin, destination):
    features = ['day_of_week', 'month', 'day', 'days_until_flight', 'is_weekend', 'is_holiday', 'days_until_departure']
    X = df[features]
    y = df['price']
    
    X['origin'] = origin
    X['destination'] = destination
    X = pd.get_dummies(X, columns=['origin', 'destination'], drop_first=True)
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if use_xgb:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    
    model.fit(X_train, y_train)
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    return model, train_mae, test_mae, feature_names

@st.cache_data
def train_model_with_retry(df, origin, destination, max_retries=3):
    for attempt in range(max_retries):
        try:
            with st.spinner(f'Training model (attempt {attempt + 1}/{max_retries})...'):
                model, train_mae, test_mae, feature_names = train_model(df, origin, destination)
            return model, train_mae, test_mae, feature_names
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"An error occurred during model training (attempt {attempt + 1}). Retrying...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                st.error(f"Failed to train model after {max_retries} attempts. Error: {str(e)}")
                logging.error(f"Model training failed: {str(e)}")
                return None, None, None, None

def predict_prices(model, feature_names, start_date, end_date, origin, destination):
    date_range = pd.date_range(start=start_date, end=end_date)
    future_data = pd.DataFrame({'departure': date_range})
    future_data['query_date'] = datetime.now()
    future_data = engineer_features(future_data)
    future_data['origin'] = origin
    future_data['destination'] = destination
    
    features = ['day_of_week', 'month', 'day', 'days_until_flight', 'is_weekend', 'is_holiday', 'days_until_departure', 'origin', 'destination']
    future_data_encoded = pd.get_dummies(future_data[features], columns=['origin', 'destination'], drop_first=True)
    
    # Ensure all columns from training are present
    for col in feature_names:
        if col not in future_data_encoded.columns:
            future_data_encoded[col] = 0
    
    # Ensure columns are in the same order as during training
    future_data_encoded = future_data_encoded[feature_names]
    
    future_data['predicted_price'] = model.predict(future_data_encoded)
    
    return future_data

def predict_best_buy_days(model, feature_names, travel_date, origin, destination):
    today = datetime.now().date()
    date_range = pd.date_range(start=today, end=travel_date)
    buy_data = pd.DataFrame({'query_date': date_range, 'departure': travel_date})
    buy_data = engineer_features(buy_data)
    buy_data['origin'] = origin
    buy_data['destination'] = destination
    
    features = ['day_of_week', 'month', 'day', 'days_until_flight', 'is_weekend', 'is_holiday', 'days_until_departure', 'origin', 'destination']
    buy_data_encoded = pd.get_dummies(buy_data[features], columns=['origin', 'destination'], drop_first=True)
    
    for col in feature_names:
        if col not in buy_data_encoded.columns:
            buy_data_encoded[col] = 0
    
    # Ensure columns are in the same order as during training
    buy_data_encoded = buy_data_encoded[feature_names]
    
    buy_data['predicted_price'] = model.predict(buy_data_encoded)
    
    return buy_data.nsmallest(5, 'predicted_price')

def plot_prices(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['departure'], y=df['predicted_price'], mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price ($)')
    return fig

def format_best_days_table(best_days):
    best_days['Date'] = best_days['departure'].dt.strftime('%b %d, %Y (%a)')
    best_days['Price'] = best_days['predicted_price'].apply(lambda x: f'${x:.2f}')
    result = best_days[['Date', 'Price']].rename(columns={'Price': 'Predicted Price'})
    return result

def validate_input(origin, destination, outbound_date):
    if not origin or not destination:
        st.error("Please enter both origin and destination airport codes.")
        return False
    if outbound_date <= datetime.now().date():
        st.error("Please select a future date for your outbound flight.")
        return False
    return True

def initialize_openai():
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets. Please add it to continue.")
        st.stop()
    openai.api_key = st.secrets["OPENAI_API_KEY"]

def chatbot(user_input, context):
    try:
        initialize_openai()  # Only initialize when the chatbot is used
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a flight price prediction app. You can answer questions about flights, travel to Italy, and using the app."},
                {"role": "user", "content": f"Context: {context}\n\nUser question: {user_input}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error in chatbot function: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}. Please try again later."

def main():
    st.title("✈️ Flight Price Predictor for Italy 2025")
    st.write("Plan your trip to Italy for Tanner & Jill's wedding!")

    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Enter origin airport code (e.g., LAX):", "EWR")
    with col2:
        destination = st.text_input("Enter destination airport code in Italy (e.g., FCO):", "FCO")

    min_date = datetime.now().date()
    max_date = datetime(2025, 12, 31).date()
    default_date = datetime(2025, 6, 1).date()

    travel_date = st.date_input("Select travel date:", 
                                min_value=min_date,
                                max_value=max_date,
                                value=default_date)

    if st.button("Predict Price"):
        if validate_input(origin, destination, travel_date):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress, status):
                progress_bar.progress(progress)
                status_text.text(status)

            try:
                update_progress(0, "Initializing...")
                
                update_progress(10, "Loading data from GCS...")
                df = load_data_from_gcs(origin, destination)

                update_progress(30, "Checking if API call is needed...")
                if df.empty or should_call_api(origin, destination):
                    update_progress(40, "Fetching new data from API...")
                    new_data = fetch_and_process_data(origin, destination, travel_date, travel_date + relativedelta(months=12))
                    df = pd.concat([                df = pd.concat([df, new_data]).drop_duplicates().reset_index(drop=True)
                    update_progress(50, "Saving updated data to GCS...")
                    save_data_to_gcs(df, origin, destination)

                update_progress(60, "Engineering features...")
                df = engineer_features(df)

                # Check data size and sample if necessary
                if len(df) > 5000:  # Reduced threshold
                    df = df.sample(n=5000, random_state=42)
                    st.info("Using a subset of the data for model training due to performance constraints.")

                update_progress(70, "Training model...")
                model, train_mae, test_mae, feature_names = train_model_with_retry(df, origin, destination)
                if model is None:
                    st.stop()

                update_progress(80, "Predicting prices...")
                future_prices = predict_prices(model, feature_names, travel_date, travel_date + timedelta(days=30), origin, destination)
                best_buy_days = predict_best_buy_days(model, feature_names, travel_date, origin, destination)

                update_progress(90, "Generating visualizations...")
                st.write(f"Model performance - Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}")
                
                fig = plot_prices(future_prices, f"Predicted Prices for {origin} to {destination}")
                st.plotly_chart(fig)
                
                st.write("Best days to fly in the next 30 days:")
                st.table(format_best_days_table(future_prices.nsmallest(5, 'predicted_price')))

                st.write("Best days to buy tickets for your selected travel date:")
                st.table(format_best_days_table(best_buy_days))

                update_progress(100, "Completed!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error in prediction process: {str(e)}")

    st.subheader("💬 Chat with our AI Assistant")
    user_input = st.text_input("Ask a question about flights, travel to Italy, or using this app:")
    if user_input:
        try:
            with st.spinner("AI Assistant is thinking..."):
                context = f"The user is using a flight price prediction app for travel to Italy in 2025. They can input origin and destination airport codes and select a date to predict flight prices."
                response = chatbot(user_input, context)
            st.write("AI Assistant:", response)
        except Exception as e:
            st.error(f"An error occurred with the AI Assistant: {str(e)}")
            logging.error(f"Error in AI Assistant: {str(e)}")

    st.subheader("🇮🇹 About Italy")
    st.write("""
    Italy is a country located in Southern Europe, known for its rich history, 
    stunning architecture, delicious cuisine, and beautiful landscapes. 
    Some popular destinations include Rome, Florence, Venice, and the Amalfi Coast.
    """)

    st.markdown("---")
    st.markdown("Developed with ❤️ for Tanner & Jill's wedding")

if __name__ == "__main__":
    main()
