import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from amadeus import Client, ResponseError
from google.cloud import storage
from io import StringIO
import pandas as pd
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
import json
from dotenv import load_dotenv
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost", "support_credentials": True}})

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize clients for Amadeus API and Google Cloud Storage
def initialize_clients():
    try:
        # Initialize Amadeus client
        amadeus = Client(
            client_id=os.environ["AMADEUS_CLIENT_ID"],
            client_secret=os.environ["AMADEUS_CLIENT_SECRET"]
        )
        logging.info("Amadeus client initialized successfully")

        # Initialize Google Cloud Storage client
        gcp_credentials_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])
        credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(os.environ["GCS_BUCKET_NAME"])
        logging.info("GCS client initialized successfully")

        return amadeus, bucket
    except Exception as e:
        logging.error(f"Error initializing clients: {str(e)}")
        raise

# Check environment variables
def check_env_variables():
    required_vars = ["AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET", "GCP_SERVICE_ACCOUNT", "GCS_BUCKET_NAME"]
    for var in required_vars:
        if var not in os.environ:
            raise EnvironmentError(f"Missing required environment variable: {var}")

check_env_variables()

# Call the function to initialize clients
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
        time_since_last_call = now - last_call_time
        if time_since_last_call < timedelta(hours=12):
            time_until_next_call = timedelta(hours=12) - time_since_last_call
            return False, time_until_next_call

    api_calls[route_key] = now.isoformat()
    blob.upload_from_string(json.dumps(api_calls), content_type="application/json")

    return True, timedelta(0)

def fetch_and_process_data(origin, destination, start_date):
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
                logging.error(f"Error fetching data from Amadeus API: {error}")
            except Exception as e:
                logging.error(f"Unexpected error in fetch_and_process_data: {str(e)}")

        current_date += relativedelta(months=1)

    df = pd.DataFrame(all_data)
    if not df.empty:
        df['departure'] = pd.to_datetime(df['departure'])
    return df

def update_data(origin, destination, start_date):
    existing_data = load_data_from_gcs(origin, destination)

    can_call_api, time_until_next_call = should_call_api(origin, destination)

    if can_call_api:
        logging.info("Fetching new data from API...")
        new_data = fetch_and_process_data(origin, destination, start_date)
        if not new_data.empty:
            existing_data = pd.concat([existing_data, new_data], ignore_index=True)
            existing_data = existing_data.sort_values('departure').drop_duplicates(subset=['departure', 'origin', 'destination'], keep='last')
            save_data_to_gcs(existing_data, origin, destination)
            logging.info(f"Data updated successfully. Total records: {len(existing_data)}")
        else:
            logging.warning("Unable to fetch new data from API. Proceeding with existing data.")
    else:
        hours, remainder = divmod(time_until_next_call.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        logging.info(f"API call limit reached for this route. Using existing data. Next API call possible in {hours} hours and {minutes} minutes.")

    if existing_data.empty:
        raise ValueError("No data available for prediction. Please try again later or with a different route.")

    return existing_data

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
    today = datetime.now().date()
    date_range = pd.date_range(start=max(today, start_date.date()), end=end_date.date())
    future_data = pd.DataFrame({'departure': date_range})
    future_data = engineer_features(future_data)
    future_data['origin'] = origin
    future_data['destination'] = destination

    features = ['day_of_week', 'month', 'day', 'days_until_flight', 'is_weekend', 'is_holiday']
    future_data['predicted_price'] = model.predict(future_data[features])

    return future_data

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        origin = data['origin']
        destination = data['destination']
        departure_date = datetime.strptime(data['date'], '%Y-%m-%d')

        logging.info(f"Received prediction request for {origin}-{destination} departing on {departure_date}")

        df = update_data(origin, destination, departure_date)

        df = engineer_features(df)
        model, train_mae, test_mae = train_model(df)

        logging.info(f"Model trained. Estimated price accuracy: ±${test_mae:.2f} (based on test data)")

        today = datetime.now()
        future_prices = predict_prices(model, today, departure_date, origin, destination)

        # Filter out past dates
        future_prices = future_prices[future_prices['departure'] >= today]

        result = {
            'dates': future_prices['departure'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': future_prices['predicted_price'].tolist(),
            'average': float(future_prices['predicted_price'].mean()),
            'min': float(future_prices['predicted_price'].min()),
            'max': float(future_prices['predicted_price'].max()),
            'accuracy': float(test_mae)
        }

        # Calculate best days to book (lowest prices)
        best_days = future_prices.nsmallest(5, 'predicted_price')[['departure', 'predicted_price']]
        result['best_days'] = [
            {'date': d.strftime('%Y-%m-%d'), 'price': float(p)}
            for d, p in zip(best_days['departure'], best_days['predicted_price'])
        ]

        return jsonify(result)

    except ValueError as ve:
        logging.error(f"ValueError in predict route: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Unexpected error in predict route: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    app.logger.info("Health check endpoint called")
    return jsonify({"status": "healthy"}), 200

# Root route
@app.route('/')
def root():
    return jsonify({"message": "Flight Price Prediction API is running"}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
