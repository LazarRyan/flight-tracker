# data-management-service/app.py

from flask import Flask, request, jsonify
from google.cloud import storage
from google.oauth2 import service_account
from io import StringIO
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize GCS client
credentials = service_account.Credentials.from_service_account_file(
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
)
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.bucket(os.environ.get("GCS_BUCKET_NAME"))

def get_data_filename(origin, destination):
    return f"flight_prices_{origin}_{destination}.csv"

@app.route('/load', methods=['GET'])
def load_data():
    origin = request.args.get('origin')
    destination = request.args.get('destination')
    filename = get_data_filename(origin, destination)
    blob = bucket.blob(filename)
    
    if blob.exists():
        try:
            content = blob.download_as_text()
            df = pd.read_csv(StringIO(content))
            df['departure'] = pd.to_datetime(df['departure'])
            logging.info(f"Loaded {len(df)} records for {origin} to {destination}")
            return jsonify(df.to_dict(orient='records')), 200
        except Exception as e:
            logging.error(f"Error loading data for {origin} to {destination}: {str(e)}")
            return jsonify({"error": "Failed to load data"}), 500
    else:
        return jsonify({"message": "No data found"}), 404

@app.route('/save', methods=['POST'])
def save_data():
    data = request.json
    df = pd.DataFrame(data)
    origin = df['origin'].iloc[0]
    destination = df['destination'].iloc[0]
    filename = get_data_filename(origin, destination)
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob = bucket.blob(filename)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    logging.info(f"Saved {len(df)} records for {origin} to {destination}")
    return jsonify({"message": "Data saved successfully"}), 200

@app.route('/should_call_api', methods=['GET'])
def should_call_api():
    origin = request.args.get('origin')
    destination = request.args.get('destination')
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
            return jsonify({"should_call": False, "time_until_next_call": str(time_until_next_call)}), 200

    api_calls[route_key] = now.isoformat()
    blob.upload_from_string(json.dumps(api_calls), content_type="application/json")
    return jsonify({"should_call": True}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
