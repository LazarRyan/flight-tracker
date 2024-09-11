# tourism-advice-service/app.py

from flask import Flask, request, jsonify
from openai import OpenAI
import os
import boto3
from botocore.exceptions import NoCredentialsError
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name='us-east-2'
)

def save_advice_to_s3(destination, advice):
    bucket_name = 'flightai'  # Replace with your actual bucket name
    file_name = f"{destination.lower().replace(' ', '_')}_advice.txt"
    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=advice)
        return True
    except NoCredentialsError:
        logging.error("AWS credentials not available")
        return False
    except Exception as e:
        logging.error(f"Error saving to S3: {str(e)}")
        return False

@app.route('/advice', methods=['GET'])
def get_tourism_advice():
    destination = request.args.get('destination')
    if not destination:
        return jsonify({"error": "Destination is required"}), 400

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant providing detailed advice about tourist attractions."},
                {"role": "user", "content": f"Provide detailed information about {destination}, including must-visit attractions and cultural insights."}
            ]
        )
        advice = response.choices[0].message.content
        if save_advice_to_s3(destination, advice):
            logging.info(f"Advice for {destination} saved to S3 successfully")
        else:
            logging.warning(f"Failed to save advice for {destination} to S3")
        return jsonify({"advice": advice}), 200
    except Exception as e:
        logging.error(f"Error getting tourism advice: {str(e)}")
        return jsonify({"error": "Failed to retrieve tourism advice"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
