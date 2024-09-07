# Flight Price Predictor and AI Tourism Advice

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Usage Guide](#usage-guide)
6. [Project Structure](#project-structure)
7. [API Integration](#api-integration)
8. [Machine Learning Model](#machine-learning-model)
9. [AI Tourism Advice](#ai-tourism-advice)
10. [Data Storage and Management](#data-storage-and-management)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [License](#license)
14. [Acknowledgements](#acknowledgements)

## Overview
This Streamlit application is designed to assist travelers planning their trip to Italy for Tanner & Jill's wedding in 2025. It combines flight price prediction capabilities with AI-powered tourism advice, offering a comprehensive tool for travel planning.

## Features
- **Flight Price Prediction**: Utilizes historical data and machine learning to forecast flight prices.
- **Best Booking Days**: Identifies and suggests the most cost-effective days to book flights.
- **Interactive Data Visualization**: Displays predicted prices through an interactive graph using Plotly.
- **AI Tourism Advice**: Generates personalized travel recommendations using OpenAI's GPT model.
- **Data Caching**: Implements efficient data storage and retrieval using Google Cloud Storage.
- **Real-time Flight Data**: Integrates with Amadeus API for up-to-date flight information.

## Technologies Used
- **Python**: Primary programming language
- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data manipulation and numerical computations
- **Scikit-learn**: Machine learning model implementation
- **Plotly**: Interactive data visualization
- **Amadeus API**: Real-time flight data retrieval
- **Google Cloud Storage**: Data storage and management
- **OpenAI API**: AI-powered tourism advice generation

## Setup and Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/flight-price-predictor.git
   cd flight-price-predictor
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up Streamlit secrets:
   - Create a `.streamlit/secrets.toml` file
   - Add your API keys and credentials:
     ```toml
     AMADEUS_CLIENT_ID = "your_amadeus_client_id"
     AMADEUS_CLIENT_SECRET = "your_amadeus_client_secret"
     OPENAI_API_KEY = "your_openai_api_key"
     gcs_bucket_name = "your_gcs_bucket_name"
     
     [gcp_service_account]
     # Your GCP service account JSON key here
     ```

5. Run the application:
   ```
   streamlit run app.py
   ```

## Usage Guide
1. **Flight Price Prediction**:
   - Enter the origin airport code
   - Enter the destination airport code (in Italy)
   - Select the outbound flight date
   - Click "Predict Prices" to view forecasts and recommendations

2. **AI Tourism Advice**:
   - Scroll to the "AI Tourism Advice" section
   - Enter a destination
   - Click "Get Tourism Advice" for personalized recommendations

## Project Structure
