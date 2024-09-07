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


## API Integration
- **Amadeus API**: Used for fetching real-time flight data. Ensure your Amadeus API credentials are correctly set in the Streamlit secrets.
- **OpenAI API**: Powers the AI tourism advice feature. Verify that your OpenAI API key is properly configured.

## Machine Learning Model
- Utilizes Gradient Boosting Regressor for price prediction
- Features engineered include day of week, month, days until flight, etc.
- Model is retrained periodically with newly acquired data

## AI Tourism Advice
- Leverages OpenAI's GPT-3.5-turbo model
- Provides detailed information about destinations, attractions, and cultural insights
- Responses are generated in real-time based on user input

## Data Storage and Management
- Google Cloud Storage is used for storing historical flight data
- Implements caching mechanism to reduce API calls and improve performance
- Data is updated regularly to ensure accuracy of predictions

## Troubleshooting
- **API Key Issues**: Ensure all API keys in `.streamlit/secrets.toml` are correct and up-to-date
- **Data Loading Errors**: Check your internet connection and GCS bucket permissions
- **Model Prediction Failures**: Verify the integrity of the saved model file and input data format

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
[MIT License]

## Acknowledgements
- Developed with ❤️ for Tanner & Jill's wedding in Italy, 2025
- Thanks to Amadeus and OpenAI for their powerful APIs
- Streamlit for making web app development in Python a breeze
