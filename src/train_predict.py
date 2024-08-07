import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def preprocess_data(df):
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
    df['DayOfWeek'] = df['DepartureDate'].dt.dayofweek
    df['Month'] = df['DepartureDate'].dt.month
    return df

def train_model(df):
    X = df[['DayOfWeek', 'Month']]
    y = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future_prices(model, future_df):
    future_df['DayOfWeek'] = future_df['DepartureDate'].dt.dayofweek
    future_df['Month'] = future_df['DepartureDate'].dt.month
    X_future = future_df[['DayOfWeek', 'Month']]
    future_df['PredictedPrice'] = model.predict(X_future)
    return future_df

if __name__ == "__main__":
    historical_csv = 'flight_prices.csv'
    historical_df = pd.read_csv(historical_csv)
    historical_df = preprocess_data(historical_df)

    model = train_model(historical_df)
    joblib.dump(model, 'flight_price_model.pkl')

    # Predict prices for the next 365 days
    future_dates = pd.date_range(start='2024-12-02', end='2025-09-05')
    future_df = pd.DataFrame(future_dates, columns=['DepartureDate'])
    future_df = predict_future_prices(model, future_df)
    future_df.to_csv('future_prices.csv', index=False)
    print("Future prices predicted and saved to future_prices.csv")

