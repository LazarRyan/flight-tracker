import pandas as pd

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

if __name__ == "__main__":
    historical_csv = 'flight_prices.csv'
    historical_df = load_data(historical_csv)
    print("Data loaded successfully")

