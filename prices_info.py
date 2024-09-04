import pandas as pd
import json

def safe_json_loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return None
    return x

def analyze_csv():
    try:
        # Read the CSV file
        df = pd.read_csv('flight_prices.csv', header=None, 
                         names=['date', 'price', 'itineraries', 'carriers', 'price_details'])
        
        print(f"Total rows in the file: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Basic info about each column
        for column in df.columns:
            print(f"\nColumn: {column}")
            print(f"  Data type: {df[column].dtype}")
            print(f"  Number of non-null values: {df[column].count()}")
            print(f"  Number of unique values: {df[column].nunique()}")
            
            if column in ['date', 'price']:
                print(f"  Min value: {df[column].min()}")
                print(f"  Max value: {df[column].max()}")
            
            if column == 'itineraries':
                # Try to parse the first non-null itinerary
                sample_itinerary = df['itineraries'].dropna().iloc[0] if not df['itineraries'].isnull().all() else None
                if sample_itinerary:
                    parsed_itinerary = safe_json_loads(sample_itinerary)
                    if parsed_itinerary:
                        print("  Sample itinerary structure:")
                        print(json.dumps(parsed_itinerary, indent=2)[:500] + "...")  # Print first 500 characters
        
        # Check for any completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            print(f"\nCompletely empty columns: {empty_columns}")
        
        # Check for any columns with all unique values
        unique_columns = df.columns[df.nunique() == len(df)].tolist()
        if unique_columns:
            print(f"\nColumns with all unique values: {unique_columns}")
        
        # Check for any potential delimiter issues
        with open('flight_prices.csv', 'r') as f:
            first_line = f.readline().strip()
            print(f"\nFirst line of the file:")
            print(first_line)
            print(f"Number of commas in the first line: {first_line.count(',')}")
        
    except FileNotFoundError:
        print("The file 'flight_prices.csv' was not found in the current directory.")
    except pd.errors.EmptyDataError:
        print("The file 'flight_prices.csv' is empty.")
    except Exception as e:
        print(f"An error occurred while analyzing the CSV: {str(e)}")

if __name__ == "__main__":
    analyze_csv()
