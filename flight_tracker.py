# flight_tracker.py
import os

# Step 1: Fetch Data
os.system('python3 src/fetch_data.py')

# Step 2: Load Data
os.system('python3 src/load_data.py')

# Step 3: Train and Predict
os.system('python3 src/train_predict.py')

# Step 4: Run Streamlit App
os.system('streamlit run src/run_app.py')

