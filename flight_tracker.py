# flight_tracker.py

import subprocess
import os
import streamlit as st

# Define the base directory and paths to the scripts
base_dir = os.path.expanduser("~/flight-tracker/src")
scripts = [
    os.path.join(base_dir, "fetch_data.py"),
    os.path.join(base_dir, "load_data.py"),
    os.path.join(base_dir, "clean_data.py"),
    os.path.join(base_dir, "train_predict.py"),
    # Exclude run_app.py as it should be run separately with Streamlit
]

# Function to run each script
def run_script(script_path):
    try:
        result = subprocess.run(["python3", script_path], check=True, capture_output=True, text=True)
        st.text(f"Output of {script_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        st.text(f"Error running {script_path}:\n{e.stderr}")

# Streamlit interface
st.title("Flight Tracker")

if st.button("Run All Scripts (except run_app.py)"):
    for script in scripts:
        run_script(script)

st.write("To run the Streamlit application, execute the following command separately:")
st.code("streamlit run src/run_app.py")

