# flight_tracker.py

import subprocess
import os
import streamlit as st
from time import sleep

# Define the base directory and paths to the scripts
base_dir = os.path.expanduser("~/flight-tracker/src")
scripts = [
    os.path.join(base_dir, "fetch_data.py"),
    os.path.join(base_dir, "load_data.py"),
    os.path.join(base_dir, "clean_data.py"),
    os.path.join(base_dir, "train_predict.py")
]

# Function to run each script with progress
def run_script_with_progress(script_path, progress_bar, status_text, index, total):
    try:
        status_text.text(f"Running {script_path}...")
        result = subprocess.run(["python3", script_path], check=True, capture_output=True, text=True)
        progress_bar.progress((index + 1) / total)
        st.text(f"Output of {script_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        st.text(f"Error running {script_path}:\n{e.stderr}")

# Streamlit interface
st.title("Flight Tracker")

if st.button("Run All Data Processing Scripts"):
    with st.spinner("Running data processing scripts..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_scripts = len(scripts)
        for index, script in enumerate(scripts):
            run_script_with_progress(script, progress_bar, status_text, index, total_scripts)
        status_text.text("All data processing scripts have been run successfully.")
        st.success("Data processing completed.")

st.write("After running the data processing scripts, you can launch the Streamlit application to visualize the data.")
if st.button("Launch Streamlit Application"):
    run_app_command = "streamlit run src/run_app.py"
    subprocess.Popen(run_app_command, shell=True)
    st.write("Streamlit application is now running. Check the terminal or your Streamlit Cloud deployment.")

