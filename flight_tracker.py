import subprocess
import os
import streamlit as st

# Define the paths to the scripts
base_dir = os.path.expanduser("~/flight-tracker/src")
scripts = [
    os.path.join(base_dir, "fetch_data.py"),
    os.path.join(base_dir, "load_data.py"),
    os.path.join(base_dir, "clean_data.py"),
    os.path.join(base_dir, "train_predict.py"),
    os.path.join(base_dir, "run_app.py")
]

# Function to run each script
def run_script(script_path):
    result = subprocess.run(["python3", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Error running {script_path}:\n{result.stderr}")
    else:
        st.text(f"Output of {script_path}:\n{result.stdout}")

# Streamlit interface
st.title("Flight Tracker")

if st.button("Run All Data Processing Scripts"):
    with st.spinner("Running data processing scripts..."):
        for script in scripts[:-1]:  # Exclude run_app.py from this loop
            run_script(script)
        st.success("All data processing scripts have been run successfully.")

if st.button("Launch Streamlit Application"):
    run_script(scripts[-1])  # Run run_app.py
    st.write("Streamlit application is now running. Check the terminal or your Streamlit Cloud deployment.")

