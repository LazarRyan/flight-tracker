import streamlit as st
import pandas as pd
import sys
import os

# Add src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import functions from the provided scripts
from fetch_data import fetch_data
from load_data import load_data
from clean_data import clean_data
from train_predict import train_model, make_prediction
from run_app import display_results

def main():
    st.title('Flight Tracker')

    st.sidebar.title('Settings')
    option = st.sidebar.selectbox('Choose a task', ['Fetch Data', 'Load Data', 'Clean Data', 'Train Model', 'Predict', 'Show Results'])

    if option == 'Fetch Data':
        st.header('Fetch Data')
        if st.button('Fetch Data'):
            df = fetch_data()
            st.success('Data fetched successfully!')
            st.write(df)

    elif option == 'Load Data':
        st.header('Load Data')
        if st.button('Load Data'):
            df = load_data()
            st.success('Data loaded successfully!')
            st.write(df)

    elif option == 'Clean Data':
        st.header('Clean Data')
        if st.button('Clean Data'):
            df = load_data()
            df = clean_data(df)
            st.success('Data cleaned successfully!')
            st.write(df)

    elif option == 'Train Model':
        st.header('Train Model')
        if st.button('Train Model'):
            df = load_data()
            df = clean_data(df)
            model = train_model(df)
            st.success('Model trained successfully!')

    elif option == 'Predict':
        st.header('Make Predictions')
        if st.button('Make Predictions'):
            df = load_data()
            df = clean_data(df)
            model = train_model(df)
            predictions = make_prediction(model)
            st.write(predictions)
            st.success('Predictions made successfully!')

    elif option == 'Show Results':
        st.header('Results')
        df = load_data()
        df = clean_data(df)
        model = train_model(df)
        predictions = make_prediction(model)
        display_results(predictions)

if __name__ == '__main__':
    main()

