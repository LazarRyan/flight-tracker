import streamlit as st
import openai
import os

# Try to get the API key from environment variable first (for Streamlit Cloud)
openai_api_key = os.environ.get("OPENAI_API_KEY")

# If not found in environment, try to get it from Streamlit secrets (for local development)
if not openai_api_key:
    try:
        openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    except KeyError:
        st.error("OPENAI_API_KEY not found in Streamlit secrets or environment variables. Please set it up properly.")
        openai_api_key = None

openai.api_key = openai_api_key

def get_ai_tourism_advice(destination):
    if not openai.api_key:
        return "OpenAI API key is not set. Unable to fetch tourism advice."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant providing detailed advice about tourist attractions."},
                {"role": "user", "content": f"Provide detailed information about {destination}, including must-visit attractions and cultural insights."}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Sorry, I couldn't retrieve tourism advice at the moment. Please try again later."

def main():
    st.title("🏛️ AI Tourism Advice")
    
    if not openai.api_key:
        st.warning("OpenAI API key is not set. The app will not be able to provide tourism advice.")
        st.stop()
    
    destination = st.text_input("Enter a destination:", "")
    
    if st.button("Get Tourism Advice"):
        if destination:
            with st.spinner("Fetching tourism advice..."):
                advice = get_ai_tourism_advice(destination)
                st.subheader(f"AI Tourism Advice for {destination}:")
                st.write(advice)
        else:
            st.warning("Please enter a destination.")

    st.info("This AI tourism advice is generated based on the destination you enter. For more accurate results, consider using specific city or country names.")

if __name__ == "__main__":
    main()
