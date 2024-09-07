import os
import openai
import toml

# Load secrets from the secrets.toml file
secrets = toml.load('.streamlit/secrets.toml')

# Access the OpenAI API key using the correct key name
openai.api_key = secrets['openai']['OPENAI_API_KEY']  # Use 'OPENAI_API_KEY' instead of 'api_key'

def get_ai_tourism_advice(destination):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant providing detailed advice about tourist attractions."},
                {"role": "user", "content": f"Provide detailed information about {destination}, Italy, including must-visit attractions and cultural insights."}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, I couldn't retrieve tourism advice at the moment. Please try again later."

def main():
    destination = input("Enter a destination in Italy: ")
    advice = get_ai_tourism_advice(destination)
    print("\nAI Tourism Advice:")
    print(advice)

if __name__ == "__main__":
    main()
