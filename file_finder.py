import json
import toml
from pathlib import Path
import os

def list_json_files(directory):
    json_files = list(directory.glob('*.json'))
    if not json_files:
        print(f"No JSON files found in {directory}")
        return None
    print("\nJSON files found:")
    for i, file in enumerate(json_files):
        print(f"{i+1}. {file.name}")
    return json_files

def select_file(files):
    while True:
        try:
            selection = int(input("Enter the number of the GCP key file: ")) - 1
            if 0 <= selection < len(files):
                return files[selection]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def read_json_file(file_path):
    print(f"Attempting to read: {file_path}")
    if file_path.stat().st_size == 0:
        print(f"Error: The file {file_path} is empty.")
        return None

    try:
        with file_path.open('r', encoding='utf-8') as file:
            file_content = file.read()
            print("File content (first 100 characters):")
            print(file_content[:100])
            
            try:
                return json.loads(file_content)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Error location: Line {e.lineno}, Column {e.colno}")
                print("Error message:", e.msg)
                return None
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def main():
    # Get the current working directory (should be the flight-tracker repository)
    current_dir = Path.cwd()
    cloud_stuff_path = current_dir / "Cloud_Stuff"

    print(f"Looking for JSON files in: {cloud_stuff_path}")

    if not cloud_stuff_path.exists():
        print(f"Error: The directory {cloud_stuff_path} does not exist.")
        return

    json_files = list_json_files(cloud_stuff_path)
    if not json_files:
        return

    selected_file = select_file(json_files)
    gcp_key = read_json_file(selected_file)

    if not gcp_key:
        print("Failed to read the JSON file. Please check the file content and try again.")
        return

    print("JSON file successfully parsed.")

    # Ensure the .streamlit directory exists within the repository
    streamlit_dir = current_dir / ".streamlit"
    streamlit_dir.mkdir(parents=True, exist_ok=True)

    # Get the private key and replace '\n' with actual newlines
    private_key = gcp_key['private_key'].replace('\\n', '\n')

    # Prepare the TOML data
    toml_data = {
        'gcp_service_account': {
            'type': gcp_key['type'],
            'project_id': gcp_key['project_id'],
            'private_key_id': gcp_key['private_key_id'],
            'private_key': private_key,
            'client_email': gcp_key['client_email'],
            'client_id': gcp_key['client_id'],
            'auth_uri': gcp_key['auth_uri'],
            'token_uri': gcp_key['token_uri'],
            'auth_provider_x509_cert_url': gcp_key['auth_provider_x509_cert_url'],
            'client_x509_cert_url': gcp_key['client_x509_cert_url']
        },
        'gcs_bucket': {
            'name': input("Enter your GCS bucket name: ")
        }
    }

    # Write to TOML file
    secrets_file_path = streamlit_dir / "secrets.toml"
    try:
        with secrets_file_path.open('w') as f:
            toml.dump(toml_data, f)
        print(f"TOML file created successfully at {secrets_file_path}")
    except PermissionError:
        print(f"Error: Permission denied when trying to write to {secrets_file_path}")
        print("Please check your file permissions and try again.")

if __name__ == "__main__":
    main()
