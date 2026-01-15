
import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI

def test_azure_connection():
    # Load environment variables
    # Explicitly point to studio/backend/.env since it's not in the root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "studio", "backend", ".env")
    
    if os.path.exists(env_path):
        print(f"Loading .env from: {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        # Fallback to default search
        print("Searching for .env in current directory...")
        load_dotenv()

    # Get credentials from environment or defaults
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

    print("--- Azure OpenAI Connection Test ---")
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment}")
    print(f"API Version: {api_version}")
    
    if not api_key or not endpoint:
        print("\n❌ Error: Missing credentials.")
        print("Please ensure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set in your .env file.")
        return

    try:
        # Initialize the client
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

        print("\nSending test request...")
        
        # Create a simple chat completion request
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Please reply with 'Connection successful!' if you receive this."}
            ],
            max_tokens=50
        )

        # Print the response
        content = response.choices[0].message.content
        print(f"\n✅ Success! Response from Azure OpenAI:\n{content}")

    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_azure_connection()
