from google import genai
import os
from dotenv import load_dotenv

load_dotenv("app/.env")
api_key = os.getenv("GEMINI_API_KEY_QUERY")
client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="Hello"
    )
    print("Response received OK")
except Exception as e:
    print(f"Error: {e}")
