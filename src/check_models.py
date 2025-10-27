import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("🔍 Checking available Gemini models...\n")

for m in genai.list_models():
    if "generateContent" in getattr(m, "supported_generation_methods", []):
        print("✅", m.name)

