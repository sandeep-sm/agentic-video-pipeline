"""Quick script to list available Gemini models and find the best one."""
import os
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
for model in client.models.list():
    name = model.name
    if "gemini" in name.lower():
        print(name)
