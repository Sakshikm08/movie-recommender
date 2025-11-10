import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Test API call with UPDATED MODEL
try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say 'Hello, Groq API is working!' in a fun way",
            }
        ],
        model="llama-3.3-70b-versatile",  # ✅ UPDATED
    )
    
    print("✅ SUCCESS! Groq API is working!")
    print(f"Response: {chat_completion.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("Check your API key in .env file")
