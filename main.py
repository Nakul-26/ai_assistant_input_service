# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import json

# Import the Hugging Face InferenceClient
from huggingface_hub import InferenceClient

app = FastAPI(title="Input Service for AI Assistant")

# Pydantic models for data validation
class InputPayload(BaseModel):
    text: str

class Intent(BaseModel):
    task: str
    requirements: list[str]
    platform: str

# -----------------
# Environment Variables
# -----------------
# Get your Hugging Face API token from an environment variable for security
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN environment variable not set")

# -----------------
# Hugging Face LLM Client Setup
# -----------------
# Initialize the InferenceClient with your API token
# You can use a model like Mistral 7B, which is a good balance of performance and size
# The InferenceClient automatically handles API calls to the hosted model
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_API_TOKEN)

def extract_intent_from_llm(user_input: str) -> Intent:
    """Uses a hosted LLM to extract intent from user input."""
    # This is a key part of the service. We craft a detailed prompt to get a structured response.
    prompt = f"""You are an expert at parsing user commands for an AI assistant.
    Given the user input, extract the primary task, a list of requirements, and the target platform.
    
    Return the response as a single, valid JSON object. Do not include any other text.
    
    User Input: "{user_input}"
    
    Expected JSON format:
    {{
      "task": "string",
      "requirements": ["string"],
      "platform": "string"
    }}

    Example:
    User Input: "build a login page with a password field for a web app"
    {{
      "task": "Build login page",
      "requirements": ["password field"],
      "platform": "Web"
    }}
    """
    
    try:
        # Use the InferenceClient to generate a text response based on the prompt
        response = client.text_generation(prompt, max_new_tokens=256, temperature=0.1, stop=["\n"])
        
        # Strip any extra text and load the JSON
        response_text = response.strip()
        intent_data = json.loads(response_text)
        
        # Validate the parsed data against our Pydantic model
        validated_intent = Intent(**intent_data)
        return validated_intent
        
    except Exception as e:
        # If the LLM doesn't return valid JSON or another error occurs, raise an exception
        raise HTTPException(status_code=500, detail=f"Failed to parse intent from LLM: {e}")

# -----------------
# API Endpoint
# -----------------

@app.get("/")
async def root():
    return {"message": "Input Service is running!"}

@app.post("/parse-intent", response_model=Intent)
async def parse_intent_endpoint(payload: InputPayload):
    """
    Receives user input and returns a structured intent.
    This is the main API endpoint for the Input Service.
    """
    print(f"Received input: {payload.text}")
    
    # Use the helper function to get the structured intent
    intent = extract_intent_from_llm(payload.text)
    
    # In a real-world scenario, you would initialize the session context
    # and pass it to the next service, but for the MVP, we just return the intent.
    
    return intent