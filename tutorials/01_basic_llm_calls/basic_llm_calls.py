"""
Tutorial 1: Basic LLM Calls

Goal: Connect and call LLMs from Anthropic (Claude), Google (Gemini), and OpenAI (GPT).

Requirements:
- Python 3.13+
- Install: requests, openai, anthropic, google-generativeai
- Set API keys as environment variables or in a .env file:
  - ANTHROPIC_API_KEY
  - GOOGLE_API_KEY
  - OPENAI_API_KEY

Outputs are saved to console and basic_llm_outputs.txt.
"""

import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime

OUTPUT_FILE = "basic_llm_outputs.txt"
def save_output(model_name, prompt, response):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {model_name}\nPrompt: {prompt}\nResponse: {response}\n\n")

# Anthropic (Claude)
try:
    import anthropic
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("Set ANTHROPIC_API_KEY in your environment.")
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    prompt = "What is the capital of France?"
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.content[0].text
    print("Anthropic Response:", result)
    save_output("Anthropic", prompt, result)
except Exception as e:
    print("Anthropic Error:", e)

# Google Gemini
try:
    import google.generativeai as genai
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Set GOOGLE_API_KEY in your environment.")
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    prompt = "What is the capital of France?"
    response = model.generate_content(prompt)
    result = response.text
    print("Gemini Response:", result)
    save_output("Gemini", prompt, result)
except Exception as e:
    print("Gemini Error:", e)

# OpenAI GPT
try:
    import openai
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Set OPENAI_API_KEY in your environment.")
    openai_client = openai.OpenAI(api_key=openai_api_key)
    prompt = "What is the capital of France?"
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    print("OpenAI Response:", result)
    save_output("OpenAI", prompt, result)
except Exception as e:
    print("OpenAI Error:", e)

print("\nAll outputs saved to basic_llm_outputs.txt")
