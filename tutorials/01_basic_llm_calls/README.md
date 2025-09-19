# Tutorial 1: Basic LLM Calls

Welcome to the OneStop-AI-Suite tutorial series!

## Overview
This module demonstrates how to connect and call Large Language Models (LLMs) from the three major providers:
- Anthropic (Claude)
- Google (Gemini)
- OpenAI (GPT)

You will:
- Run minimal examples: one text prompt → one text response
- Save outputs to both the console and a text file
- Use both a Jupyter Notebook (`basic_llm_calls.ipynb`) and a Python script (`basic_llm_calls.py`)

## Prerequisites
- Python 3.9+
- Install required packages:
  ```sh
  pip install requests openai anthropic google-generativeai python-dotenv
  ```
- Set your API keys as environment variables or in a `.env` file:
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`
  - `OPENAI_API_KEY`

## Files
- `basic_llm_calls.ipynb`: Step-by-step notebook with explanations and runnable code
- `basic_llm_calls.py`: Equivalent Python script for quick execution
- `basic_llm_outputs.txt`: All model outputs are saved here for reproducibility

## How to Run
1. Add your API keys to a `.env` file or your environment
2. Run the notebook or script:
   - Notebook: Open in Jupyter and run each cell
   - Script: `python basic_llm_calls.py`
3. Check `basic_llm_outputs.txt` for saved results

## Sample Output
```
[2025-09-19 10:00:00] Anthropic
Prompt: What is the capital of France?
Response: The capital of France is Paris.

[2025-09-19 10:00:01] Gemini
Prompt: What is the capital of France?
Response: Paris

[2025-09-19 10:00:02] OpenAI
Prompt: What is the capital of France?
Response: Paris is the capital of France.
```

## Next Steps
- Try changing the prompt and rerun the cells/scripts
- Proceed to the next tutorial for RAG basics
