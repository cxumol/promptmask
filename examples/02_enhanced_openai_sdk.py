# examples/enhanced_openai_sdk.py

import os
from promptmask.adapter.openai import OpenAIMasked

# NOTE: This example requires two things to run:
# 1. A local LLM server (e.g., ollama, llama-cpp, vllm, etc.) running and exposing an
#    OpenAI-compatible API at http://localhost:port/v1. This local LLM is used
#    by PromptMask for the PII masking process.
# 2. Your OPENAI_API_KEY environment variable must be set.
#    - On Linux/macOS: export OPENAI_API_KEY="sk-..."
#    - On Windows: set OPENAI_API_KEY="sk-..."
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://api.openai.com/v1")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4.1-mini")

def main():
    """Demo the drop-in replacement: OpenAI -> OpenAIMasked"""
    
    user_prompt = """Hello my name is Son Alterman. Call me at 789123456, or email me at sona@oftenai.com
My daddy is Johnson Hung, and his contact information is 456789123 (phone) and johnsonhung@mvidia.com (email). We both use irelyonGPU as our password.

Please rewrite the information in CSV format with following CSV headers:  
Person Name, Phone Number, Email Address, Password"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
        {"role": "user", "content": user_prompt},
    ]

    pm_config = {
        # umcomment the following lines or create your promptmask.config.user.toml
        # "llm_api": {
        #     "base": "http://localhost:11434/v1",  # <-- URL of your local LLM
        #     "key": "not-needed",
        #     "model": ""  # Auto-detected if not specified
        # },
        # "general": {"verbose": True}, # To debug
    }
    
    print(f"OPENAI_API_BASE: {OPENAI_API_BASE}\nOPENAI_API_MODEL: {OPENAI_API_MODEL}")
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable not set. \n" 
        "Please set it before running the script.")
        return

    # The enhanced openai.OpenAI client
    client = OpenAIMasked(
        base_url=OPENAI_API_BASE,
        api_key=os.getenv("OPENAI_API_KEY"),
        promptmask_config=pm_config # use it as a normal OpenAI client, with additional optional `promptmask_config`
    )

    print("\n--- 1. NON-STREAMING Regular Call ---")
    print("Making a regular API call. The prompt will be masked before sending to Cloud AI...")
    try:
        response = client.chat.completions.create(
            model=OPENAI_API_MODEL,
            messages=messages,
            stream=False,
        )
        # The adapter adds the `original_content` attribute to show the masked text
        print("\n[Masked content sent to OpenAI]:")
        print(response.choices[0].message.original_content)
        
        # The final response content is automatically unmasked
        print("\n[Final, unmasked response]:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your OPENAI_API_KEY is valid and the local LLM server is running at the specified address.")

    print("\n--- 2. STREAMING ---")
    print("Making a streaming API call. The response will be unmasked on-the-fly...")
    try:
        stream = client.chat.completions.create(
            model=OPENAI_API_MODEL,
            messages=messages,
            stream=True,
        )
        full_original_content = ""
        print("\n[Final, unmasked streamed response]:")
        for chunk in stream:
            if not (chunk.choices and (delta := chunk.choices[0].delta)):
                continue
            full_original_content += delta.original_content or ""
            content = delta.content
            if content:
                print(content, end="", flush=True)
        print("\n")
        print(f"[Original, masked stream response]: {full_original_content}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()