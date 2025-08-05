# PromptMask

Keep your secrets while chatting with AI. `PromptMask` uses a locally deployed LLM (like Ollama) to automatically redact your sensitive data *before* sending it to a third-party AI service (like OpenAI) and un-redacts the response.

## Features

-   **Client-Side Privacy**: Your sensitive data never leaves your machine.
-   **Seamless Integration**: Drop-in wrapper for the `openai` Python library.
-   **Intelligent Masking**: Uses a local LLM to find PII, credentials, and other sensitive info in unstructured text.
-   **Highly Configurable**: Define what's sensitive and how it's masked via a simple TOML file.
-   **Flexible Usage**: Use it as a Python library, an OpenAI adapter, or a standalone Web API.

## Installation

```bash
pip install promptmask
```

For the optional web API server:

```bash
pip install "promptmask[web]"
```

## Quickstart: `OpenAIMasked` Adapter

This is the easiest way to get started. It automatically masks requests and unmasks responses.

**Prerequisites**:
1.  Have a local LLM running and accessible via an OpenAI-compatible API (e.g., [Ollama](https://ollama.com/)).
2.  Set your remote AI provider's API key (e.g., `OPENAI_API_KEY`).

```python
import os
from promptmask import OpenAIMasked

# This client behaves just like the original OpenAI client
# but with automatic masking and unmasking.
client = OpenAIMasked(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # Optionally, provide a custom config for PromptMask
    # promtmask_config={"general": {"verbose": True}}
)

# Your sensitive data will be masked before this call
# and unmasked in the response.
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "user", "content": "My API key is sk-12345ABCDE and my user ID is 'johndoe'. Please tell me a joke about them."}
    ]
)

print(response.choices[0].message.content)
# The response from OpenAI will have the original values restored.
```

## How It Works

1.  **Intercept**: When you make an API call (e.g., `client.chat.completions.create`), `PromptMask` intercepts it.
2.  **Mask**: It sends your prompt content to your **local LLM** with a specialized system prompt, asking it to identify and create masks for sensitive data (e.g., `{"sk-12345ABCDE": "${API_KEY_1}"}`).
3.  **Replace**: It replaces the sensitive data in your original prompt with the generated masks.
4.  **Forward**: It sends the now-safe, *masked* prompt to the remote AI service (e.g., OpenAI).
5.  **Unmask**: When the response arrives, it performs a simple find-and-replace to restore the original data, ensuring the final output is coherent.

## Configuration

`PromptMask` looks for a `promptmask.config.user.toml` in the directory you run your script from. You can use it to override the defaults.

Copy `promptmask.config.default.toml` and modify it to your needs. For example, you can change the local LLM endpoint or tweak the system prompt for better masking performance.

## Web API Server

If you installed the `web` extras, you can run:

```bash
promptmask-web
```

The server will be available at `http://localhost:8000`. Check out the interactive documentation at `http://localhost:8000/docs`.