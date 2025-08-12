# PromptMask

A local-first privacy layer for Large Language Models.

> Cloud AI is **smart** but sacrifices privacy.   
Local AI **keeps your secret** but is dumb.  
What if we can combine the advantages of both sides?

[![PyPI version](https://badge.fury.io/py/promptmask.svg)](https://badge.fury.io/py/promptmask)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/promptmask.svg)](https://pypi.org/project/promptmask/)

PromptMask ensures your private data never leaves your machines. It redacts and un-redacts sensitive data locally, so that only anonymized data is sent to third-party AI services.

## Table of Contents

* [Table of Contents](#table-of-contents)
* [How It Works](#how-it-works)
* [Quickstart](#quickstart)
    + [Prerequisites](#prerequisites)
    + [For General Users: local OpenAI-compatible API Gateway](#for-general-users-local-openai-compatible-api-gateway)
    + [For Python Developers: OpenAIMasked](#for-python-developers-openaimasked)
* [Configuration](#configuration)
* [Advanced Usage: PromptMask Class](#advanced-usage-promptmask-class)
* [Web Server: WebUI & API](#web-server-webui-api)
* [Development & Contribution](#development-contribution)
* [License](#license)

## How It Works

The core principle is to use a trusted (local) model as a "privacy filter" for a powerful, remote model. The process is fully automated.

<!-- ![diagram_placeholder](TBD.svg) -->

## Quickstart

### Prerequisites

Ensure you have a local LLM running with an OpenAI-compatible API endpoint. [Ollama](https://ollama.com/) is a popular and straightforward option. By default, `PromptMask` will attempt to connect to `http://localhost:11434/v1`.

Other options to run a local OpenAI-compatible LLM API include llama.cpp and vLLM.

### For General Users: local OpenAI-compatible API Gateway

You can point any existing tool or application at the local gateway. It's the seamless way to add `PromptMask` layer without coding in Python.

1.  **Install promptmask-web via pip:**
    ```bash
    pip install "promptmask[web]"
    ```

2.  **Run the web server:**
    ```bash
    promptmask-web
    ```
    The gateway is now running at `http://localhost:8000`.

3.  **Use the gateway endpoint:**
    Simply replace the official OpenAI API base URL with the local gateway's URL in your tool of choice.

    For example, using `curl`:

    ```bash
    curl http://localhost:8000/gateway/v1/chat/completions \
      -H "Authorization: Bearer $YOUR_OPENAI_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gpt-99-ultra",
        "messages": [
          {
            "role": "user",
            "content": "My name is Ho Shih-Chieh and my appointment ID is Y1a2e87. I booked a dental appointment on Oct 26, but I have to cancel for a meeting. Please help me write a cancellation request email in French."
          }
        ]
      }'
    ```
    Your sensitive data (`Ho Shih Chieh`, `Y1a2e87`) will be redacted before being sent to OpenAI, and then restored in the final response.

    If you are using other cloud AI providers, such as Google Gemini, you need to add `web.upstream_oai_api_base` to your config file (more detail on [configuration](#configuration) section)

    ```toml
    [web]
    upstream_oai_api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
    ```

### For Python Developers: OpenAIMasked

The `OpenAIMasked` class is a drop-in replacement for the standard `openai.OpenAI` client.

1.  **Install the base package:**
    ```bash
    pip install promptmask
    ```

2.  **Mask the OpenAI client in your code:**
    The adapter automatically handles masking/unmasking for standard and streaming requests.

    Simply replace the standard `openai.OpenAI` client as follows:

    ```python
    # from openai import OpenAI
    from promptmask import OpenAIMasked
    # client = OpenAI()
    client = OpenAIMasked()
    ```

    Full example:

    ```python
    from promptmask import OpenAIMasked

    # This client has the same interface as openai.OpenAI, but with automatic privacy redaction.
    client = OpenAIMasked(base_url="https://api.cloud-ai-service.example.com/v1") # reads OPENAI_API_KEY from environment variables by default.

    # --- Standard Request ---
    response = client.chat.completions.create(
        model="gpt-100-pro",
        messages=[
            {"role": "user", "content": "My user ID is johndoe and my phone number is 4567890. Please help me write an application letter."}
        ]
    )
    # The response content is automatically unmasked.
    print(response.choices[0].message.content)

    # --- Streaming Request ---
    stream = client.chat.completions.create(
        model="gpt-101-turbo-mini",
        stream=True,
        messages=[
            {"role": "user", "content": "My patient, Jensen Huang (Patient ID: P123456789), is taking metformin and is experiencing nausea. What are the common side effects and management strategies?"}
        ]
    )

    # The stream chunks are unmasked on-the-fly.
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
    ```

## Configuration

To customize, create a `promptmask.config.user.toml` file in your working directory. For example, to change the categories of data to mask:

```toml
# promptmask.config.user.toml

[llm_api]
# Specify a particular local model to use for masking
model = "qwen2.5:7b"

# Define what data is considered sensitive.
[sensitive]
# Override the default one
include = "personal ID and passwords"
```

Check [promptmask.config.default.toml](src/promptmask/promptmask.config.default.toml) for a full config file example. 

Environment variables can also be used to override specific settings:
*   `LOCALAI_API_BASE`: The Base URL for your local LLM's API (e.g., `http://192.168.1.234:11434/v1`).
*   `LOCALAI_API_KEY`: The API key for your local LLM, if required.


<details>
<summary>Configuration Priority Hierarchy</summary>

`PromptMask` is configured through a hierarchy of sources, from highest to lowest priority:

0. `LOCALAI_API_BASE` and `LOCALAI_API_KEY` environment variables.
1.  A `dict` passed directly to the `PromptMask` constructor (`config` parameter).
2.  A path to a TOML file (`config_file` parameter).
3.  A `promptmask.config.user.toml` file in the current working directory.
4.  The packaged `promptmask.config.default.toml`.

</details>

## Advanced Usage: PromptMask Class

For more granular control, you can import the `PromptMask` class directly to perform masking and unmasking as separate steps.

```python
import asyncio # PromptMask also runs syncrounously
from promptmask import PromptMask

async def main():
    masker = PromptMask()

    original_text = "Please process the visa application for Jensen Huang, passport number A12345678."

    # 1. Mask the text
    masked_text, mask_map = await masker.async_mask_str(original_text)

    print(f"Masked Text: {masked_text}")
    # Expected output (may vary): Masked Text: Please process the visa application for ${PERSON_NAME}, passport number ${PASSPORT_NUMBER}.
    
    print(f"Mask Map: {mask_map}")
    # Expected output: Mask Map: {"Jensen Huang": "${PERSON_NAME}", "A12345678": "${PASSPORT_NUMBER}"}

    # (Imagine sending masked_text to a remote API and getting a response)
    remote_response_text = "The visa application for ${PERSON_NAME} with passport ${PASSPORT_NUMBER} is now under review."

    # 2. Unmask the response
    unmasked_response = masker.unmask_str(remote_response_text, mask_map)
    print(f"Unmasked Response: {unmasked_response}")
    # Expected output: Unmasked Response: The visa application for Jensen Huang with passport A12345678 is now under review.

if __name__ == "__main__":
    asyncio.run(main())
```

## Web Server: WebUI & API

When you run `promptmask-web` with the installed `promptmask[web]`, a full-featured web service is launched at `http://localhost:8000`. It includes:

*   A simple **Web UI**
    * to try out features including masking/unmasking and the gateway
    * A **Configuration Manager** to view and hot-reload settings.
*   Interactive API documentation (via Swagger UI) at `http://localhost:8000/docs`
    *   **API Gateway** at `/gateway/v1/chat/completions` to take care of your privacy seamlessly.
    *   Direct **Masking/Unmasking API Endpoints** (details on API documentation).
    *   Edit configuration via `/config` endpoint.

## Development & Contribution

Contributions are welcome.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cxumol/promptmask.git
    cd promptmask
    ```
2.  **Install in editable mode with development dependencies:**
    ```bash
    pip install -e ".[dev,web]"
    ```
3.  **Run tests:**
    ```bash
    pytest
    ```
4.  **Lint and format code (optional):**
    `ruff` for linting and formatting, if you want.
    ```bash
    ruff check .
    ruff format .
    ```

Please open an issue or submit a pull request for any bugs or feature proposals.

## License

PromptMask is distributed under the MIT License. See `LICENSE` for more information.