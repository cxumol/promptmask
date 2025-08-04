# src/promptmask/core.py

import json
import string
from typing import List, Dict, Tuple, AsyncGenerator, Generator
from openai import OpenAI, AsyncOpenAI, APITimeoutError

from .config import load_config
from .utils import _btwn, logger, is_dict_str_str

class PromptMask:
    def __init__(self, config: dict = {}, config_file: str =  ""):
        """
        Initializes the PromptMask instance.

        Args:
            config (dict, optional): A dictionary to override default settings.
            config_file (str, optional): Path to a custom TOML config file.
        """
        self.config = load_config(config, config_file)
        
        self.client = OpenAI(base_url=self.config["llm_api"]["base"], api_key=self.config["llm_api"]["key"])
        self.async_client = AsyncOpenAI(base_url=self.config["llm_api"]["base"], api_key=self.config["llm_api"]["key"])

        # Auto-detect model if not specified
        if not self.config["llm_api"].get("model"):
            try:
                models = self.client.models.list()
                if models.data:
                    self.config["llm_api"]["model"] = models.data[0].id
                    logger.debug(f"Auto-selected local model: {self.config['llm_api']['model']}")
                else:
                    raise ValueError("No models found at the local LLM API endpoint.")
            except Exception as e:
                logger.error(f"Failed to auto-detect a model from {self.config['llm_api']['base']}. Please specify a model in your config. Error: {e}")
                raise

    def _build_mask_prompt(self, text: str) -> List[Dict[str, str]]:
        """Constructs the full prompt for the local masking LLM."""
        cfg = self.config
        user_content = string.Template(cfg["prompt"]["user_template"]).safe_substitute(text_to_mask=text)
        
        messages = [{"role": "system", "content": cfg["prompt"]["system_template"]}]
        messages.extend(cfg["prompt"]["examples"])
        messages.append({"role": "user", "content": user_content})
        
        return messages

    def _parse_mask_response(self, response_content: str) -> Dict[str, str]:
        """Parses the local LLM response to extract the mask map."""
        try:
            json_str = _btwn(response_content, "{", "}")
            logger.debug("json_str::",json_str)
            mask_map = json.loads(json_str)
            if not is_dict_str_str(mask_map):
                raise TypeError("Mask map should be a dictionary mapping strings to strings.")
            # Ensure 1:1 mapping by reversing the map to check for duplicate masks
            reversed_map = {v: k for k, v in mask_map.items()} #raise TypeError if v is unhashable
            if len(reversed_map) != len(mask_map):
                logger.warning("Duplicate masks detected in LLM response. The result might be inconsistent.")
            return mask_map
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse mask response: {e}\nResponse: {response_content}")
            return {"err":type(e).__name__}

    def _oai_chat_comp(self, messages:str) -> str:
        try:
            completion = self.client.chat.completions.create(
            model=self.config["llm_api"]["model"],
            messages=messages,
            temperature=0.0,
            timeout=self.config["llm_api"]["timeout"],
        )
            return completion.choices[0].message.content
        except APITimeoutError as e:
            return f'{"err":"{type(e).__name__}"}'

    # --- Synchronous Methods ---

    def mask_str(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Masks a single string."""
        if not text:
            return "", {}
            
        messages = self._build_mask_prompt(text)
        logger.debug(f"Message sending to local LLM: {messages}")

        response_content = self._oai_chat_comp(messages)
        logger.debug(f"Mask mapping by local LLM (length: {len(response_content)}): {response_content}")

        mask_map = self._parse_mask_response(response_content)
        
        masked_text = text
        for original, mask in mask_map.items():
            masked_text = masked_text.replace(original, mask)
        
        return masked_text, mask_map

    def mask_messages(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Masks 'content' in a list of chat messages."""
        # We only mask 'user' and 'assistant' roles to avoid corrupting system prompts.
        text_to_mask = "\n".join([m["content"] for m in messages if m.get("role") in ["user", "assistant"] and m.get("content")])
        
        if not text_to_mask.strip():
            return messages, {}
            
        _, mask_map = self.mask_str(text_to_mask)
        
        masked_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if new_msg.get("content") and new_msg.get("role") in ["user", "assistant"]:
                content = new_msg["content"]
                for original, mask in mask_map.items():
                    content = content.replace(original, mask)
                new_msg["content"] = content
            masked_messages.append(new_msg)
            
        return masked_messages, mask_map

    def unmask_str(self, text: str, mask_map: Dict[str, str]) -> str:
        """Unmasks a single string using the provided map."""
        unmasked_text = text
        for original, mask in mask_map.items():
            unmasked_text = unmasked_text.replace(mask, original)
        return unmasked_text

    def unmask_messages(self, messages: List[Dict[str, str]], mask_map: Dict[str, str]) -> List[Dict[str, str]]:
        """Unmasks 'content' in a list of chat messages."""
        unmasked_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if new_msg.get("content"):
                new_msg["content"] = self.unmask_str(new_msg["content"], mask_map)
            unmasked_messages.append(new_msg)
        return unmasked_messages

    def unmask_stream(self, stream: Generator, mask_map: Dict[str, str]) -> Generator:
        """Wraps a streaming response to unmask content on-the-fly."""
        buffer = ""
        # Create an inverted map for efficient replacement
        inverted_map = {mask: original for original, mask in mask_map.items()}

        for chunk in stream:
            # Assuming chunk is an OpenAI-like stream object with choices[0].delta.content
            content = chunk.choices[0].delta.content or ""
            buffer += content
            
            # Simple, non-overlapping unmasking
            # A more robust solution might use regex with word boundaries
            # but this is a good start.
            unmasked_chunk = buffer
            for mask, original in inverted_map.items():
                unmasked_chunk = unmasked_chunk.replace(mask, original)

            # Yield what's been unmasked and keep the rest in buffer
            # This is tricky; a simple approach is to yield all and clear buffer
            chunk.choices[0].delta.content = unmasked_chunk
            yield chunk
            buffer = ""

    # --- Asynchronous Methods ---

    async def async_mask_str(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Async version of mask_str."""
        if not text:
            return "", {}
            
        messages = self._build_mask_prompt(text)
        logger.debug(f"Async masking request: {messages}")
            
        response_content = self._oai_chat_comp(messages)
        logger.debug(f"Async masking response: {response_content}")
            
        mask_map = self._parse_mask_response(response_content)
        
        masked_text = text
        for original, mask in mask_map.items():
            masked_text = masked_text.replace(original, mask)
            
        return masked_text, mask_map

    async def async_mask_messages(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Async version of mask_messages."""
        text_to_mask = "\n".join([m["content"] for m in messages if m.get("role") in ["user", "assistant"] and m.get("content")])
        
        if not text_to_mask.strip():
            return messages, {}
            
        _, mask_map = await self.async_mask_str(text_to_mask)
        
        masked_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if new_msg.get("content") and new_msg.get("role") in ["user", "assistant"]:
                content = new_msg["content"]
                for original, mask in mask_map.items():
                    content = content.replace(original, mask)
                new_msg["content"] = content
            masked_messages.append(new_msg)
            
        return masked_messages, mask_map

    async def async_unmask_stream(self, stream: AsyncGenerator, mask_map: Dict[str, str]) -> AsyncGenerator:
        """Async wrapper for unmasking a stream."""
        buffer = ""
        inverted_map = {mask: original for original, mask in mask_map.items()}
        async for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            buffer += content
            
            unmasked_chunk = buffer
            for mask, original in inverted_map.items():
                unmasked_chunk = unmasked_chunk.replace(mask, original)

            chunk.choices[0].delta.content = unmasked_chunk
            yield chunk
            buffer = ""