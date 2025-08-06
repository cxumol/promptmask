# src/promptmask/core.py

import json
import string
import asyncio
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
        self._init_config_override = config
        self._init_config_file = config_file
        self._lock = asyncio.Lock() # 添加一个锁来处理并发重载请求
        self._initialize_clients()

    def _initialize_clients(self):
        """
        Loads configuration and initializes API clients.
        This method can be called to re-initialize the instance.
        """
        logger.info("Initializing or reloading PromptMask configuration...")
        self.config = load_config(self._init_config_override, self._init_config_file)
        
        self.client = OpenAI(base_url=self.config["llm_api"]["base"], api_key=self.config["llm_api"]["key"], timeout=self.config["llm_api"]["timeout"])
        self.async_client = AsyncOpenAI(base_url=self.config["llm_api"]["base"], api_key=self.config["llm_api"]["key"], timeout=self.config["llm_api"]["timeout"])

        # Auto-detect model if not specified
        if not self.config["llm_api"].get("model"):
            try:
                models = self.client.models.list()
                if not models.data:
                    raise ValueError("No models found at the local LLM API endpoint.")
                self.config["llm_api"]["model"] = models.data[0].id
                logger.info(f"Auto-selected local model: {self.config['llm_api']['model']}")
            except Exception as e:
                logger.error(f"Failed to auto-detect a model from {self.config['llm_api']['base']}. Please specify a model in your config. Error: {e}")
                raise
        logger.info("PromptMask configuration loaded successfully.")

    async def reload_config(self):
        """
        Asynchronously reloads the configuration from the disk and re-initializes clients.
        This makes configuration changes effective without restarting the server.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._initialize_clients)
        logger.info("Configuration reloaded successfully.")
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
            reversed_map = json.loads(json_str)
            if not is_dict_str_str(reversed_map):
                raise TypeError("Mask map should be a dictionary mapping strings to strings.")
            # Ensure 1:1 mapping by reversing the map to check for duplicate masks
            mask_map = {v: k for k, v in reversed_map.items() if len(v)>0} #raise TypeError if v is unhashable
            if len(reversed_map) != len(mask_map):
                logger.warning("Duplicate masks detected in LLM response. The result might be inconsistent.")
            
            #wrap mask w/ self.config["mask_wrapper"]
            mask_wrapper = self.config.get("mask_wrapper", {})
            mask_left, mask_right = mask_wrapper.get("left", ""), mask_wrapper.get("right", "")
            wrapped_mask_map = {
                original_value: f"{mask_left}{mask_key.upper()}{mask_right}"
                for original_value, mask_key in mask_map.items()
            }
            
            return wrapped_mask_map
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse mask response: {e}\nResponse: {response_content}")
            return {"err":type(e).__name__}

    def _oai_chat_comp(self, messages:str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.config["llm_api"]["model"],
                messages=messages,
                temperature=0.0
        )
            return completion.choices[0].message.content
        except APITimeoutError as e:
            return str({"err":type(e).__name__})

    async def _async_oai_chat_comp(self, messages: List[Dict[str, str]]) -> str:
        """Asynchronous chat completion call."""
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.config["llm_api"]["model"],
                messages=messages,
                temperature=0.0,
            )
            return completion.choices[0].message.content
        except APITimeoutError as e:
            return str({"err":type(e).__name__})

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
        text_to_mask = "\n".join([m["content"] for m in messages if m.get("role") not in ["system"] and m.get("content")])
        
        if not text_to_mask.strip():
            return messages, {}
            
        _, mask_map = self.mask_str(text_to_mask)
        
        masked_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if new_msg.get("content") and new_msg.get("role") not in ["system"]:
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
            
        response_content = await self._async_oai_chat_comp(messages)
        logger.debug(f"Async masking response: {response_content}")

        mask_map = self._parse_mask_response(response_content)
        
        masked_text = text
        for original, mask in mask_map.items():
            masked_text = masked_text.replace(original, mask)
            
        return masked_text, mask_map

    async def async_mask_messages(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Async version of mask_messages."""
        text_to_mask = "\n".join([m["content"] for m in messages if m.get("role") not in ["system"] and m.get("content")])
        
        if not text_to_mask.strip():
            return messages, {}
            
        _, mask_map = await self.async_mask_str(text_to_mask)
        
        masked_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if new_msg.get("content") and new_msg.get("role") not in ["system"]:
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