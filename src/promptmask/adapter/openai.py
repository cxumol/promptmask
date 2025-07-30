# src/promptmask/adapter/openai.py

from openai import OpenAI
from openai import AsyncOpenAI as BaseAsyncOpenAI
import inspect

from ..core import PromptMask

class OpenAIMasked(OpenAI):
    """
    An OpenAI client that automatically masks and unmasks sensitive data.
    It inherits from openai.OpenAI and overrides the chat.completions.create method.
    """
    def __init__(self, *args, promtmask_config: dict = None, **kwargs):
        """
        Initializes the masked client.
        
        Args:
            *args: Positional arguments for openai.OpenAI client.
            promtmask_config (dict, optional): Configuration for PromptMask.
            **kwargs: Keyword arguments for openai.OpenAI client.
        """
        super().__init__(*args, **kwargs)
        self._promptmask = PromptMask(config=promtmask_config)
        self._hijack_chat_completions()

    def _hijack_chat_completions(self):
        # Store original methods
        self._original_chat_create = self.chat.completions.create
        
        # Replace the 'create' method with our wrapper
        def_create = self._create_wrapper(self._original_chat_create)
        self.chat.completions.create = def_create

    def _create_wrapper(self, original_create_method):
        """Creates the wrapper for the 'create' method to handle masking."""
        
        def masked_create(*args, **kwargs):
            messages = kwargs.get("messages", [])
            stream = kwargs.get("stream", False)

            # Mask the messages
            masked_messages, mask_map = self._promptmask.mask_messages(messages)
            kwargs["messages"] = masked_messages

            # Call the original 'create' method
            response = original_create_method(*args, **kwargs)

            # Unmask the response
            if stream:
                return self._promptmask.unmask_stream(response, mask_map)
            else:
                if response.choices:
                    # Unmask content in choices
                    unmasked_content = self._promptmask.unmask_str(
                        response.choices[0].message.content, mask_map
                    )
                    response.choices[0].message.content = unmasked_content
                return response
        
        return masked_create
