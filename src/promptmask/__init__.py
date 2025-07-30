# src/promptmask/__init__.py

"""
PromptMask: Keep your secret while chatting with AI.
"""
from .core import PromptMask
from .adapter.openai import OpenAIMasked

__all__ = ["PromptMask", "OpenAIMasked"]