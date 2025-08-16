# src/promptmask/__init__.py

"""
A local-first privacy layer for Large Language Model users.
"""
from .core import PromptMask
from .adapter.openai import OpenAIMasked

__all__ = ["PromptMask", "OpenAIMasked"]