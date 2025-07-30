# src/promptmask/utils.py

import sys
import logging

# Tomli/Tomllib compatibility
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Setup basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("PromptMask")


def _btwn(s: str, b: str, e: str) -> str:
    """A helper function to extract a substring between two markers."""
    i, j = s.find(b), s.rfind(e)
    if i == -1 or j == -1 or i >= j:
        raise ValueError(f"Markers not found or in wrong order within the string.\nString: '{s[:100]}...'\nStart: '{b}'\nEnd: '{e}'")
    return s[i:j+ len(e)]

def merge_configs(base, override):
    """Recursively merge dictionaries."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = merge_configs(base[key], value)
        else:
            base[key] = value
    return base