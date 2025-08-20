# src/promptmask/utils.py

import sys
import logging

from typing import Dict, Union, get_args

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

def is_dict_str_str(data: dict) -> bool:
    return all(isinstance(key, str) and isinstance(value, str) for key, value in data.items())

def flatten_dict(input_dict: dict, parent_key: str = '', separator: str = '_') -> Dict[str, str]:
    """
    Recursively flatten a nested dictionary into a single-level dictionary.
    """
    if not isinstance(input_dict, dict):
        return {}

    result = {}
    for key, value in input_dict.items():
        # (parent_key, K) -> parent_key_K
        processed_key = parent_key + separator + str(key) if parent_key else str(key)

        if isinstance(value, dict) and value:
            # V: Dict -> result.update(flatten(V))
            result.update(flatten_dict(value, processed_key, separator=separator))
        elif isinstance(value, list) and value:
            # V: List -> result[K_0]=V[0], result[K_1]=V[1]...
            for i, item in enumerate(value):
                item_key = processed_key + separator + str(i)
                if isinstance(item, dict) and item:
                    # V[i]: Dict -> result.update(flatten(V[i]))
                    result.update(flatten_dict(item, item_key, separator=separator))
                elif type(item) in (str, int, float):
                    # V[i]: (str, int, float) -> result[K_i] = str(V[i])
                    result[item_key] = str(item)
        elif type(value) in (str, int, float):
            # V: (str, int, float) -> result[K] = str(V)
            result[processed_key] = str(value)

    return result