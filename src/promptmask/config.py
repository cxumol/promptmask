# src/promptmask/config.py

import os
import string
from pathlib import Path
from .utils import tomllib, merge_configs, logger

import importlib.resources as pkg_resources

DEFAULT_CONFIG_FILENAME = "promptmask.config.default.toml"
USER_CONFIG_FILENAME = "promptmask.config.user.toml"
PKG_NAME = "promptmask"

_is_verbose  = lambda config:config.get("general", {}).get("verbose")

def load_config(config_override = {}, config_file: str = "") -> dict:
    """
    Loads configuration with a clear priority order.
    Priority:
    1. `config_override` dictionary argument.
    2. `config_file` path argument.
    3. User-specific config file (`promptmask.config.user.toml`).
    4. Default config file packaged with the library.
    """
    # priority 4
    # default_config_path = Path(__file__).parent / DEFAULT_CONFIG_FILENAME
    try: #py3.9+
        config_path = pkg_resources.files(PKG_NAME).joinpath(DEFAULT_CONFIG_FILENAME)
        config_text = config_path.read_text(encoding='utf-8')
    except AttributeError: #py38
        with pkg_resources.open_text(PKG_NAME, DEFAULT_CONFIG_FILENAME, encoding='utf-8') as f:
            config_text = f.read()
    config = tomllib.loads(config_text)
    if _is_verbose(config):
        logger.info(f"Loaded default config from {default_config_path}")

    # 3. Load user config if it exists
    user_config_path = Path.cwd() / USER_CONFIG_FILENAME
    if user_config_path.exists():
        with open(user_config_path, "rb") as f:
            user_config = tomllib.load(f)
            config = merge_configs(config, user_config)
            if _is_verbose(config):
                logger.info(f"Loaded and merged user config from {user_config_path}")

    # 2. Load specified config file if provided
    if config_file:
        path = Path(config_file)
        if path.exists():
            with open(path, "rb") as f:
                file_config = tomllib.load(f)
                config = merge_configs(config, file_config)
                if _is_verbose(config):
                    logger.info(f"Loaded and merged specified config from {path}")
        else:
            logger.warning(f"Specified config file not found: {config_file}")

    # 1. Apply direct override
    if config_override:
        config = merge_configs(config, config_override)
        if _is_verbose(config):
            logger.info("Applied direct config override dictionary.")

    # Apply environment variables
    config["llm_api"]["base"] = os.getenv("LOCALAI_API_BASE", config["llm_api"]["base"])
    config["llm_api"]["key"] = os.getenv("LOCALAI_API_KEY", config["llm_api"]["key"])

    # Apply variables -> see core._build_mask_prompt

    if _is_verbose(config):
        logger.setLevel("DEBUG")
    logger.debug(f"Final loaded config:\n{config}")
    
    return config