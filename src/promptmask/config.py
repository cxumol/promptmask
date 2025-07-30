# src/promptmask/config.py

import os
from pathlib import Path
from .utils import tomllib, merge_configs, logger

DEFAULT_CONFIG_FILENAME = "promptmask.config.default.toml"
USER_CONFIG_FILENAME = "promptmask.config.user.toml"

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
    # 4. Load default config
    default_config_path = Path(__file__).parent.parent.parent / DEFAULT_CONFIG_FILENAME
    with open(default_config_path, "rb") as f:
        config = tomllib.load(f)
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

    if _is_verbose(config):
        logger.info(f"config: {config}")

    return config