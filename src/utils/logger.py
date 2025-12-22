"""
Logging configuration for Context Windows Lab.

Provides a ring-buffer logging system with configurable handlers.
"""

import json
import logging
import logging.config
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "context_windows_lab",
    config_path: Optional[Path] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Set up and return a configured logger.

    Args:
        name: Logger name
        config_path: Path to log config JSON file
        log_level: Default log level if config not found

    Returns:
        Configured logger instance
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "logs" / "config" / "log_config.json"

    # Ensure logs directory exists
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Load config if exists
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Get settings from ring_buffer config format
        ring_config = config.get("ring_buffer", {})
        level = ring_config.get("log_level", log_level).upper()
        fmt = ring_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        logging.basicConfig(
            level=getattr(logging, level),
            format=fmt,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        # Fallback to basic config
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    return logging.getLogger(name)


def get_logger(name: str = "context_windows_lab") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)

    return logger


# Create default logger instance
logger = get_logger()
