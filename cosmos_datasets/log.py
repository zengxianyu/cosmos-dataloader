"""Simple logging utility for cosmos_datasets."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# Create a default logger for backward compatibility
log = get_logger(__name__)