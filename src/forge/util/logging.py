# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# FIXME: remove this once wandb fixed this issue
# https://github.com/wandb/wandb/issues/10890
# Patch importlib.metadata.distributions before wandb imports it
# to filter out packages with None metadata
import importlib.metadata

# Guard to ensure this runs only once
if not hasattr(importlib.metadata, "_distributions_patched"):
    _original_distributions = importlib.metadata.distributions

    def _patched_distributions():
        """Filter out distributions with None metadata"""
        for distribution in _original_distributions():
            if distribution.metadata is not None:
                yield distribution

    importlib.metadata.distributions = _patched_distributions
    importlib.metadata._distributions_patched = True

import logging
from functools import lru_cache

from torch import distributed as dist


def get_logger(level: str | None = None) -> logging.Logger:
    """
    Get a logger with a stream handler.

    Args:
        level (str | None): The logging level. See https://docs.python.org/3/library/logging.html#levels for list of levels.

    Example:
        >>> logger = get_logger("INFO")
        >>> logger.info("Hello world!")
        INFO:forge.util.logging: Hello world!

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if level is not None:
        level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger


def log_rank_zero(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only on rank zero.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    logger.log(level, msg, stacklevel=2)


@lru_cache(None)
def log_once(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only once. LRU cache is used to ensure a specific message is
    logged only once, similar to how :func:`~warnings.warn` works when the ``once``
    rule is set via command-line or environment variable.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    log_rank_zero(logger=logger, msg=msg, level=level)
