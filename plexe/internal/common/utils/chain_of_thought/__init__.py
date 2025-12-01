"""
Chain of thought capturing and logging for agent systems.

This package provides a framework-agnostic way to capture, format, and display
the chain of thought reasoning from different agent frameworks.
"""

import contextvars
from typing import Optional

from plexe.internal.common.utils.chain_of_thought.protocol import StepSummary, ToolCall
from plexe.internal.common.utils.chain_of_thought.adapters import extract_step_summary_from_smolagents
from plexe.internal.common.utils.chain_of_thought.callable import ChainOfThoughtCallable
from plexe.internal.common.utils.chain_of_thought.emitters import (
    ChainOfThoughtEmitter,
    ConsoleEmitter,
    LoggingEmitter,
    MultiEmitter,
)
from plexe.internal.common.utils.chain_of_thought.websocket_emitter import WebSocketEmitter

# Context variable to store the current emitter for the thread/async context
# This allows tools and sub-agents to access the emitter without explicit passing
_current_emitter: contextvars.ContextVar[Optional[ChainOfThoughtEmitter]] = contextvars.ContextVar(
    "current_emitter", default=None
)


def set_current_emitter(emitter: Optional[ChainOfThoughtEmitter]) -> contextvars.Token:
    """
    Set the current emitter for the current context.

    Args:
        emitter: The emitter to set as current

    Returns:
        A token that can be used to reset the emitter
    """
    return _current_emitter.set(emitter)


def get_current_emitter() -> Optional[ChainOfThoughtEmitter]:
    """
    Get the current emitter for the current context.

    Returns:
        The current emitter, or None if not set
    """
    return _current_emitter.get()


def reset_current_emitter(token: contextvars.Token) -> None:
    """
    Reset the current emitter to its previous value.

    Args:
        token: The token returned by set_current_emitter
    """
    _current_emitter.reset(token)


__all__ = [
    "StepSummary",
    "ToolCall",
    "extract_step_summary_from_smolagents",
    "ChainOfThoughtCallable",
    "ChainOfThoughtEmitter",
    "ConsoleEmitter",
    "LoggingEmitter",
    "MultiEmitter",
    "WebSocketEmitter",
    "set_current_emitter",
    "get_current_emitter",
    "reset_current_emitter",
]
