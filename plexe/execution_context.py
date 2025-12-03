
import contextvars
from typing import Optional, Callable

# Context variable to store the current chain of thought callable
_chain_of_thought_context = contextvars.ContextVar("chain_of_thought_context", default=None)

def set_chain_of_thought_callable(callable_obj: Callable):
    """Set the chain of thought callable for the current context."""
    return _chain_of_thought_context.set(callable_obj)

def get_chain_of_thought_callable() -> Optional[Callable]:
    """Get the chain of thought callable for the current context."""
    return _chain_of_thought_context.get()

def reset_chain_of_thought_callable(token):
    """Reset the chain of thought callable to its previous value."""
    _chain_of_thought_context.reset(token)
