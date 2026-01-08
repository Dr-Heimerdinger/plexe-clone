"""
This module provides a generic Registry pattern implementation for storing and retrieving objects by name or prefix.
"""

import logging
import dataclasses
import copy
import traceback
from typing import Dict, List, Type, TypeVar, Any, Optional


T = TypeVar("T")
logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Exception raised for registry operations errors."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        item_type: Optional[str] = None,
        item_name: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        self.operation = operation
        self.item_type = item_type
        self.item_name = item_name
        self.cause = cause
        self.stack_trace = traceback.format_exc()
        
        # Build detailed message
        parts = [message]
        if item_type:
            parts.append(f"type={item_type}")
        if item_name:
            parts.append(f"name={item_name}")
        if cause:
            parts.append(f"caused by: {type(cause).__name__}: {str(cause)}")
        
        super().__init__(" | ".join(parts))
    
    def get_full_trace(self) -> str:
        """Get complete error trace."""
        traces = [f"RegistryError during '{self.operation}': {str(self)}"]
        traces.append(f"Stack trace:\n{self.stack_trace}")
        if self.cause and hasattr(self.cause, '__traceback__'):
            cause_trace = ''.join(traceback.format_tb(self.cause.__traceback__))
            traces.append(f"Cause trace:\n{cause_trace}")
        return "\n".join(traces)


@dataclasses.dataclass
class Item:
    item: T
    immutable: bool = False


class ObjectRegistry:
    """
    Registry for storing and retrieving objects by name.

    This class implements the Singleton pattern so that registry instances are shared
    across the application. It provides methods for registering, retrieving, and
    managing objects in a type-safe manner.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObjectRegistry, cls).__new__(cls)
            cls._instance._items = {}
        return cls._instance

    @staticmethod
    def _get_uri(t: Type[T], name: str) -> str:
        return f"{str(t)}://{name}"

    def register(self, t: Type[T], name: str, item: T, overwrite: bool = False, immutable: bool = False) -> None:
        """
        Register an item with a given name.

        :param t: type prefix for the item
        :param name: identifier for the item - must be unique within the prefix
        :param item: the item to register
        :param overwrite: whether to overwrite an existing item with the same name
        :param immutable: whether the item should be treated as immutable (not modifiable)
        """
        uri = self._get_uri(t, name)
        was_overwrite = overwrite and uri in self._items

        if not overwrite and uri in self._items:
            raise ValueError(f"Item '{uri}' already registered, use a different name")

        self._items[uri] = Item(item, immutable=immutable)

        # Enhanced logging with context
        action = "overwrote" if was_overwrite else "registered"
        logger.info(f"Registry: {action} {uri} (immutable={immutable}, total: {len(self._items)} items)")

    def register_multiple(
        self, t: Type[T], items: Dict[str, T], overwrite: bool = False, immutable: bool = False
    ) -> None:
        """
        Register multiple items with a given prefix.

        :param t: type prefix for the items
        :param overwrite: whether to overwrite existing items with the same names
        :param immutable: whether the items should be treated as immutable (not modifiable)
        :param items: dictionary of item names and their corresponding objects
        """
        for name, item in items.items():
            self.register(t, name, item, overwrite=overwrite, immutable=immutable)

    def get(self, t: Type[T], name: str) -> T:
        """
        Retrieve an item by name.

        :param t: type prefix for the item
        :param name: the name of the item to retrieve
        :return: The registered item
        :raises RegistryError: If the item is not found in the registry (wraps KeyError)
        """
        uri = self._get_uri(t, name)
        if uri not in self._items:
            available_items = self.list_by_type(t)
            error_msg = (
                f"Item '{uri}' not found in registry. "
                f"Available items of type {t}: {available_items[:10]}{'...' if len(available_items) > 10 else ''}"
            )
            logger.warning(f"⚠️ {error_msg}")
            raise RegistryError(
                message=f"Item not found",
                operation="get",
                item_type=str(t),
                item_name=name,
            )
        logger.debug(f"Registry: Retrieved {uri} (immutable={self._items[uri].immutable})")
        return self._items[uri].item if not self._items[uri].immutable else copy.deepcopy(self._items[uri].item)

    def get_or_none(self, t: Type[T], name: str) -> Optional[T]:
        """
        Retrieve an item by name, returning None if not found.
        
        Use this when the item might not exist and that's acceptable.

        :param t: type prefix for the item
        :param name: the name of the item to retrieve
        :return: The registered item or None
        """
        try:
            return self.get(t, name)
        except (RegistryError, KeyError):
            return None

    def get_multiple(self, t: Type[T], names: List[str]) -> Dict[str, T]:
        """
        Retrieve multiple items by name.

        :param t: type prefix for the items
        :param names: List of item names to retrieve
        :return: Dictionary mapping item names to items
        :raises RegistryError: If any item is not found in the registry
        """
        result = {}
        missing = []
        for name in names:
            try:
                result[name] = self.get(t, name)
            except (RegistryError, KeyError):
                missing.append(name)
        
        if missing:
            raise RegistryError(
                message=f"Items not found: {missing}",
                operation="get_multiple",
                item_type=str(t),
            )
        return result

    def get_all(self, t: Type[T]) -> Dict[str, T]:
        """
        Retrieve all items for a given prefix.

        :param t: type prefix for the items
        :return: Dictionary mapping item names to items
        """
        return {name: item.item for name, item in self._items.items() if name.startswith(str(t))}

    def delete(self, t: Type[T], name: str) -> None:
        """
        Delete an item by name.

        :param t: type prefix for the item
        :param name: the name of the item to delete
        :raises RegistryError: If the item is not found
        """
        uri = self._get_uri(t, name)
        if uri in self._items:
            del self._items[uri]
            logger.debug(f"Registry: Deleted {uri}")
        else:
            raise RegistryError(
                message=f"Cannot delete - item not found",
                operation="delete",
                item_type=str(t),
                item_name=name,
            )

    def clear(self) -> None:
        """
        Clear all registered items.
        """
        self._items.clear()

    def list(self) -> List[str]:
        """
        List all registered item names.

        :return: List of item names in the registry
        """
        return list(self._items.keys())

    def list_by_type(self, t: Type[T]) -> List[str]:
        """
        List all registered names for a specific type.

        :param t: type prefix for the items
        :return: List of item names (without the type prefix) for the given type
        """
        prefix = str(t)
        return [uri.split("://")[1] for uri in self._items.keys() if uri.startswith(prefix)]

    # TODO: unclear if this is needed, consider deleting
    def get_all_solutions(self) -> List[Dict[str, Any]]:
        """
        Get all solutions tracked during model building.

        This method extracts solution information from the registry, focusing on
        code, performance metrics, and other solution-specific data for checkpointing.

        :return: List of solution data dictionaries
        """
        solutions = []

        # Extract training code and their results
        from plexe.internal.models.entities.code import Code
        from plexe.core.entities.solution import Solution

        # Get all code objects
        code_items = self.get_all(Code)
        node_items = self.get_all(Solution)

        # Build solution data from code and node objects
        for uri, code_obj in code_items.items():
            if isinstance(code_obj, Code):
                # Extract code ID and try to find associated node
                code_id = uri.split("://")[1]
                solution_data = {
                    "code_id": code_id,
                    "code": code_obj.code,
                    "iteration": getattr(code_obj, "iteration", 0),
                }

                # Look for associated node to get performance metrics
                for node_uri, node in node_items.items():
                    if isinstance(node, Solution) and node.training_code == code_obj.code:
                        if node.performance:
                            solution_data["performance"] = {
                                "name": node.performance.name,
                                "value": node.performance.value,
                                "comparison_method": getattr(
                                    node.performance.comparator, "comparison_method", "HIGHER_IS_BETTER"
                                ),
                            }
                        break

                solutions.append(solution_data)

        return solutions
