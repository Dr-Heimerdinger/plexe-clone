"""
Tools related to code generation, including solution planning, training code, 
and inference code generation.
"""

import logging

from smolagents import tool

from plexe.core.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)


@tool
def register_best_solution(best_solution_id: str) -> str:
    """
    Register the solution with the best performance as the final selected solution in the object
    registry. This step is required in order for the solution to be available for final model building.

    Args:
        best_solution_id: 'solution_id' of the best performing solution

    Returns:
        Success message confirming registration
    """
    from plexe.core.entities.solution import Solution

    object_registry = ObjectRegistry()

    try:
        # Get the best solution
        best_solution = object_registry.get(Solution, best_solution_id)

        # Register the solution with a fixed ID for easy retrieval
        object_registry.register(Solution, "best_performing_solution", best_solution, overwrite=True)

        logger.debug(f"✅ Registered best performing solution with ID '{best_solution_id}'")
        return f"Successfully registered solution with ID '{best_solution_id}' as the best performing solution."

    except Exception as e:
        logger.warning(f"⚠️ Error registering best solution: {str(e)}")
        raise RuntimeError(f"Failed to register best solution: {str(e)}")
