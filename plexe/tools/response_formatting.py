"""
This module provides tools for forcing an agent to return its response in a specific format.
"""

from typing import Dict

from smolagents import tool


@tool
def format_final_orchestrator_agent_response(
    best_solution_id: str,
    performance_metric_name: str,
    performance_metric_value: float,
    performance_metric_comparison_method: str,
    model_review_output: Dict[str, str],
) -> dict:
    """
    Returns a dictionary containing the exact fields that the agent must return in its final response. The purpose
    of this tool is to 'package' the final deliverables of the ML engineering task. The best_solution_id should be
    the ID of the solution that was selected as the best performing one.

    Args:
        best_solution_id: The solution ID for the selected best ML solution
        performance_metric_name: The name of the performance metric to optimise that was used in this task
        performance_metric_value: The value of the performance attained by the selected ML model
        performance_metric_comparison_method: The comparison method used to evaluate the performance metric
        model_review_output: The output of the 'review_model' tool which contains a review of the selected ML model

    Returns:
        Dictionary containing the fields that must be returned by the agent in its final response
    """
    from plexe.core.object_registry import ObjectRegistry
    from plexe.core.entities.solution import Solution

    # Get the solution plan from the best solution
    object_registry = ObjectRegistry()
    try:
        best_solution = object_registry.get(Solution, best_solution_id)
        solution_plan = best_solution.plan or "Solution plan not available"
    except Exception:
        solution_plan = "Solution plan not available"

    return {
        "solution_plan": solution_plan,
        "performance": {
            "name": performance_metric_name,
            "value": performance_metric_value,
            "comparison_method": performance_metric_comparison_method,
        },
        "metadata": model_review_output,
    }
