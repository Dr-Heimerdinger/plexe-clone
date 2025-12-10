"""
Conversational Agent for guiding users through ML model definition and initiation.

This module defines a ConversationalAgent that helps users define their ML requirements
through natural conversation, validates their inputs, and initiates model building
when all necessary information has been gathered.
"""

import logging
import os
from typing import Optional, Callable

from smolagents import ToolCallingAgent, LiteLLMModel

from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import get_dataset_preview
from plexe.tools.conversation import validate_dataset_files, initiate_model_build, validate_db_connection

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """
    Agent for conversational model definition and build initiation.

    This agent guides users through defining their ML requirements via natural
    conversation, helps clarify the problem, validates dataset availability,
    and initiates the model building process when all requirements are met.
    """

    def __init__(
        self,
        model_id: str | None = None,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the conversational agent.

        Args:
            model_id: Model ID for the LLM to use for conversation
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callable for chain of thought logging
        """
        # Choose model id with this precedence:
        # 1. explicit `model_id` argument
        # 2. env var `PLEXE_CONVERSATIONAL_MODEL`
        # 3. automatic selection based on available API keys
        #    - GEMINI -> gemini/gemini-2.5-flash
        #    - OPENAI -> gemini/gemini-2.5-flash
        #    - ANTHROPIC -> anthropic/claude-sonnet-4-20250514
        # 4. fallback to anthropic default
        env_model = os.environ.get("PLEXE_CONVERSATIONAL_MODEL")

        if model_id:
            self.model_id = model_id
        elif env_model:
            self.model_id = env_model
        else:
            # Auto-detect from available API keys
            if os.environ.get("GEMINI_API_KEY"):
                self.model_id = "gemini/gemini-2.5-flash"
            elif os.environ.get("OPENAI_API_KEY"):
                self.model_id = "gemini/gemini-2.5-flash"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.model_id = "anthropic/claude-sonnet-4-20250514"
            else:
                self.model_id = "anthropic/claude-sonnet-4-20250514"
        self.verbose = verbose

        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create the conversational agent with necessary tools
        self.agent = ToolCallingAgent(
            name="ModelDefinitionAssistant",
            description=(
                "Expert ML and DL consultant that helps users define their machine learning and deep learning requirements "
                "through conversational guidance. Specializes in clarifying problem definitions, "
                "understanding data requirements, and initiating model builds when ready. "
                "Maintains a friendly, helpful conversation while ensuring all technical "
                "requirements are properly defined before proceeding with model creation."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                get_dataset_preview,
                validate_dataset_files,
                validate_db_connection,
                initiate_model_build,
            ],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            step_callbacks=[chain_of_thought_callable] if chain_of_thought_callable else None,
            prompt_templates=get_prompt_templates(
                base_template_name="toolcalling_agent.yaml",
                override_template_name="conversational_prompt_templates.yaml",
            ),
        )
