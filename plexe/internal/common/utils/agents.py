"""
This module provides utilities for working with agents defined using the smolagents library.
"""

import yaml
import importlib
from plexe.config import config


def get_prompt_templates(
    base_template_name: str, override_template_name: str, template_vars: dict | None = None
) -> dict:
    """
    Given the name of a smolagents prompt template (the 'base template') and a plexe prompt template
    (the 'overriding template'), this function loads both templates and returns a merged template in which
    all keys from the overriding template overwrite the matching keys in the base template.
    """
    base_template_content = importlib.resources.files("smolagents.prompts").joinpath(base_template_name).read_text()
    override_template_content = (
        importlib.resources.files("plexe")
        .joinpath("templates/prompts/agent")
        .joinpath(override_template_name)
        .read_text()
    )

    # Perform string replacements for template variables
    if template_vars:
        for key, value in template_vars.items():
            base_template_content = base_template_content.replace(f"{{{{{key}}}}}", str(value))
            override_template_content = override_template_content.replace(f"{{{{{key}}}}}", str(value))

    base_template: dict = yaml.safe_load(base_template_content)
    override_template: dict = yaml.safe_load(
        override_template_content.replace("{{allowed_packages}}", str(config.code_generation.allowed_packages))
    )

    # Recursively merge two dictionaries to ensure deep merging
    def merge_dicts(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = merge_dicts(base[key], value)
            else:
                base[key] = value
        return base

    return merge_dicts(base_template, override_template)
