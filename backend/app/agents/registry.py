"""Importing this module registers every agent handler with the runtime."""
from app.agents import analyst, clean, evaluate, orchestrator, report, understand  # noqa: F401
from app.agents.runtime import HANDLERS

AGENT_KINDS = tuple(HANDLERS)
