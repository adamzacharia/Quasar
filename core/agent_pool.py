# core/agent_pool.py
"""
Agent Pool — Instance reuse, auto-scaling, and idle cleanup.

Keeps initialized sub-agents warm across queries so we don't
re-instantiate SearchService, PlottingService, etc. every time.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class AgentPool:
    """Pool of initialized sub-agents for reuse across queries."""

    def __init__(self, max_idle_seconds: int = 300):
        self._pool: Dict[str, Any] = {}       # agent_type → agent instance
        self._timestamps: Dict[str, float] = {}  # agent_type → last used
        self.max_idle_seconds = max_idle_seconds

    def get_agent(
        self,
        agent_type: str,
        factory_fn: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        Get an agent from the pool, creating it if needed.

        Parameters
        ----------
        agent_type : str
            The type of agent (e.g. "archive", "literature").
        factory_fn : callable, optional
            Factory function to create the agent if not in pool.

        Returns
        -------
        BaseSubAgent or None
        """
        if agent_type in self._pool:
            self._timestamps[agent_type] = time.time()
            logger.debug("AgentPool: reusing %s", agent_type)
            return self._pool[agent_type]

        if factory_fn:
            agent = factory_fn()
            self._pool[agent_type] = agent
            self._timestamps[agent_type] = time.time()
            logger.info("AgentPool: created new %s", agent_type)
            return agent

        return None

    def release(self, agent_type: str) -> None:
        """Return an agent to the pool (mark as available)."""
        self._timestamps[agent_type] = time.time()

    def cleanup_idle(self) -> int:
        """Remove agents that have been idle longer than max_idle_seconds."""
        now = time.time()
        to_remove = [
            at
            for at, ts in self._timestamps.items()
            if now - ts > self.max_idle_seconds
        ]
        for at in to_remove:
            del self._pool[at]
            del self._timestamps[at]
            logger.info("AgentPool: cleaned up idle %s", at)
        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Return pool statistics."""
        now = time.time()
        return {
            "pool_size": len(self._pool),
            "agents": {
                at: {
                    "idle_seconds": round(now - ts, 1),
                    "type": type(agent).__name__,
                }
                for at, (agent, ts) in zip(
                    self._pool.keys(),
                    zip(self._pool.values(), self._timestamps.values()),
                )
            },
        }

    def clear(self) -> None:
        """Clear all agents from the pool."""
        self._pool.clear()
        self._timestamps.clear()
