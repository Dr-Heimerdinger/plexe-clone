from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AgentProgress:
    """Track agent progress and phase information."""
    current_agent: str = ""
    current_phase: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_agent": self.current_agent,
            "current_phase": self.current_phase,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "progress_pct": int(self.completed_steps / max(self.total_steps, 1) * 100)
        }

