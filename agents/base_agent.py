"""
Base Agent Class
Foundation for all RadioFlow agents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
import time


@dataclass
class AgentResult:
    """Standardized result from any agent"""
    agent_name: str
    status: str  # "success", "error", "warning"
    data: Dict[str, Any]
    processing_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "data": self.data,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "error_message": self.error_message
        }


class BaseAgent(ABC):
    """
    Abstract base class for all RadioFlow agents.
    Provides common functionality and interface.
    """
    
    def __init__(self, name: str, model_name: str = None):
        self.name = name
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.is_loaded = False
        self._call_count = 0
        self._total_time_ms = 0
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model into memory. Returns True if successful."""
        pass
    
    @abstractmethod
    def process(self, input_data: Any, context: Optional[Dict] = None) -> AgentResult:
        """Process input and return result."""
        pass
    
    def __call__(self, input_data: Any, context: Optional[Dict] = None) -> AgentResult:
        """Execute the agent with timing."""
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                self.load_model()
            
            result = self.process(input_data, context)
            
        except Exception as e:
            result = AgentResult(
                agent_name=self.name,
                status="error",
                data={},
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
        
        # Update metrics
        self._call_count += 1
        self._total_time_ms += result.processing_time_ms
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "agent_name": self.name,
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "call_count": self._call_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / self._call_count if self._call_count > 0 else 0
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self._call_count = 0
        self._total_time_ms = 0
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model_name}', loaded={self.is_loaded})"
