from typing import List, Optional, Dict, Tuple, Literal
from enum import Enum
from pydantic import BaseModel, Field
import time
import uuid

class TimeType(str, Enum):
    POINT = "point"       # e.g., "visited museum at 2pm"
    DURATION = "duration" # e.g., "lived in London 2020-2022"
    UNKNOWN = "unknown"

class SemanticTime(BaseModel):
    time_type: TimeType
    time_str: Optional[str] = None      # The raw extracted string or normalized date
    start_time: Optional[float] = None  # Unix timestamp if resolvable
    end_time: Optional[float] = None    # Unix timestamp if resolvable

class MemoryObject(BaseModel):
    """
    Episode Stream Item: Represents a raw chunk of conversation or text.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: Optional[List[float]] = None
    timestamp: float = Field(default_factory=time.time)
    speaker: Optional[str] = "user"
    topic_label: Optional[str] = None
    raw_meta: Dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class GraphEdge(BaseModel):
    """
    Semantic Chrono-Graph Edge: Represents a relationship between two entities.
    Definition: e = <r, w, T_sys, T_sem, sigma, s>
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    target: str
    relation: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Association strength (0-1)")
    
    # System Lifecycle [record_time, archive_time]
    # If active, archive_time is infinity (or None/Max)
    system_window: Tuple[float, float] = Field(description="[record_time, archive_time]")
    
    # Semantic Time (Real-world event time)
    semantic_time: SemanticTime = Field(default_factory=lambda: SemanticTime(time_type=TimeType.UNKNOWN))
    
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Bayesian confidence")
    status: Literal['active', 'archived'] = 'active'
    attributes: Dict = Field(default_factory=dict, description="Specific attributes")

    def is_active(self) -> bool:
        return self.status == 'active'

    @property
    def time_window(self) -> Tuple[float, float]:
        """Backward compatibility alias for system_window, used in tests."""
        return self.system_window



class RetrievalResult(BaseModel):
    """
    Unified result object for retrieval operations.
    """
    source: Literal['vector', 'graph']
    content: str
    score: float
    timestamp: float
    metadata: Dict = Field(default_factory=dict)
