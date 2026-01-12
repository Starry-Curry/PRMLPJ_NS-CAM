from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.models.data_models import MemoryObject, GraphEdge

class MemoryStoreInterface(ABC):
    """
    Abstract Base Class for Dual Memory Store.
    """

    @abstractmethod
    def add_episodic_memory(self, memory: MemoryObject) -> str:
        """Add a chunk to the vector store."""
        pass

    @abstractmethod
    def retrieve_episodic(self, query_vector: List[float], k: int = 5) -> List[MemoryObject]:
        """Retrieve similar chunks."""
        pass

    @abstractmethod
    def add_graph_edge(self, edge: GraphEdge) -> None:
        """Add or update an edge in the knowledge graph."""
        pass

    @abstractmethod
    def get_graph_neighbors(self, node_id: str) -> List[GraphEdge]:
        """Get all edges connected to a node."""
        pass
    
    @abstractmethod
    def get_node_edges(self, source: str, relation: Optional[str] = None) -> List[GraphEdge]:
        """Get specific edges from a source node."""
        pass

    @abstractmethod
    def persist(self):
        """Save state to disk."""
        pass
