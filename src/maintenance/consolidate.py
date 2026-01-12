from typing import List
import time
from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

class MemoryConsolidator:
    def __init__(self, store: DualMemoryStore, km: KnowledgeManager, llm: LLMInterface):
        self.store = store
        self.km = km
        self.llm = llm

    def consolidate_recent_memories(self, k: int = 10):
        """
        Simulate sleep consolidation:
        1. Fetch recent episodic memories (Mocking "recent" by query or just last K if store supported iteration).
           Since Chroma doesn't easily support "last K added" without specific metadata query, 
           we will assume we query for "recent" or use a buffer. 
           For this implementation, we will query a generic dummy vector to get *some* memories.
        2. Extract triples using LLM.
        3. Add to Graph.
        """
        logger.info("Starting memory consolidation...")
        
        # 1. Fetch memories (Simulation)
        # In production, we would query metadata={'timestamp': {'$gt': last_sync}}
        # Here we just grab some.
        q_vec = [0.1] * 384
        memories = self.store.retrieve_episodic(q_vec, k=k)
        
        count = 0
        for mem in memories:
            # 2. Extract Triples
            triples = self.llm.extract_triples(mem.content)
            
            # 3. Add to Graph
            for sub, pred, obj in triples:
                # Use memory timestamp for the fact
                self.km.add_knowledge_triple(sub, pred, obj, timestamp=mem.timestamp)
                count += 1
                
        logger.info(f"Consolidation complete. Extracted {count} facts from {len(memories)} memories.")
        return count

    def prune_old_memories(self, threshold_score: float = 0.1):
        """
        Prune low confidence edges or archived ones that are too old.
        (Implementation placeholder per plan requirements)
        """
        # Iterate graph edges?
        pass
