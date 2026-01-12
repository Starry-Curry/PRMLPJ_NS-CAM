import pytest
import shutil
import os
import time
from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.maintenance.consolidate import MemoryConsolidator
from src.models.data_models import MemoryObject

@pytest.fixture
def consolidator():
    test_dir = "./test_data_consol"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    store = DualMemoryStore(persist_dir=test_dir)
    km = KnowledgeManager(store)
    llm = LLMInterface(model_name="mock")
    mc = MemoryConsolidator(store, km, llm)
    yield mc
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_consolidation_flow(consolidator):
    # 1. Add episodic memory that contains extracting pattern
    # Mock LLM extract_triples looks for "X lives in Y"
    mem = MemoryObject(content="User lives in Berlin.", timestamp=1234567890.0)
    consolidator.store.add_episodic_memory(mem)
    
    # 2. Run consolidate
    count = consolidator.consolidate_recent_memories(k=5)
    
    # 3. Assert graph has edge
    assert count >= 1
    edges = consolidator.store.get_node_edges("User", "lives_in")
    assert len(edges) == 1
    assert edges[0].target == "Berlin"
    assert edges[0].time_window[0] == 1234567890.0
