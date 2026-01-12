import pytest
import shutil
import os
from src.models.data_models import MemoryObject, GraphEdge
from src.storage.dual_memory_store import DualMemoryStore

@pytest.fixture
def store():
    # Setup
    test_dir = "./test_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    store = DualMemoryStore(persist_dir=test_dir)
    yield store
    
    # Teardown
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_episodic_memory(store):
    mem = MemoryObject(content="User lives in Paris", raw_meta={"topic": "location"})
    saved_id = store.add_episodic_memory(mem)
    
    # Mock query vector (must match dimension used in store, here 384 from mock)
    # We cheat and use the same mock embedding we know is used inside (or close enough)
    # Ideally integration test uses real embeddings or mock works by identity
    q_vec = [0.1] * 384
    results = store.retrieve_episodic(q_vec, k=1)
    
    assert len(results) >= 1
    assert results[0].content == "User lives in Paris"
    assert results[0].id == saved_id

def test_graph_memory(store):
    edge = GraphEdge(
        source="User",
        target="Paris",
        relation="lives_in",
        time_window=(0.0, 10.0),
        weight=0.9
    )
    store.add_graph_edge(edge)
    
    # Verify persistence manually if needed, or via neighbors
    neighbors = store.get_graph_neighbors("User")
    assert len(neighbors) == 1
    assert neighbors[0].target == "Paris"
    assert neighbors[0].relation == "lives_in"
    assert neighbors[0].confidence == 0.5 # Default

def test_graph_persistence(store):
    edge = GraphEdge(source="A", target="B", relation="rel", time_window=(0,1))
    store.add_graph_edge(edge)
    store.persist()
    
    # Reload
    new_store = DualMemoryStore(persist_dir=store.persist_dir)
    neighbors = new_store.get_graph_neighbors("A")
    assert len(neighbors) == 1
    assert neighbors[0].target == "B"
