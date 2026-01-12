import pytest
import shutil
import os
import time
from src.models.data_models import GraphEdge
from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager

@pytest.fixture
def km():
    test_dir = "./test_data_logic"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    store = DualMemoryStore(persist_dir=test_dir)
    km = KnowledgeManager(store)
    yield km
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_add_knowledge_basic(km):
    edge = km.add_knowledge_triple("Alice", "likes", "Apples", timestamp=100.0)
    assert edge.source == "Alice"
    assert edge.target == "Apples"
    assert edge.confidence == 0.5
    
    # Verify in store
    saved = km.store.get_graph_neighbors("Alice")
    assert len(saved) == 1
    assert saved[0].target == "Apples"

def test_reinforcement(km):
    # First add
    km.add_knowledge_triple("Bob", "knows", "Java", timestamp=100.0)
    
    # Second add (reinforce)
    km.add_knowledge_triple("Bob", "knows", "Java", timestamp=200.0)
    
    edges = km.store.get_node_edges("Bob", "knows")
    assert len(edges) == 1
    edge = edges[0]
    
    # Check time window extension
    assert edge.system_window == (100.0, 200.0)
    # Check bayesian update: 0.5 + 0.2*(1-0.5) = 0.6
    assert abs(edge.confidence - 0.6) < 0.001

def test_conflict_resolution(km):
    # Whitelisted relation 'located_in'
    
    # T1: Paris
    km.add_knowledge_triple("User", "located_in", "Paris", timestamp=100.0)
    
    # T2: London (Conflict)
    km.add_knowledge_triple("User", "located_in", "London", timestamp=200.0)
    
    edges = km.store.get_node_edges("User", "located_in")
    assert len(edges) == 2
    
    paris_edge = next(e for e in edges if e.target == "Paris")
    london_edge = next(e for e in edges if e.target == "London")
    
    # Check archive status
    assert paris_edge.status == 'archived'
    assert paris_edge.system_window[1] == 200.0
    
    # Check new edge
    assert london_edge.status == 'active'
    assert london_edge.system_window == (200.0, float('inf'))

def test_non_conflicting_updates(km):
    # Relation NOT in whitelist, e.g. 'visited'
    km.add_knowledge_triple("User", "visited", "Paris", timestamp=100.0)
    km.add_knowledge_triple("User", "visited", "London", timestamp=200.0)
    
    edges = km.store.get_node_edges("User", "visited")
    assert len(edges) == 2
    for e in edges:
        assert e.status == 'active'

def test_spreading_activation(km):
    # A -> B -> C
    km.add_knowledge_triple("A", "link", "B", timestamp=100.0)
    km.add_knowledge_triple("B", "link", "C", timestamp=100.0)
    
    scores = km.spreading_activation(["A"], steps=2)
    
    assert "A" in scores
    assert "B" in scores
    assert "C" in scores
    
    # Score A = 1.0
    # Score B = 1.0 * w(1.0) * decay(0.6) = 0.6
    # Score C = 0.6 * w(1.0) * decay(0.6) = 0.36
    assert scores["A"] == 1.0
    assert abs(scores["B"] - 0.6) < 0.001
    assert abs(scores["C"] - 0.36) < 0.001
