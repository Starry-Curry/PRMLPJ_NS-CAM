import pytest
import shutil
import os
from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.agents.langgraph_workflow import NSCAMWorkflow

@pytest.fixture
def workflow():
    test_dir = "./test_data_wf"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    store = DualMemoryStore(persist_dir=test_dir)
    km = KnowledgeManager(store)
    llm = LLMInterface(model_name="mock")
    wf = NSCAMWorkflow(store, km, llm)
    
    # Pre-populate some data
    km.add_knowledge_triple("User", "lives_in", "Tokyo", timestamp=100.0)
    
    yield wf
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_workflow_run(workflow):
    # Depending on Mock LLM, "Where" query -> "retrieve_graph"
    # User lives in Tokyo -> Graph has "User lives_in Tokyo"
    
    # We need to make sure AssociativeRetriever finds "User" from query
    # Our simple implementation looks for exact node name in query.
    result = workflow.run("Where does the User live?")
    
    assert result is not None
    assert "Tokyo" in str(result.get('graph_context')) or "Tokyo" in result.get('final_answer')
    assert result['intent'] in ['retrieve_graph', 'mixed']

def test_workflow_fallback(workflow):
    # Query something not in graph -> Profiler (mock) might say 'retrieve_vec' or 'mixed'
    # "details about trip" -> 'retrieve_vec'
    
    result = workflow.run("What are the details about the trip?")
    assert result['intent'] == 'retrieve_vec'
    # Should have vector context (even if mock/empty list returned by store)
    assert 'vector_context' in result
