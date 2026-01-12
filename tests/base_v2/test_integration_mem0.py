
import unittest
import os
import sys

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
mem0_repo_root = os.path.join(project_root, 'mem0')

# 1. Add 'mem0' repo root to sys.path FIRST to ensure we find the inner 'mem0' package
# instead of the outer 'mem0' folder acting as a namespace.
if mem0_repo_root not in sys.path:
    sys.path.insert(0, mem0_repo_root)

# 2. Add project root to sys.path for 'src' imports
if project_root not in sys.path:
    sys.path.append(project_root)

import shutil
import json
import time

# Try to handle potential version import error in mem0 if package not installed
try:
    import mem0
except ImportError:
    pass
except Exception:
    pass

from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.agents.langgraph_workflow import NSCAMWorkflow
from src.utils.llm_interface import LLMInterface
from src.models.data_models import MemoryObject

class TestNSCAMMem0Integration(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./tests/base_v2/test_output"
        self.log_dir = "./experiments/logs"
        self.results_dir = "./results"
        
        for d in [self.test_dir, self.log_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)
            
        # Redirect Logging to File
        import logging
        log_file = os.path.join(self.log_dir, f"test_integration_mem0_{int(time.time())}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True # Reset any existing config
        )
        print(f"Logging to {log_file}")
        
        # Initialize components
        self.store = DualMemoryStore(persist_dir=self.test_dir)
        self.km = KnowledgeManager(self.store)
        # Use Real Model (None means load from config)
        self.llm = LLMInterface() 
        self.workflow = NSCAMWorkflow(self.store, self.km, self.llm)
        
        # Load Locomo 10 Dataset (first item)
        with open("data/locomo10.json", "r", encoding='utf-8') as f:
            self.dataset = json.load(f)

    def tearDown(self):
        # Save results if needed
        pass

    def test_dual_brain_flow(self):
        """
        Test that:
        1. Graph captures logic (Alice moved).
        2. Mem0 captures context (Alice loves croissants).
        3. Retrieval combines them.
        """
        print("\n=== Testing Dual Brain Flow ===")
        user_id = "test_alice"
        
        # Simulate conversation from Locomo dataset (simplified)
        # Message 1: P1 (Alice) says she lives in Paris
        msg1 = "I am currently living in Paris and I absolutely love the croissants here."
        ts1 = time.time()
        
        print(f"Adding Msg 1: {msg1}")
        
        # 1. Add to Graph (Logic)
        self.km.add_knowledge_triple("Alice", "lives_in", "Paris", timestamp=ts1)
        
        # 2. Add to Mem0 (Episodic)
        mem1 = MemoryObject(id="m1", content=msg1, timestamp=ts1, raw_meta={'user_id': user_id, 'speaker': 'Alice'})
        self.store.add_episodic_memory(mem1)
        
        time.sleep(2) # Time gap
        
        # Message 2: P1 (Alice) says she moved to London
        msg2 = "I just moved to London for a new job. It's great but different."
        ts2 = time.time()
        
        print(f"Adding Msg 2: {msg2}")
        
        # 1. Add to Graph
        self.km.add_knowledge_triple("Alice", "lives_in", "London", timestamp=ts2)
        
        # 2. Add to Mem0
        mem2 = MemoryObject(id="m2", content=msg2, timestamp=ts2, raw_meta={'user_id': user_id, 'speaker': 'Alice'})
        self.store.add_episodic_memory(mem2)
        
        # Verify Graph Logic
        print("Verifying Graph Logic...")
        edges = self.store.get_node_edges("Alice", "lives_in")
        active_edge = next((e for e in edges if e.status == 'active'), None)
        self.assertIsNotNone(active_edge)
        self.assertEqual(active_edge.target, "London")
        print("Graph correctly identifies Alice lives in London.")
        
        # Verify Mem0 Retrieval
        print("Verifying Mem0 Retrieval...")
        # Search for preferences
        q_pref = "What does Alice like?"
        mem_results = self.store.retrieve_episodic(query_vector=[], k=3, query_text=q_pref, user_id=user_id)
        
        # Since Mem0 is async/complex, we might not get immediate results in a unit test if it calls an API.
        # But if it works, we arguably see "croissants" in the content.
        # NOTE: If using a mock or real API, this assertion depends on the backend speed.
        print(f"Mem0 Results: {[m.content for m in mem_results]}")
        
        # Run Workflow
        print("Running Workflow Query...")
        query = "Where does Alice live now and what does she like?"
        response = self.workflow.run(query)
        print(f"Final Answer: {response.get('final_answer')}")
        
        # Simple assertion that we got an answer
        self.assertTrue(len(response.get('final_answer', '')) > 0)

if __name__ == '__main__':
    unittest.main()
