
import os
import sys
import logging
from collections import defaultdict

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.storage.dual_memory_store import DualMemoryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InspectStore")

def inspect_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Path determined from test_locomo_eval.py logic
    store_path = os.path.join(project_root, 'experiments', 'test_data_locomo_conv-26')
    
    if not os.path.exists(store_path):
        logger.error(f"Store path not found: {store_path}")
        return

    logger.info(f"Opening store at: {store_path}")
    store = DualMemoryStore(persist_dir=store_path)
    
    # 1. Check Graph for "Caroline"
    logger.info("--- Inspecting Graph ---")
    caroline_edges = store.get_graph_edges(source_node="Caroline")
    logger.info(f"Edges from Caroline: {len(caroline_edges)}")
    
    found_adoption = False
    for e in caroline_edges:
        logger.info(f"  -> [{e.relation}] -> {e.target} (Status: {e.system_window})")
        if "adoption" in e.target.lower() or "agency" in e.target.lower():
            found_adoption = True
            
    if not found_adoption:
        logger.warning("!!! 'Adoption' related edge NOT found for Caroline !!!")
    else:
        logger.info("VALID: Adoption edge found.")

    # 2. Check Vector for "adoption"
    logger.info("\n--- Inspecting Vector Store (Search Test) ---")
    query = "Researching adoption agencies"
    results = store.search_episodic_memory(query, n_results=10)
    
    found_doc = False
    for res in results:
        logger.info(f"  Doc: {res.content} (Score: {res.score if hasattr(res, 'score') else 'N/A'})")
        if "adoption" in res.content.lower():
            found_doc = True
            
    if not found_doc:
        logger.warning("!!! 'Adoption' related document NOT retrieved in top 10 !!!")
    else:
        logger.info("VALID: Adoption document exists and is retrievable.")

if __name__ == "__main__":
    inspect_data()
