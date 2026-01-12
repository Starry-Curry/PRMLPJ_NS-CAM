
import os
import sys
import time
import logging
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.agents.langgraph_workflow import NSCAMWorkflow
from src.models.data_models import MemoryObject, SemanticTime, TimeType

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugRetrieval")

def debug_caroline_adoption():
    # 1. Setup
    debug_dir = "./debug_data"
    import shutil
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        
    store = DualMemoryStore(persist_dir=debug_dir)
    # store.clear() # Start fresh
    
    # Use default model from .env (DeepSeek/DashScope)
    llm = LLMInterface() 
    km = KnowledgeManager(store)
    # Inject our LLM instance to save resources/ensure consistency
    km.llm = llm 
    
    workflow = NSCAMWorkflow(store, km, llm)
    
    # 2. Ingest Data
    # Text from locomo10.json (Session 13ish)
    # Context: Caroline is talking to Melanie.
    # Timestamp: Feb 25, 2022 (approx)
    
    ts_2022 = 1645794420.0 # 2022-02-25 1:07 PM approx
    
    text_snippet = "Researching adoption agencies — it's been a dream to have a family and give a loving home to kids who need it."
    full_text = f"Caroline: {text_snippet}"
    
    logger.info(f"Ingesting: {full_text} at TS={ts_2022}")
    
    # A. Add to Episodic
    mem = MemoryObject(content=full_text, timestamp=ts_2022, speaker="Caroline")
    store.add_episodic_memory(mem)
    
    # B. Extract Triples (Simulation of what happens in test_locomo_eval)
    logger.info("Extracting triples...")
    extracted_triples = llm.extract_triples(full_text, reference_timestamp=ts_2022)
    logger.info(f"Extracted Triples: {extracted_triples}")
    
    for item in extracted_triples:
        t_str, t_type_val = "None", "unknown"
        if len(item) == 5:
            sub, pred, obj, t_str, t_type_val = item
        elif len(item) == 3:
            sub, pred, obj = item
        else:
            continue
            
        try:
            safe_type = TimeType(t_type_val)
        except ValueError:
            safe_type = TimeType.UNKNOWN

        st = SemanticTime(time_type=safe_type, time_str=t_str)
        km.add_knowledge_triple(sub, pred, obj, timestamp=ts_2022, semantic_time_obj=st)
        
    # Check Graph persistence
    # We expect an edge from Caroline -> adoption agencies (or similar)
    
    # 3. Query
    questions = ["What did Caroline research?", "What is Caroline researching?"]
    
    for q in questions:
        logger.info(f"\n--- Query: {q} ---")
        # Run workflow
        result = workflow.run(q)
        
        logger.info(f"Intent: {result.get('intent')}")
        
        # Check Retrieval
        graph_ctx = result.get('graph_context', {})
        vec_ctx = result.get('vector_context', [])
        
        logger.info(f"Graph Context Edges ({len(graph_ctx.get('edges', []))}):")
        for e in graph_ctx.get('edges', []):
            logger.info(f"  {e.source} --[{e.relation}]--> {e.target}")
            
        logger.info(f"Vector Context Docs ({len(vec_ctx)}):")
        for d in vec_ctx:
            logger.info(f"  {d.content} (Score: N/A)")
            
        logger.info(f"Final Answer: {result.get('final_answer')}")

if __name__ == "__main__":
    debug_caroline_adoption()
