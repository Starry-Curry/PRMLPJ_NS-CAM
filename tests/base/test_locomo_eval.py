import os
import json
import shutil
import time
import re
from datetime import datetime
from collections import defaultdict
import logging
import sys

# Ensure project root is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from src.storage.dual_memory_store import DualMemoryStore
from src.models.data_models import SemanticTime, TimeType
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.agents.langgraph_workflow import NSCAMWorkflow
from src.models.data_models import MemoryObject
from src.utils.logging_setup import setup_logging
from src.evaluation.metrics import Evaluator

# Setup Logger
logger = logging.getLogger('LocomoTest')
logging.basicConfig(level=logging.INFO)

def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)

def test_locomo_eval():
    logger = logging.getLogger('LocomoTest')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    data_path = os.path.join(project_root, 'data', 'locomo10.json')
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        return

    logger.info("Loading Locomo dataset...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use first sample only as requested
    sample = data[0]
    sample_id = sample.get('sample_id', 'sample_0')
    logger.info(f"Testing on Sample ID: {sample_id}")

    # Use timestamp for unique graph per run (User Requirement 1)
    run_ts = int(time.time())
    test_dir_name = f"test_data_locomo_{sample_id}_{run_ts}"
    test_dir = os.path.join(project_root, 'experiments', test_dir_name)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    ensure_dirs(test_dir)
    logger.info(f"Graph Store Directory: {test_dir}")

    # Logging and results
    logs_dir = os.path.join(project_root, 'experiments', 'logs')
    results_dir = os.path.join(project_root, 'results', 'locomo_eval')
    ensure_dirs(logs_dir)
    ensure_dirs(results_dir)
    
    log_file = os.path.join(logs_dir, f'eval_{sample_id}_{int(time.time())}.log')

    # Setup Logger
    # Use simpler console for user interaction, file for details
    # (重置 Logger: 终端仅显示进度，文件记录详细日志)
    root_val = logging.getLogger()
    for h in root_val.handlers[:]:
        root_val.removeHandler(h)
    
    # 1. File Handler (Detailed)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # 2. Console Handler (Progress Only)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

    # Root logger gets the file handler (so all libraries log to file)
    root_val.addHandler(fh)
    root_val.setLevel(logging.INFO) # Capture everything for file

    # Quiet down external libraries in root
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    # Configure main script logger (LocomoTest) to use Console + File
    logger = logging.getLogger('LocomoTest')

    logger.setLevel(logging.INFO)
    logger.propagate = False # Don't duplicate up to root (which has file handler)
    logger.addHandler(fh)
    logger.addHandler(ch) # Show LocomoTest INFO on console
    
    logger.info(f"Log file: {log_file}")
    
    # Init system


    llm = LLMInterface()
    store = DualMemoryStore(persist_dir=test_dir)
    km = KnowledgeManager(store)
    wf = NSCAMWorkflow(store, km, llm)
    evaluator = Evaluator(llm)

    # --- INGESTION ---
    logger.info("Ingesting Conversation History...")
    conversation = sample.get('conversation', {})
    
    # Sort sessions if needed, but let's just grab all valid session keys
    # Assuming session_X naming or just iterating dict
    session_keys = [k for k in conversation.keys() if k.startswith('session_')]
    # Basic sort by number if they look like session_1, session_2
    def sort_key(k):
        m = re.search(r'session_(\d+)', k)
        return int(m.group(1)) if m else 999
    session_keys = sorted(session_keys, key=sort_key)

    msg_count = 0
    triples_count = 0
    
    # Use tqdm only if we want fancy bars, but simple logging as requested
    total_sessions = len(session_keys)
    
    for i, sk in enumerate(session_keys):
        # Look for explicit date_time key first (e.g. "session_1" -> "session_1_date_time")
        dt_key = f"{sk}_date_time"
        date_str = conversation.get(dt_key, "")
        
        # Parse timestamp from string (Format: "1:07 pm on 25 February, 2022")
        # Fallback to advancing mock time if parse fails
        current_session_ts = time.time() + msg_count * 0.1
        if date_str:
            try:
                # Need to handle format like "1:07 pm on 25 February, 2022"
                # Remove 'on ' to make parsing easier? Or just use format string.
                # Format: "%I:%M %p on %d %B, %Y"
                dt_obj = datetime.strptime(date_str, "%I:%M %p on %d %B, %Y")
                current_session_ts = dt_obj.timestamp()
            except ValueError:
                # Try partial or alternative formats if needed
                 logger.warning(f"Could not parse date: {date_str}, using fallback.")

        val = conversation[sk]
        if isinstance(val, list): # List of turns
            
            # --- Streaming / Chunked Ingestion Simulation (User Requirement 2) ---
            chunk_buffer = [] 
            MAX_CHUNK_SIZE = 5 # Process every 5 messages or end of session
            
            for t_idx, turn in enumerate(val):
                text = turn.get('text', '')
                speaker = turn.get('speaker', 'Unknown')
                dia_id = turn.get('dia_id', '')
                
                if text:
                    content = f"{speaker}: {text}"
                    chunk_buffer.append(content)
                    
                    # Episodic Memory (Instant)
                    # Use Session TS + small offset
                    msg_ts = current_session_ts + t_idx * 1.0
                    mem = MemoryObject(content=content, timestamp=msg_ts, raw_meta={'source_id': dia_id})
                    store.add_episodic_memory(mem)
                    msg_count += 1
                
                # Check buffer condition
                if len(chunk_buffer) >= MAX_CHUNK_SIZE or t_idx == len(val) - 1:
                    # Extract from chunk
                    # Context includes previous messages implicitly if we keep a rolling buffer?
                    # For strict streaming, we just process this chunk.
                    chunk_text = "\n".join(chunk_buffer)
                    
                    # Call LLM Extraction
                    extracted_triples = llm.extract_triples(chunk_text, reference_timestamp=current_session_ts)
                    
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
                        
                        # Use Smart Ingestion (Requirement 2: Check Redundancy)
                        # We use the midpoint TS of the chunk for the knowledge
                        km.add_knowledge_triple_smart(sub, pred, obj, timestamp=msg_ts, semantic_time_obj=st)
                        triples_count += 1
                    
                    # Clear buffer
                    chunk_buffer = []
        
        # Progress Log
        if (i+1) % 5 == 0 or (i+1) == total_sessions:
            logger.info(f"Ingestion Progress: {i+1}/{total_sessions} sessions processed. (Messages: {msg_count}, Triples: {triples_count})")
    
    logger.info(f"Ingestion Complete. Total Messages: {msg_count}, Total Triples: {triples_count}")

    # --- TESTING ---

    logger.info("Starting QA Evaluation...")
    qa_list = sample.get('qa', [])
    
    # Group by category
    by_category = defaultdict(list)
    for idx, item in enumerate(qa_list):
        cat = item.get('category', 'unknown')
        by_category[cat].append(item)
    
    results = {
        "sample_id": sample_id,
        "metrics_by_category": {},
        "details": []
    }

    total_qs = len(qa_list)
    processed = 0

    for cat in sorted(by_category.keys()):
        logger.info(f"Processing Category {cat} ({len(by_category[cat])} questions)...")
        
        cat_metrics = {"rouge1": [], "rougeL": [], "f1": [], "llm_score": []}
        
        for q_item in by_category[cat]:
            question = q_item['question']
            
            # Handle cases where 'answer' key might be missing (e.g., category 5/adversarial)
            if 'answer' in q_item:
                 gold = str(q_item['answer'])
            elif 'adversarial_answer' in q_item:
                 gold = str(q_item['adversarial_answer'])
            else:
                 gold = ""
                 logger.warning(f"No answer found for question: {question}")
            
            # Run Workflow

            start_t = time.time()
            res_state = wf.run(question)
            duration = time.time() - start_t
            
            prediction = res_state.get('final_answer', '')
            vec_ctx = res_state.get('vector_context', [])
            graph_ctx = res_state.get('graph_context', {})
            
            # Retrieve Evidence IDs for logging
            retrieved_chunks = [v.content[:100] for v in vec_ctx]
            
            # Evaluate
            metrics = evaluator.evaluate_all(question, gold, prediction)
            
            # Store Metrics
            cat_metrics['f1'].append(metrics['f1'])
            for rk in ['rouge1', 'rougeL']:
                if rk in metrics['rouge']:
                    cat_metrics[rk].append(metrics['rouge'][rk])
            
            if metrics['llm_eval']['score'] is not None:
                cat_metrics['llm_score'].append(metrics['llm_eval']['score'])
            
            # Log Detail
            detail = {
                "question": question,
                "gold": gold,
                "prediction": prediction,
                "category": cat,
                "metrics": metrics,
                "retrieval": {
                    "vector_docs_count": len(vec_ctx),
                    "graph_path_found": bool(graph_ctx),
                    "snippet": retrieved_chunks[:3]
                },
                "duration": duration
            }
            results['details'].append(detail)
            
            processed += 1
            if processed % 5 == 0:
                logger.info(f"Progress: {processed}/{total_qs}")

        # Avg for Category
        avg_metrics = {}
        for k, v in cat_metrics.items():
            if v:
                avg_metrics[k] = sum(v) / len(v)
            else:
                avg_metrics[k] = 0.0
        
        results["metrics_by_category"][cat] = avg_metrics
        logger.info(f"Category {cat} Results: {avg_metrics}")

    # Save Results
    res_path = os.path.join(results_dir, f'eval_result_{sample_id}.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation Complete. Results saved to {res_path}")

if __name__ == "__main__":
    test_locomo_eval()
