import os
import json
import shutil
import time
import logging
import sys
import numpy as np
import re
import uuid
from tqdm import tqdm

# Ensure project root in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try adding mem0 path if it exists, similar to other tests
mem0_path = os.path.join(project_root, 'mem0')
if os.path.exists(mem0_path) and mem0_path not in sys.path:
    sys.path.insert(0, mem0_path)

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.agents.langgraph_workflow import NSCAMWorkflow
from src.models.data_models import MemoryObject
from src.evaluation.metrics import Evaluator

# Setup Logger
logger = logging.getLogger('LocomoEvalLarge')
logger.setLevel(logging.INFO)

def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)

def setup_evaluation_logging(sample_id):
    """Sets up dual logging: simplified to console, detailed to file."""
    experiments_dir = os.path.join(project_root, 'experiments')
    logs_dir = os.path.join(experiments_dir, 'logs')
    ensure_dirs(logs_dir)
    
    timestamp = int(time.time())
    log_file = os.path.join(logs_dir, f'eval_{sample_id}_{timestamp}.log')
    
    # Remove existing handlers to avoid duplicates
    root_val = logging.getLogger()
    for h in root_val.handlers[:]:
        root_val.removeHandler(h)
    
    # File Handler (Detailed)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Console Handler (Progress Only - ERROR/WARNING only so tqdm isn't interrupted)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

    root_val.addHandler(fh)
    root_val.addHandler(ch)
    root_val.setLevel(logging.INFO)

    # Clean up third-party logs
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.WARNING)
    
    return log_file

def extract_knowledge(text: str, llm: LLMInterface) -> list:
    """
    Extracts knowledge triples (Subject, Predicate, Object) from text using LLM.
    Returns a list of tuples: (sub, pred, obj)
    """
    prompt = f"""
    Extract knowledge triples (Subject, Predicate, Object) from the text.
    - Subject and Object should be entities.
    - Predicate should be a relationship (verb phrase).
    - If time is mentioned, ignore it for the triple structure (handled separately by system).
    - Return EMPTY LIST if no factual relationships found.
    
    Text: "{text}"
    
    Output Format: JSON list of lists [["S","P","O"], ...]
    """
    try:
        resp = llm.generate(prompt).strip()
        # Cleanup markdown if any
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].split("```")[0].strip()
            
        triples = json.loads(resp)
        # Validate format
        valid_triples = []
        if isinstance(triples, list):
            for t in triples:
                if isinstance(t, list) and len(t) >= 3:
                    valid_triples.append((str(t[0]), str(t[1]), str(t[2])))
        return valid_triples
    except Exception as e:
        logger.warning(f"Extraction failed for text: {text[:30]}... Error: {e}")
        return []

def run_evaluation():
    # 1. Load Dataset
    data_path = os.path.join(project_root, 'data', 'locomo10.json')
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Target first sample
    sample = data[0]
    sample_id = sample.get('sample_id', 'sample_unknown')
    
    # Setup Logging
    log_file = setup_evaluation_logging(sample_id)
    logger.info(f"Starting Evaluation for Sample: {sample_id}")
    logger.info(f"Logging details to: {log_file}")

    # 2. Setup isolated test environment
    # CONFIG: Set this to a path string (e.g., "experiments/test_data_conv-26_1768204352") to skip ingestion
    # Set to None to run full ingestion
    REUSE_EXISTING_DB_PATH = "experiments/test_data_conv-26_1768204352" 
    
    # Always define run_ts for output file naming
    run_ts = int(time.time())

    if REUSE_EXISTING_DB_PATH:
        test_dir = os.path.join(project_root, REUSE_EXISTING_DB_PATH)
        if not os.path.exists(test_dir):
            logger.error(f"Reuse path does not exist: {test_dir}")
            return
        logger.info(f"Reusing existing test directory (Skipping Ingestion): {test_dir}")
    else:
        test_dir_name = f"test_data_{sample_id}_{run_ts}"
        test_dir = os.path.join(project_root, 'experiments', test_dir_name)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        ensure_dirs(test_dir)
        logger.info(f"Test Directory: {test_dir}")

    # 3. Initialize Components
    try:
        # Initialize Dual Store
        dual_store = DualMemoryStore(persist_dir=test_dir)
        
        # Initialize Logic Components
        knowledge_manager = KnowledgeManager(dual_store)
        llm = LLMInterface()
        
        # Initialize Workflow
        workflow = NSCAMWorkflow(
            store=dual_store,
            km=knowledge_manager,
            llm=llm
        )
        
        # Initialize Evaluator
        evaluator = Evaluator(llm_interface=llm)
        
    except Exception as e:
        logger.critical(f"Initialization Failed: {e}")
        return

    # 4. Ingest Documents (The Context)
    logger.info("--- Phase 1: Ingestion (Episodic + Graph) ---")
    
    eval_user_id = "eval_user"

    # Skip ingestion if reusing DB
    if REUSE_EXISTING_DB_PATH:
        logger.info("Skipping ingestion phase because REUSE_EXISTING_DB_PATH is set.")
    else:
        conversation = sample.get('conversation', {})

        
        # Sort sessions by index (session_1, session_2, ...)
        # Filter only keys that start with session_ and no date_time suffix
        session_keys = [k for k in conversation.keys() if k.startswith('session_') and not k.endswith('date_time')]
        
        # Sort based on number in key (e.g., session_2 vs session_10)
        def extract_sess_num(k):
            m = re.search(r'session_(\d+)', k)
            return int(m.group(1)) if m else 0
            
        from datetime import datetime
    
        def parse_locomo_date(date_str):
            """Parse locomo date format (e.g. '1:56 pm on 8 May, 2023') into unix timestamp float."""
            try:
                # Format: 1:56 pm on 8 May, 2023
                # Clean up extra spaces just in case
                date_str = " ".join(date_str.split())
                dt = datetime.strptime(date_str, "%I:%M %p on %d %B, %Y")
                return dt.timestamp()
            except ValueError:
                # Fallback attempts
                try:
                    # Try coarser formats if detailed fails
                    dt = datetime.strptime(date_str, "%Y-%m-%d") 
                    return dt.timestamp()
                except:
                    pass
                return time.time() # Absolute fallback to now if unparseable
    
        session_keys.sort(key=extract_sess_num)
        
        total_turns_imbibed = 0
        
        for sess_idx, sess_key in enumerate(tqdm(session_keys, desc="Ingesting Sessions")):
            turns = conversation[sess_key]
            sess_date_str = conversation.get(f"{sess_key}_date_time", "")
            
            # Resolve real timestamp from string
            sess_timestamp = parse_locomo_date(sess_date_str)
            
            logger.info(f"Processing {sess_key} ({len(turns)} turns) [Date: {sess_date_str}]")
            
            for turn in tqdm(turns, desc=f"Turns", leave=False):
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                dia_id = turn.get('dia_id', '')
                
                # Combine for semantic density
                full_content = f"[{sess_date_str}] {speaker}: {text}"
                
                # A. Add to Episodic Memory (Mem0)
                mem_id = str(uuid.uuid4())
                memory = MemoryObject(
                    id=mem_id,
                    content=full_content,
                    # CRITICAL: Use parsed real timestamp, not system time
                    timestamp=sess_timestamp, 
                    raw_meta={
                        "source": "locomo_eval", 
                        "user_id": eval_user_id,
                        "session": sess_key,
                        "dia_id": dia_id,
                        "speaker": speaker,
                        # Store string for display, but logic uses float timestamp
                        "timestamp_str": sess_date_str 
                    }
                )
                try:
                    dual_store.add_episodic_memory(memory)
                except Exception as e:
                    logger.error(f"Failed to add episodic memory: {e}")
                
                # B. Extract and Add to Graph (KnowledgeManager)
                # Only extract facts if it contains info (heuristic > 5 words)
                if len(text.split()) > 3:
                    triples = extract_knowledge(full_content, llm)
                    if triples:
                        for sub, pred, obj in triples:
                            # Simple naive absolute time for now (or use current run time as system time)
                            # In real system, we'd parse sess_date to timestamp
                            knowledge_manager.add_knowledge_triple(
                                sub=sub, 
                                pred=pred, 
                                obj=obj, 
                                timestamp=time.time(),
                                attributes={"source_dia": dia_id}
                            )
                
                total_turns_imbibed += 1
                # Rate limit slightly
                # time.sleep(0.05)
    
        logger.info(f"Ingestion Complete. Processed {total_turns_imbibed} turns.")

    # 5. Evaluate QA Pairs
    logger.info("--- Phase 2: QA Evaluation ---")
    qa_pairs = sample.get('qa', [])
    
    results = []
    # Initialize score containers
    scores = {
        'rouge1': [],
        'rougeL': [],
        'f1': [],
        'llm_score': []
    }
    
    # Category-wise scores: { category_id: { 'rouge1': [], ... } }
    category_scores = {}

    total_qa = len(qa_pairs)
    for idx, qa in enumerate(tqdm(qa_pairs, desc="Evaluating QA Pairs")):
        question = qa.get('question', '')
        ground_truth = str(qa.get('answer', ''))
        category = qa.get('category', 'unknown') # Extract category
        
        # FIX: Category 5 (Adversarial) usually lacks an 'answer' key. 
        # The correct behavior is to refuse/state info is missing.
        # We set a standard ground truth for evaluation.
        if (category == 5 or str(category) == "5") and not ground_truth:
            ground_truth = "Information not mentioned in the context."

        # Ensure separate tracking for this category
        if category not in category_scores:
            category_scores[category] = {
                'rouge1': [],
                'rougeL': [],
                'f1': [],
                'llm_score': []
            }
        
        logger.info(f"Evaluating Q{idx+1}/{total_qa} [Cat: {category}]: {question}")
        
        # Run inference
        try:
            start_time = time.time()
            
            # Using workflow.run with the fixed user_id param
            state_result = workflow.run(
                query=question, 
                user_id=eval_user_id
            )
            prediction = state_result.get('final_answer', '')
            intent = state_result.get('intent', 'unknown')
            
            latency = time.time() - start_time
            logger.info(f"  -> Intent: {intent}, Latency: {latency:.2f}s")
            
        except Exception as e:
            logger.error(f"Inference failed for Q{idx+1}: {e}")
            prediction = "ERROR"
            intent = "error"
            latency = 0

        # Calculate metrics
        eval_metrics = evaluator.evaluate_all(question, ground_truth, prediction)
        
        # Normalize & Store
        r1 = eval_metrics['rouge'].get('rouge1', 0.0) if isinstance(eval_metrics['rouge'], dict) else 0.0
        rL = eval_metrics['rouge'].get('rougeL', 0.0) if isinstance(eval_metrics['rouge'], dict) else 0.0
        f1 = eval_metrics['f1']
        
        llm_s = eval_metrics['llm_eval'].get('score')
        if llm_s is None: 
            llm_s = 0.0
        else:
            llm_s = float(llm_s) / 10.0
            
        scores['rouge1'].append(r1)
        scores['rougeL'].append(rL)
        scores['f1'].append(f1)
        scores['llm_score'].append(llm_s)
        
        # Add to Category Scores
        category_scores[category]['rouge1'].append(r1)
        category_scores[category]['rougeL'].append(rL)
        category_scores[category]['f1'].append(f1)
        category_scores[category]['llm_score'].append(llm_s)

        record = {
            "id": idx,
            "category": category, # Added category field
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "intent": intent,
            "metrics": {
                "rouge1": r1,
                "rougeL": rL,
                "f1": f1,
                "llm_score": llm_s,
                "llm_reason": eval_metrics['llm_eval'].get('reason')
            },
            "latency": latency
        }
        results.append(record)
        
        logger.info(f"  -> Prediction: {prediction[:100]}...")
        logger.info(f"  -> Scores: F1={f1:.2f}, R1={r1:.2f}, LLM={llm_s:.2f}")

    # 6. Aggregate Results
    avg_scores = {k: np.mean(v) if v else 0.0 for k, v in scores.items()}
    
    # Calculate Category Averages
    avg_category_scores = {}
    for cat, metrics_dict in category_scores.items():
        avg_category_scores[cat] = {k: np.mean(v) if v else 0.0 for k, v in metrics_dict.items()}

    summary = {
        "sample_id": sample_id,
        "timestamp": run_ts,
        "total_questions": total_qa,
        "aggregates": avg_scores,
        "category_aggregates": avg_category_scores, # Added to summary
        "details": results
    }

    # 7. Save Results
    results_dir = os.path.join(project_root, 'results', 'locomo_eval')
    ensure_dirs(results_dir)
    res_file = os.path.join(results_dir, f'eval_result_{sample_id}_{run_ts}.json')
    
    # Save Main Result File
    with open(res_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save Separate Aggregate Summary File (as requested)
    summary_file = os.path.join(results_dir, f'eval_summary_{sample_id}_{run_ts}.json')
    simple_summary = {
        "overall": avg_scores,
        "categories": avg_category_scores
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(simple_summary, f, ensure_ascii=False, indent=2)

    logger.info("--- Evaluation Complete ---")
    logger.info(f"Results saved to: {res_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("Aggregate Scores:")
    for k, v in avg_scores.items():
        logger.info(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    run_evaluation()