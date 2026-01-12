import os
import json
import shutil
import time
from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.agents.langgraph_workflow import NSCAMWorkflow
from src.models.data_models import MemoryObject
from src.utils.logging_setup import setup_logging


def ensure_dirs(base):
    os.makedirs(base, exist_ok=True)


def test_base_sample():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(project_root, 'data', 'locomo10.json')
    assert os.path.exists(data_path), f"Dataset not found: {data_path}"

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sample = data[0]

    # Prepare environment
    test_dir = os.path.join(project_root, 'test_data_base')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # Logging and results
    logs_dir = os.path.join(project_root, 'logs')
    results_dir = os.path.join(project_root, 'results', 'base_test')
    ensure_dirs(logs_dir)
    ensure_dirs(results_dir)
    setup_logging(log_file_path=os.path.join(logs_dir, 'base_test.json'))

    # Init system
    store = DualMemoryStore(persist_dir=test_dir)
    km = KnowledgeManager(store)
    llm = LLMInterface()
    wf = NSCAMWorkflow(store, km, llm)

    # Ingest timeline events from sample
    timeline = sample.get('timeline', [])
    for ev in timeline:
        mem = MemoryObject(content=ev.get('text', ''), timestamp=ev.get('timestamp', time.time()))
        store.add_episodic_memory(mem)
        # add triple if exists
        if 'fact' in ev:
            f = ev['fact']
            km.add_knowledge_triple(f['sub'], f['pred'], f['obj'], timestamp=ev.get('timestamp', time.time()))

    # Run the first question
    questions = sample.get('questions', [])
    q = questions[0]
    res = wf.run(q['query'])

    # Save result
    out = {
        'query': q['query'],
        'gold': q.get('gold_answer'),
        'result': res,
    }
    out_path = os.path.join(results_dir, 'run1.json')
    with open(out_path, 'w', encoding='utf-8') as wfout:
        json.dump(out, wfout, ensure_ascii=False, indent=2)

    # Cleanup
    try:
        del store
        del km
        del wf
    except:
        pass

    return out_path
