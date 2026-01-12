import sys
import shutil
import os
import time
from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.agents.langgraph_workflow import NSCAMWorkflow
from src.utils.llm_interface import LLMInterface
from src.models.data_models import MemoryObject
from src.maintenance.consolidate import MemoryConsolidator
import colorama

def type_writer(text, speed=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()

def main():
    colorama.init()
    print(colorama.Fore.CYAN + """
    =======================================================
       NS-CAM: Neuro-Symbolic Chrono-Agentic Memory
       System Demo v0.1
    =======================================================
    """ + colorama.Style.RESET_ALL)
    
    # 1. Initialization
    persist_dir = "./demo_data"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        
    print(f"Initializing Memory Store at {persist_dir}...")
    store = DualMemoryStore(persist_dir=persist_dir)
    km = KnowledgeManager(store)
    llm = LLMInterface(model_name="mock")
    workflow = NSCAMWorkflow(store, km, llm)
    consolidator = MemoryConsolidator(store, km, llm)
    
    print(colorama.Fore.GREEN + "System Ready." + colorama.Style.RESET_ALL)
    
    # Pre-load some data
    print("\n[Scenario]: Alice moves from Paris to London.")
    
    # T=0
    print(f"T=0: Alice lives in Paris.")
    km.add_knowledge_triple("Alice", "lives_in", "Paris", timestamp=100.0)
    
    # T=10
    print(f"T=10: Alice moves to London.")
    time.sleep(1)
    # Conflict!
    km.add_knowledge_triple("Alice", "lives_in", "London", timestamp=200.0)
    
    print(colorama.Fore.YELLOW + "Memory Consolidated. Conflict Resolved." + colorama.Style.RESET_ALL)
    
    while True:
        try:
            print("\n" + "-"*50)
            query = input(colorama.Fore.WHITE + "Enter Query (or 'exit'): " + colorama.Style.RESET_ALL)
            if query.strip().lower() in ['exit', 'quit']:
                break
                
            print(colorama.Fore.BLUE + "Thinking..." + colorama.Style.RESET_ALL)
            start = time.time()
            response = workflow.run(query)
            duration = time.time() - start
            
            intent = response.get('intent', 'unknown')
            answer = response.get('final_answer', '')
            graph_ctx = response.get('graph_context', {})
            
            print(f"Intent Strategy: {colorama.Fore.MAGENTA}{intent}{colorama.Style.RESET_ALL}")
            if graph_ctx:
                edges = graph_ctx.get('edges', [])
                print(f"Graph Context: Found {len(edges)} active facts.")
                for e in edges:
                    print(f"  - {e.source} {e.relation} {e.target} (conf={e.confidence:.2f})")
            
            print("\n" + colorama.Fore.GREEN + "Answer: " + answer + colorama.Style.RESET_ALL)
            print(f"(Time: {duration:.2f}s)")
            
        except KeyboardInterrupt:
            break
    
    # Explicitly release resources
    print("Cleaning up...")
    del store
    del km
    del workflow
    del consolidator
    
    try:
        import gc
        gc.collect()
        time.sleep(1) # Give OS time to release handle
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        print("Cleanup successful.")
    except Exception as e:
        print(f"Cleanup warning: Could not fully remove {persist_dir}. ({e})")
        print("You may delete it manually.")

    print("\nBye!")

if __name__ == "__main__":
    main()
