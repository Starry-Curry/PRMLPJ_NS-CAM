import time
from typing import Dict, Any
from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.agents.agent_nodes import AgentNodes, AgentState
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Try importing LangGraph
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not found. Using sequential fallback.")

class NSCAMWorkflow:
    def __init__(self, store: DualMemoryStore, km: KnowledgeManager, llm: LLMInterface):
        self.nodes = AgentNodes(store, km, llm)
        self.app = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("profiler", self.nodes.profiler_node)
        workflow.add_node("retriever_graph", self.nodes.associative_retriever_node)
        workflow.add_node("retriever_vec", self.nodes.episodic_retriever_node)
        workflow.add_node("reasoner", self.nodes.reasoner_node)
        workflow.add_node("generator", self.nodes.generator_node)
        
        # Define Edges based on Profiler intent
        def route_profiler(state: AgentState):
            intent = state['intent']
            if intent == 'retrieve_vec':
                return "retriever_vec"
            elif intent == 'retrieve_graph':
                return "retriever_graph"
            else: # mixed or reasoning
                return "mixed_retrieval" 

        # We need a conditional edge from profiler
        # LangGraph conditional edges take a routing function and a mapping
        
        # Simplifying for this implementation: 
        # Actually, let's just make it run parallel or sequential based on intent.
        # But LangGraph nodes map 1:1 usually unless parallel.
        
        # Let's wire a simple flow:
        # Profiler -> (Logic) -> Retrievers -> Generator
        
        workflow.set_entry_point("profiler")
        
        # Routing logic
        workflow.add_conditional_edges(
            "profiler",
            route_profiler,
            {
                "retriever_vec": "retriever_vec",
                "retriever_graph": "retriever_graph",
                "mixed_retrieval": "retriever_graph" # Start with graph, then maybe vec
            }
        )
        
        # From Graph Retriever, we might check if we need Vec fallback (as per plan "Retrieval Fallback")
        def route_graph_fallback(state: AgentState):
            # Aggressive Fallback: Always chain Graph -> Vec to ensure Mem0 baseline performance
            # This ensures we get both Graph facts AND Episodic details (Hybrid RAG)
            return "retriever_vec" 

        workflow.add_conditional_edges(
            "retriever_graph",
            route_graph_fallback,
            {
                "retriever_vec": "retriever_vec"
            }
        )
        
        # Vector always goes to reasoner/generator
        workflow.add_edge("retriever_vec", "reasoner")
        
        # Reasoner -> Generator
        workflow.add_edge("reasoner", "generator")
        workflow.add_edge("generator", END)
        
        self.app = workflow.compile()

    def run(self, query: str, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Run the workflow for a given query.
        """
        initial_state = AgentState(
            query=query,
            query_time=time.time(),
            intent="",
            vector_context=[],
            graph_context={},
            final_answer="",
            user_id=user_id # Pass user_id to state
        )
        
        if LANGGRAPH_AVAILABLE and self.app:
            return self.app.invoke(initial_state)
        else:
            return self._run_sequential(initial_state)

    def _run_sequential(self, state: AgentState) -> AgentState:
        """
        Fallback sequential execution mimicking the graph logic.
        """
        logger.info("Running sequential workflow fallback")
        
        # 1. Profiler
        state = self.nodes.profiler_node(state)
        
        # 2. Retrieval Routing
        intent = state['intent']
        
        # Graph Retrieval
        if intent in ['retrieve_graph', 'mixed', 'reasoning']:
            state = self.nodes.associative_retriever_node(state)
            
        # Retrieval Fallback / Hybrid Chaining
        # Always run Episodic (Vec) retrieval unless strictly graph-only (which we rarely want for robustness)
        # For 'retrieve_graph', we still want Vec backup. 
        # For 'retrieve_vec' and 'mixed', we definitely want it.
        # Only skip if we implemented a Pure Logic mode, which we haven't.
        state = self.nodes.episodic_retriever_node(state)
             
        # 3. Reasoner
        state = self.nodes.reasoner_node(state)
        
        # 4. Generator
        state = self.nodes.generator_node(state)
        
        return state
