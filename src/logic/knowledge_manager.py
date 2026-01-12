import time
from typing import List, Dict, Optional, Tuple, Set
import math
from src.models.data_models import GraphEdge, MemoryObject, SemanticTime, TimeType
from src.storage.dual_memory_store import DualMemoryStore
from src.utils.logging_setup import get_logger
from src.utils.llm_interface import LLMInterface

logger = get_logger(__name__)

# Constants from plan
DEFAULT_CONFIDENCE = 0.5
LEARNING_RATE = 0.2 # eta for Bayesian update
DECAY_RATE = 0.1 # beta
ACTIVATION_DECAY = 0.6 # alpha for spreading activation

class KnowledgeManager:
    def __init__(self, store: DualMemoryStore):
        self.store = store
        # Cache for predicate cardinality inference
        self.predicate_cache = {}
        # Interface for LLM calls (only for cardinality check mostly)
        self.llm = LLMInterface()

    def assess_cardinality(self, predicate: str) -> str:
        """
        Determine if a predicate allows 'concurrent' values (Many) or is 'exclusive' (One).
        Returns 'Concurrent' or 'Exclusive'.
        """
        if predicate in self.predicate_cache:
            return self.predicate_cache[predicate]
        
        # Hardcoded overrides for speed/safety
        if predicate in ["lives_in", "is_located_in", "current_job", "marriage_status"]:
            return "Exclusive"
        
        prompt = f"""
        Analyze the relationship predicate: "{predicate}".

        Determine its temporal exclusivity for a human subject.
        - "Exclusive": A person typically has only ONE active value (e.g., 'lives_in', 'current_job').
        - "Concurrent": A person can have MULTIPLE active entries (e.g., 'visited', 'likes', 'friend_with', 'attended').
        
        Return ONLY "Exclusive" or "Concurrent".
        """
        try:
            result = self.llm.generate(prompt, max_tokens=10).strip()
            if "Exclusive" in result:
                res = "Exclusive"
            else:
                res = "Concurrent"
        except Exception:
            res = "Concurrent" # Default to permissive
            
        self.predicate_cache[predicate] = res
        return res

    def add_knowledge_triple(self, 
                             sub: str, 
                             pred: str, 
                             obj: str, 
                             timestamp: float,
                             attributes: Dict = None,
                             semantic_time_obj: SemanticTime = None) -> GraphEdge:
        """
        Add a knowledge triple to the graph, handling conflicts and reinforcement.
        Now supports Semantic Time.
        """
        attributes = attributes or {}
        
        # 0. Embed Nodes for Retrieval (Lazy/On-Write)
        # We embed sub and obj so they can be searched by vector later
        try:
            vecs = self.llm.embed([sub, obj])
            if vecs:
                self.store.add_node_embedding(sub, vecs[0])
                self.store.add_node_embedding(obj, vecs[1])
        except Exception as e:
            logger.warning(f"Node embedding failed: {e}")

        # 1. Check existing edges for (sub, pred)
        existing_edges = self.store.get_node_edges(sub, relation=pred)
        
        # Identify if we have a match or conflict
        match_edge = None
        conflict_edges = []
        
        for edge in existing_edges:
            if edge.status == 'archived':
                continue
                
            if edge.target == obj:
                match_edge = edge
            else:
                # Potential conflict
                conflict_edges.append(edge)
        
        if match_edge:
            return self._handle_reinforcement(match_edge, timestamp, attributes, semantic_time_obj)
        else:
            return self._handle_conflict_and_insert(sub, pred, obj, timestamp, attributes, conflict_edges, semantic_time_obj)

    def add_knowledge_triple_smart(self, sub, pred, obj, timestamp, semantic_time_obj=None):
        """
        Smart Ingestion: Checks for redundancy using LLM before adding.
        """
        # 1. Retrieval: Get existing edges for the subject
        existing_edges = self.store.get_node_edges(source=sub)
        
        # 2. Heuristic Filter: If too many, keep only lexically similar relations
        if len(existing_edges) > 20:
            import difflib
            existing_edges = [e for e in existing_edges if difflib.SequenceMatcher(None, e.relation, pred).ratio() > 0.3]

        if not existing_edges:
             return self.add_knowledge_triple(sub, pred, obj, timestamp, semantic_time_obj=semantic_time_obj)

        # 3. LLM Redundancy Check
        decision = self.llm.check_redundancy((sub, pred, obj), existing_edges)
        status = decision.get('status', 'new')
        target_id = decision.get('target_edge_id')

        if status == 'duplicate':
            logger.info(f"SmartIngest: Ignored duplicate {sub}-{pred}-{obj}")
            return None
        
        elif status == 'update' and target_id:
            # Find the edge to archive
            target_edge = next((e for e in existing_edges if e.id == target_id), None)
            if target_edge:
                target_edge.status = 'archived'
                t_start, _ = target_edge.system_window
                target_edge.system_window = (t_start, timestamp)
                self.store.add_graph_edge(target_edge)
                logger.info(f"SmartIngest: Archived {target_id} for update.")
            
            return self.add_knowledge_triple(sub, pred, obj, timestamp, semantic_time_obj=semantic_time_obj)
            
        else:
            return self.add_knowledge_triple(sub, pred, obj, timestamp, semantic_time_obj=semantic_time_obj)

    def _handle_reinforcement(self, edge: GraphEdge, timestamp: float, attributes: Dict, sem_time: SemanticTime) -> GraphEdge:
        """
        Reinforce existing edge: update confidence and extend time window.
        """
        # Bayesian Update: sigma_new = sigma_old + eta * (1 - sigma_old)
        old_conf = edge.confidence
        new_conf = old_conf + LEARNING_RATE * (1.0 - old_conf)
        edge.confidence = min(1.0, new_conf)
        
        # Update System Window
        t_rec, t_arch = edge.system_window
        # Reinforcement doesn't change record time, just keeps it active?
        # Actually, maybe we track 'last_seen'?
        # For now, system_window[0] is creation. We could add 'last_updated' to attributes.
        attributes['last_updated'] = timestamp
        
        # Update Semantic Time if new info is better?
        # If edge has 'unknown' time but new data has 'point', update it.
        if sem_time and sem_time.time_type != TimeType.UNKNOWN:
             if edge.semantic_time.time_type == TimeType.UNKNOWN:
                 edge.semantic_time = sem_time
        
        # Merge attributes
        if attributes:
            edge.attributes.update(attributes)
            
        self.store.add_graph_edge(edge)
        
        logger.info(f"Reinforced edge: {edge.source}-{edge.relation}->{edge.target} (conf: {old_conf:.2f}->{new_conf:.2f})")
        return edge

    def _handle_conflict_and_insert(self, 
                                   sub: str, 
                                   pred: str, 
                                   obj: str, 
                                   timestamp: float, 
                                   attributes: Dict,
                                   conflict_edges: List[GraphEdge],
                                   sem_time: SemanticTime) -> GraphEdge:
        """
        Handle conflicts by archiving old edges if relation is Exclusive.
        """
        # Dynamic Cardinality Check
        cardinality = self.assess_cardinality(pred)
        
        if cardinality == "Exclusive":
            for old_edge in conflict_edges:
                # Archive old edge
                old_edge.status = 'archived'
                # Close the system window
                t_start, _ = old_edge.system_window
                old_edge.system_window = (t_start, timestamp)
                self.store.add_graph_edge(old_edge)
                logger.info(f"Archived conflict: {old_edge.relation} (New: {obj}, Old: {old_edge.target})")
        
        # Create New Edge
        new_edge = GraphEdge(
            source=sub,
            target=obj,
            relation=pred,
            system_window=(timestamp, float('inf')), # Active
            semantic_time=sem_time or SemanticTime(time_type=TimeType.UNKNOWN),
            confidence=DEFAULT_CONFIDENCE,
            status='active',
            attributes=attributes
        )
        self.store.add_graph_edge(new_edge)
        return new_edge

    def spreading_activation(self, 
                             seed_nodes: List[str], 
                             steps: int = 2, 
                             query_time: float = None,
                             include_archived: bool = False,
                             query: str = None) -> Dict[str, float]:

        """
        Perform spreading activation from seed nodes.
        Returns a dictionary of {node_id: activation_score}.
        """
        if query_time is None:
            query_time = time.time()
            
        # Initial scores for seeds
        scores: Dict[str, float] = {node: 1.0 for node in seed_nodes}
        visited: Set[str] = set(seed_nodes)
        current_layer = seed_nodes
        
        for step in range(steps):
            next_layer = []
            for node in current_layer:
                current_score = scores[node]
                
                # Decay score for next hop
                # Basic idea: S(m) * w_{mn} (from plan)
                # We assume a global decay factor ACTIVATION_DECAY as well
                
                neighbors = self.store.get_graph_neighbors(node)
                
                for edge in neighbors:
                    target = edge.target
                    
                    # Filter archived edges unless necessary (plan says S(n)=0 if archived)
                    if edge.status == 'archived' and not include_archived:
                        continue
                        
                    # Relation Filtering / Weighting
                    weight_mod = 1.0
                    if query:
                         # Very simple keyword match for now. Efficient.
                         q_lower = query.lower()
                         if edge.relation.lower() in q_lower:
                             weight_mod = 2.0 
                         elif any(sub in edge.relation.lower() for sub in q_lower.split()):
                             weight_mod = 1.2

                    # Calculate weight transmission
                    # S(target) += S(source) * edge.weight * decay
                    transmission = current_score * edge.weight * ACTIVATION_DECAY * weight_mod
                    
                    if target not in scores:
                        scores[target] = 0.0
                    
                    # Accumulate score (additive model)
                    scores[target] += transmission
                    
                    if target not in visited:
                        visited.add(target)
                        next_layer.append(target)
            
            current_layer = next_layer
        
        # Apply Temporal Decay to final scores?
        # Plan: Final S(n) ... - beta * Decay(t_now - t_end)
        # We apply this to all found nodes
        final_scores = {}
        for node, score in scores.items():
            # To apply temporal decay, we need the "freshness" of the node.
            # Usually defined by the most recent active edge pointing to it or from it.
            # For simplification using pure node score, we might skp, but let's try to find max t_end.
            # This is expensive if we query edges for every node. 
            # Optimization: Carry t_end in BFS or just ignore for Phase 1.
            # Let's ignore precise node timestamping for now and trust the spreading logic.
            final_scores[node] = score
            
        return final_scores

    def get_context_subgraph(self, seeds: List[str], steps: int = 2, include_archived: bool = False, query: str = None) -> Dict:
        """
        Helper to get a human/LLM readable subgraph for context.
        """
        scores = self.spreading_activation(seeds, steps, include_archived=include_archived, query=query)
        # Get top-k nodes
        top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
        nodes = [n for n, s in top_nodes]
        
        edges = []
        for n in nodes:
            out_edges = self.store.get_node_edges(n)
            for e in out_edges:
                if e.target in nodes:
                    if e.status == 'active' or include_archived:
                        edges.append(e)
                    
        return {"nodes": nodes, "edges": edges}
