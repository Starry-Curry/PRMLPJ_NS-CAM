from typing import Dict, TypedDict, Any, List
from src.storage.dual_memory_store import DualMemoryStore
from src.logic.knowledge_manager import KnowledgeManager
from src.utils.llm_interface import LLMInterface
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# State Definition for LangGraph
class AgentState(TypedDict):
    query: str
    query_time: float
    intent: str
    vector_context: List[Any]
    graph_context: Dict[str, Any]
    final_answer: str
    is_history_query: bool # Added field
    user_id: str # Added field for multi-user support

class AgentNodes:
    def __init__(self, store: DualMemoryStore, km: KnowledgeManager, llm: LLMInterface):
        self.store = store
        self.km = km
        self.llm = llm

    def profiler_node(self, state: AgentState) -> AgentState:
        """
        Agent 1: The Profiler. Determines intent.
        (Agent 1: 意图识别模块)
        """
        query = state['query']
        prompt = f"""
        Act as a query classifier. Based on the user's query, determine the most suitable retrieval strategy.
        
        Categories:
        1. retrieve_vec:
           - User asks for specific details, events, or conversations from the past.
           - Examples: "What did X say?", "What happened at the party?", "Why did X leave?"
           
        2. retrieve_graph:
           - User asks for static facts, relationships, current status, or attributes.
           - Examples: "Where does X live?", "Who is X's brother?", "What is X's job?"
           
        3. reasoning:
           - Complex questions requiring multi-hop deduction, comparison, or summarization of long periods.
           - Examples: "How has X's attitude changed?", "Compare X and Y's jobs."
           
        4. mixed:
           - Questions involving both specific events and general facts.
           - Default choice if unsure.

        Query: "{query}"
        
        Return ONLY the category name (retrieve_vec, retrieve_graph, reasoning, mixed).
        Category:
        """
        intent = self.llm.generate(prompt).strip().lower()
        if intent not in ['retrieve_vec', 'retrieve_graph', 'reasoning', 'mixed']:
            intent = 'mixed' # Fallback
            
        # Strategy Bias: If "specific data" (phone, etc), bias to mixed/vec
        if any(char.isdigit() for char in query):
             if intent == 'retrieve_graph':
                 intent = 'mixed'
        
        # Temporal Analysis: Check for history keywords
        if any(w in query.lower() for w in ['before', 'past', 'previous', 'history', 'was', 'did']):
            state['is_history_query'] = True
        else:
            state['is_history_query'] = False

        logger.info(f"Profiler Intent: {intent} (History: {state.get('is_history_query', False)})")
        state['intent'] = intent
        return state

    def associative_retriever_node(self, state: AgentState) -> AgentState:
        """
        Agent 2: Associative Retriever (Graph).
        (Agent 2: 关联检索模块 - 知识图谱)
        Uses Vector-Anchored Entity Linking.
        """
        state['graph_context'] = {}
        if state['intent'] in ['retrieve_graph', 'reasoning', 'mixed']:
            
            # 1. Vector Search for Seeds
            # Embed the query to find semantically relevant nodes in the graph
            try:
                q_vec = self.llm.embed([state['query']])[0]
                seeds = self.store.search_nodes(q_vec, k=5)
            except Exception as e:
                logger.warning(f"Node vector search failed: {e}")
                seeds = []

            # 2. Fallback: Heuristic Entity Matching (robustness)
            # Check for specific node names that might be in the query
            # (e.g. "User_0", "Paris")
            q_tokens = state['query'].replace('?', '').split()
            # Crude entity extraction: look for existing nodes that match tokens
            # For small graphs, we can iterate. For large, this needs an inverted index or NER.
            if len(self.store.graph) < 1000:
                all_nodes = list(self.store.graph.nodes())
                for n in all_nodes:
                    if n not in seeds:
                        # Exact word match or full string match in query
                        if n in state['query'] or n.lower() in state['query'].lower():
                            seeds.append(n)
            
            # 3. Super Fallback: "User" node
            # In many personal agent scenarios, "User" is the central node
            if not seeds and self.store.graph.has_node("User"):
                seeds.append("User")

            if seeds:
                is_history = state.get('is_history_query', False)
                # Retrieve subgraph (2-hop neighborhood)
                context = self.km.get_context_subgraph(seeds, steps=2, include_archived=is_history)
                state['graph_context'] = context
                edge_count = len(context.get('edges', []))
                logger.info(f"Graph Context Retrieved: {edge_count} edges for seeds {seeds}")
            else:
                logger.info("No seeds found for graph retrieval.")
                
        return state


    def episodic_retriever_node(self, state: AgentState) -> AgentState:
        """
        Retrieves Vector Context.
        (Agent 3: 向量检索模块 - 情景记忆)
        NOW USES MEM0.
        """
        state['vector_context'] = []
        original_query = state['query']
        
        # User ID handling
        user_id = state.get('user_id', 'default_user') 

        # OPTIMIZATION: Query Rewrite for Vector Similarity
        # Transform question to declarative statement for better embedding match
        prompt = f"""
        Rewrite the user's question into a hypothetical first-person declarative statement (memory) that would contain the answer. 
        If the query is already declarative, keep it.
        
        Examples:
        Q: "Where did I go yesterday?" -> "I went to [place] yesterday."
        Q: "Who is my best friend?" -> "My best friend is [Name]."
        Q: "Did I buy milk?" -> "I bought milk."
        
        Question: "{original_query}"
        Rewritten Statement:
        """
        try:
            # Simple rewriting (low token cost, high value)
            search_query = self.llm.generate(prompt).strip().replace('"', '')
            logger.info(f"Query Rewritten: '{original_query}' -> '{search_query}'")
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            search_query = original_query

        # Increase k to 50 to match Mem0 baseline
        # Pass both original or rewritten? Usually rewritten matches better.
        results = self.store.retrieve_episodic(query_vector=[], k=50, query_text=search_query, user_id=user_id)
        
        # Fallback: If rewritten yielded 0, try original
        if not results and search_query != original_query:
            logger.info("Rewritten query returned 0 results. Retrying with original.")
            results = self.store.retrieve_episodic(query_vector=[], k=50, query_text=original_query, user_id=user_id)
            
        state['vector_context'] = results
        logger.info(f"Mem0 Context Retrieved: {len(results)} items")
            
        return state

    def reasoner_node(self, state: AgentState) -> AgentState:
        """
        Agent 3: Temporal Reasoner.
        Resolves conflicts in retrieved context logic manually to boost accuracy.
        """
        graph_ctx = state.get('graph_context', {})
        edges = graph_ctx.get('edges', [])
        
        reasoning_result = {
            "current": None,
            "past": [],
            "conflict_resolved": False
        }
        
        # Simple Logic: For 'exclusive' relations (e.g. lives_in), 
        # find the active one and the most recent archived ones.
        
        # Filter by relation if query mentions it? 
        # For now, let's just analyze "lives_in" or "located_in" as they are common benchmarks.
        
        # Find active edge
        active_edge = next((e for e in edges if e.status == 'active'), None)
        if active_edge:
            reasoning_result['current'] = active_edge.target
            
        # Find archived edges
        archived = sorted(
            [e for e in edges if e.status == 'archived'], 
            key=lambda x: x.system_window[1], # key on archive time (system_window[1])
            reverse=True
        )
        if archived:
            reasoning_result['past'] = [e.target for e in archived]
            
        # Inject reasoning into context str for Generator
        # We append a structured hint
        if state.get('is_history_query', False):
             # User asked for history
             if reasoning_result['past']:
                 hint = f"[System Reasoner]: The previous value was {reasoning_result['past'][0]}."
                 state['reasoning_hint'] = hint
        else:
             # User likely asked for current
             if reasoning_result['current']:
                 hint = f"[System Reasoner]: The current active value is {reasoning_result['current']}."
                 state['reasoning_hint'] = hint
                 
        return state

    def generator_node(self, state: AgentState) -> AgentState:
        """
        Agent 4: Generator.
        (Agent 4: 生成模块)
        """
        from datetime import datetime

        # Construct Context String
        context_str = ""
        
        # 1. Reasoning Hint (Priority)
        if state.get('reasoning_hint'):
            context_str += f"### Priority Hints:\n{state['reasoning_hint']}\n\n"
        
        # Graph Context
        if state.get('graph_context'):
            context_str += "### Knowledge Graph (Relationships & Facts):\n"
            edges = state['graph_context'].get('edges', [])
            if not edges:
                context_str += "  (No relevant graph data found)\n"
            else:
                for e in edges:
                    status_str = "(Archived/Past)" if e.status == 'archived' else "(Active/Current)"
                    # e.start_time / e.end_time if available
                    context_str += f"- {e.source} --[{e.relation}]--> {e.target} {status_str}\n"
        
        context_str += "\n"

        # Vector Context & Dynamic Date Calculation
        all_dates = []
        if state.get('vector_context'):
            context_str += "### Episodic Memory (Mem0 Semantic Logs):\n"
            for m in state['vector_context']:
                # Get timestamp values
                ts_val_float = m.timestamp
                ts_display_str = None
                
                if hasattr(m, 'raw_meta') and m.raw_meta:
                    # Check for explicit display string first
                    ts_display_str = m.raw_meta.get('timestamp_str')
                    
                    # Check for float timestamp in meta if not on object
                    if not ts_val_float:
                        meta_ts = m.raw_meta.get('timestamp')
                        if isinstance(meta_ts, (int, float)):
                            ts_val_float = float(meta_ts)

                # 1. Date Calculation Logic (using float)
                dt_obj = None
                if ts_val_float:
                     try:
                        dt_obj = datetime.fromtimestamp(ts_val_float)
                        if dt_obj.year > 2025: # Filter artifacts
                            pass
                     except:
                        pass
                
                # Fallback: try parsing string if float failed or didn't exist
                if not dt_obj and ts_display_str:
                     clean_ts = " ".join(ts_display_str.split())
                     formats = ["%I:%M %p on %d %B, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]
                     for fmt in formats:
                        try:
                            dt_obj = datetime.strptime(clean_ts, fmt)
                            break
                        except ValueError:
                            continue

                if dt_obj and dt_obj.year < 2026:
                    all_dates.append(dt_obj)
                
                # 2. Display Logic (Prefer String)
                if not ts_display_str:
                    if ts_val_float:
                        # Convert float to readable string if we don't have original
                        ts_display_str = datetime.fromtimestamp(ts_val_float).strftime('%Y-%m-%d %H:%M')
                    else:
                        ts_display_str = "Unknown Time"
                
                context_str += f"- [Time: {ts_display_str}] {m.content}\n"

        # Determine Current Reference Date (Dynamic Time Anchor)
        # Match logic from memzero/search.py
        current_ref_date = "2023-12-31" # Default fallback
        if all_dates:
            try:
                current_ref_date = max(all_dates).strftime("%Y-%m-%d")
            except Exception:
                pass

        # Optimized Prompt based on memzero/search.py
        prompt = f"""
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from conversation logs. These memories contain timestamped information.
    
    # DYNAMIC TIME ANCHOR:
    Current Reference Date: {current_ref_date}
    (Use this date as the "today" or "now" reference for all relative time calculations)

    # INSTRUCTIONS:
    1. Analyze the provided memories and their timestamps.
    2. **Timeline-Based Thinking**: Before answering, you MUST output a <Thinking> block.
       - List all relevant events from the memories in chronological order.
       - Explicitly calculate the time difference between the event and the "Current Reference Date".
       - Resolve any relative time references (e.g., "last year") using the memory's timestamp.
    3. Answer the question based *only* on the memories.
    4. **EXTRACTIVE ANSWERING (CRITICAL for F1 Score)**:
       - **Titles**: ALWAYS enclose book titles, song titles, movie titles, or art piece titles in double quotes (e.g., "Charlotte's Web").
       - **Dates**: 
         - If the question asks "When", output the date in "D Month YYYY" format (e.g., "7 May 2023").
         - **DO NOT** use ISO format (YYYY-MM-DD).
         - **DO NOT** include "On" before the date.
       - **Durations**: 
         - If the memory explicitly says "since [Year]", output "Since [Year]".
         - Otherwise, give the simplest form (e.g., "4 years").
       - **Yes/No/Likelihood**:
         - If the answer is not explicitly stated but can be inferred, use "Likely Yes" or "Likely No".
         - If the question asks "Would...", use "Likely Yes/No" if you are very unsure about the answer.
         - **IMPORTANT**: You can add explanations for "Would..." and "How/Why" questions when the answer is uncertain.
       - **Descriptive Questions (How/Why/What does X think)**:
         - Keep answers **EXTREMELY CONCISE**.
         - Use short phrases instead of full sentences where possible.
         - **DO NOT** be verbose. (e.g., instead of "She felt at one with the universe...", say "in awe of the universe").
       - **General Constraints**:
         - **DO NOT** include filler words like "The answer is", "About", "It seems", "On", "In", "Her", "His".
         - **Entities**: Output the specific name (e.g., "Sweden" instead of "Her home country"). Resolve pronouns to specific names.
         - **Wording**: Use the exact terminology from the memory if possible.
         - **Conciseness**: Remove unnecessary adjectives (e.g., "sunset" instead of "sunset-inspired painting").
         - **Lists**: If multiple items, separate by commas (e.g., "pottery, camping, painting"). Do not include extra descriptions.

    # RETRIEVED KNOWLEDGE:
    {context_str}

    Question: "{state['query']}"
    
    Output Format:
    <Thinking>
    [Chronological analysis and time calculations]
    </Thinking>
    Answer: [Your final answer]
    """
        
        # Increased max_tokens significantly to allow for CoT + Answer
        raw_answer = self.llm.generate(prompt, max_tokens=1500)
        
        # Post-processing to remove <Thinking> block and keep only the answer
        import re
        final_answer = raw_answer
        
        # Robust cleaning to isolate the answer
        # 1. Try to split by closing think tag (case insensitive, various formats)
        split_match = re.split(r'</Thinking>', raw_answer, flags=re.IGNORECASE)
        if len(split_match) > 1:
            # The last part is usually the answer
            final_answer = split_match[-1].strip()
        else:
            # 2. Fallback: Identify "Answer:" keyword (case insensitive)
            # Find the LAST occurrence of "Answer:" to avoid matching "The answer: ..." in thought process
            answer_indices = [m.start() for m in re.finditer(r'(?i)answer:', raw_answer)]
            if answer_indices:
                final_answer = raw_answer[answer_indices[-1]:]
                # Remove the "Answer:" prefix itself
                final_answer = re.sub(r'^(?i)answer:\s*', '', final_answer).strip()
            else:
                 # 3. Last resort: simple replace
                 final_answer = final_answer.replace('<Thinking>', '').replace('<Thinking>', '').strip()
            
        final_answer = re.sub(r"^The answer is\s*", "", final_answer, flags=re.IGNORECASE)
        final_answer = final_answer.strip('"').strip("'")
        
        # One more check if "Answer:" remains at start
        if final_answer.lower().startswith("answer:"):
             final_answer = final_answer[7:].strip()

        # Final cleanup for newlines that might be left over
        final_answer = final_answer.strip()

        state['final_answer'] = final_answer
        return state

