import networkx as nx
from typing import List, Optional, Dict
import os
import json
import uuid
import time
from datetime import datetime
from src.storage.memory_interface import MemoryStoreInterface
from src.models.data_models import MemoryObject, GraphEdge
from src.utils.logging_setup import get_logger
from src.utils.llm_interface import LLMInterface
# Use local Memory implementation instead of Client for finer control and avoiding network/credits if possible
try:
    from mem0.memory.main import Memory
    from mem0.configs.base import MemoryConfig, LlmConfig, EmbedderConfig, VectorStoreConfig
    MEM0_LOCAL_AVAILABLE = True
except ImportError:
    MEM0_LOCAL_AVAILABLE = False
    from mem0 import MemoryClient
    
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

# Custom Instructions for Mem0 (Optimized for Locomo/Personal Agents)
# REVISED: Must ensure JSON list output format is maintained while asking for detailed richness.
MEM0_CUSTOM_INSTRUCTIONS = """
You are a Personal Memory Assistant. Extract relevant memories from the user's messages.

Guidelines:
1. **Format**: Return a JSON object with a key "facts" containing a LIST of strings. 
   Example: {"facts": ["Melanie is researching adoption agencies.", "She feels anxious about the process."]}

2. **Content Richness**:
   - Each string in the list should be a self-contained, detailed sentence.
   - Include specific names, dates (if available), and context.
   - Capture emotions, plans, and identity-related details.
   - Avoid generic phrases like "User said". Use "The user" or specific names if known.

3. **Focus Areas**:
   - Identity & Relationships (family, LGBTQ+ context, friends)
   - Activities & Hobbies (specifics like 'painting a sunrise', not just 'art')
   - Career & Education (aspirations, specific fields)
   - Emotional Journey (feeling supported, worried, excited)

4. **Extraction Logic**:
   - Extract ONLY from the User's message. 
   - Ignoring Assistant's chatter.
   - Consolidate fragmented info into complete sentences.
"""

# Mock ChromaDB if not available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not found. Using in-memory mock.")

class MockChromaCollection:
    def __init__(self, name):
        self.name = name
        self.data = {} # id -> (doc, embedding, metadata)

    def add(self, documents, embeddings, metadatas, ids):
        for doc, emb, meta, i in zip(documents, embeddings, metadatas, ids):
            self.data[i] = (doc, emb, meta)

    def query(self, query_embeddings, n_results=5):
        # Dummy logical return: simply return first n items
        # In real life this would do cosine similarity
        items = list(self.data.items())[:n_results]
        ids = [[k for k, _ in items]]
        documents = [[v[0] for _, v in items]]
        metadatas = [[v[2] for _, v in items]]
        return {'ids': ids, 'documents': documents, 'metadatas': metadatas}

class MockChromaClient:
    def __init__(self, path):
        self.collections = {}

    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = MockChromaCollection(name)
        return self.collections[name]

class DualMemoryStore(MemoryStoreInterface):
    def __init__(self, persist_dir: str = "./data"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize Mem0 (Local Preferred)
        self.mem0 = None
        self.is_local_mem0 = False

        if MEM0_LOCAL_AVAILABLE:
            try:
                # Create Config for Local Memory
                from src.utils import config as app_config
                
                # Configure LLM to match App Config (DeepSeek/Aliyun via OpenAI Protocol)
                llm_config = LlmConfig(
                    provider="openai",
                    config={
                        "model": app_config.OPENAI_MODEL,
                        "api_key": app_config.OPENAI_API_KEY,
                        "openai_base_url": app_config.OPENAI_BASE_URL, 
                        "temperature": 0.0
                    }
                )
                
                # Configure Embedder (Aliyun text-embedding-v2 or similar)
                embedder_config = EmbedderConfig(
                    provider="openai",
                    config={
                        "model": app_config.OPENAI_EMBEDDING_MODEL,
                        "api_key": app_config.OPENAI_API_KEY,
                        "openai_base_url": app_config.OPENAI_BASE_URL
                    }
                )

                config = MemoryConfig(
                    llm=llm_config, 
                    embedder=embedder_config
                )
                
                # optimize fact extraction for detail
                config.custom_fact_extraction_prompt = MEM0_CUSTOM_INSTRUCTIONS
                
                # set history db path
                config.history_db_path = os.path.join(persist_dir, "mem0_history.db")
                
                # Configure Vector Store to be persistent in 'persist_dir'
                # Default is Chroma. We need to set the path.
                if config.vector_store.provider == "chroma":
                    # Check structure of vector_store config
                    if hasattr(config.vector_store, 'config'):
                         if config.vector_store.config is None:
                             config.vector_store.config = {}
                         
                         if isinstance(config.vector_store.config, dict):
                             config.vector_store.config['path'] = os.path.join(persist_dir, "chroma_mem0")
                             config.vector_store.config['collection_name'] = "mem0_episodic"
                
                self.mem0 = Memory(config)
                self.is_local_mem0 = True
                logger.info(f"Initialized Local Mem0 Memory (Model: {app_config.OPENAI_MODEL})")
                
            except Exception as e:
                logger.warning(f"Failed to init Local Mem0: {e}. Fallback to Cloud Client.")
        
        # Fallback to Cloud Client
        if not self.mem0:
            try:
                self.mem0 = MemoryClient(
                    api_key=os.getenv("MEM0_API_KEY"),
                    org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                    project_id=os.getenv("MEM0_PROJECT_ID"),
                )
                try:
                    self.mem0.update_project(custom_instructions=MEM0_CUSTOM_INSTRUCTIONS)
                    logger.info("Mem0 Cloud Project Instructions Updated.")
                except Exception as e:
                    logger.warning(f"Could not update Mem0 instructions: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize Mem0 Client: {e}. Episodic memory will fail.")
                self.mem0 = None

        # Initialize LLM for Pre-processing (Time Resolution)
        self.llm = LLMInterface()

        # Initialize Vector Store (ChromaDB) - KEEPing Node Store for Graph Anchoring
        if CHROMA_AVAILABLE:
            logger.info(f"Initializing ChromaDB at {os.path.join(persist_dir, 'chroma')}")
            self.chroma_client = chromadb.PersistentClient(path=os.path.join(persist_dir, "chroma"))
        else:
            self.chroma_client = MockChromaClient(path=os.path.join(persist_dir, "chroma"))
            
        # self.episodic_collection = self.chroma_client.get_or_create_collection(name="episodic_memory") # DEPRECATED
        self.nodes_collection = self.chroma_client.get_or_create_collection(name="graph_nodes")
        # Local cache to avoid re-adding existing nodes frequently
        self._known_nodes = set()

        # Initialize Graph Store (NetworkX)
        self.graph_path = os.path.join(persist_dir, "graph.json")

        self.graph = nx.MultiDiGraph() # MultiDiGraph to allow multiple edges between nodes (time variants)
        self._load_graph()

    def _resolve_relative_time(self, text: str, timestamp: float) -> str:
        """
        Resolve relative time references (yesterday, next week) to absolute dates.
        """
        # Format timestamp to readable date
        current_date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        prompt = f"""
        Current Date: {current_date_str}
        
        Rewrite the following text, resolving any relative time references (like 'yesterday', 'next week', 'last month') to absolute dates (YYYY-MM-DD) based on the Current Date.
        Keep the rest of the text unchanged. Do not add any explanations or extra text.
        
        Text: {text}
        """
        return self.llm.generate(prompt)

    def _load_graph(self):
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                logger.info("Graph loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")

    def _save_graph(self):
        try:
            data = nx.node_link_data(self.graph)
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def persist(self):
        """Save graph to disk. Chroma matches auto-persists."""
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Graph persisted.")

    def add_episodic_memory(self, memory: MemoryObject) -> str:
        """
        Add episodic memory via Mem0.
        """
        if not self.mem0:
            logger.error("Mem0 not initialized.")
            return ""

        # Ensure timestamp is in metadata
        if 'timestamp' not in memory.raw_meta:
            memory.raw_meta['timestamp'] = memory.timestamp

        # 1. Resolve Relative time
        resolved_content = self._resolve_relative_time(memory.content, memory.timestamp)
        
        # 2. Add to Mem0
        user_id = memory.raw_meta.get('user_id', 'default_user')
        
        try:
            # If using local Mem0, we can rely on our configured prompt.
            # If using Cloud Client, instructions were updated in init.
            
            # Note: For strict adherence to prompts, we ensure 'messages' format.
            self.mem0.add(
                messages=[{"role": "user", "content": resolved_content}],
                user_id=user_id,
                metadata=memory.raw_meta
            )
            logger.info(f"Added episodic memory to Mem0 (User: {user_id}). Local: {self.is_local_mem0}")
        except Exception as e:
            logger.error(f"Mem0 add failed: {e}")
            
        return memory.id

    def retrieve_episodic(self, query_vector: List[float], k: int = 5, query_text: str = None, user_id: str = 'default_user') -> List[MemoryObject]:
        """
        Retrieve from Mem0. 
        """
        if not self.mem0:
            return []
            
        if not query_text:
            logger.warning("Mem0 retrieval requires query_text. Returning empty.")
            return []

        try:
            # Mem0 search
            results = self.mem0.search(
                query=query_text,
                user_id=user_id,
                limit=k
            )

            memories = []
            if results: 
                if isinstance(results, dict):
                     results = results.get('results', [])

                for res in results:
                    content = res.get('memory', '')
                    mem_id = res.get('id', str(uuid.uuid4()))
                    score = res.get('score', 0.0)
                    meta = res.get('metadata', {}) or {} # fallback empty dict
                    
                    # 1. Try 'timestamp' in metadata
                    ts = meta.get('timestamp')
                    if not ts:
                         ts = time.time()
                    
                    # Store score in raw_meta instead of as direct attribute
                    meta['_search_score'] = float(score)
                    
                    mem = MemoryObject(
                        id=mem_id,
                        content=content,
                        timestamp=float(ts) if ts else time.time(),
                        embedding=[], 
                        raw_meta=meta
                    )
                    memories.append(mem)
            
            return memories

        except Exception as e:
            logger.error(f"Mem0 search failed: {e}")
            return []

    def add_node_embedding(self, node_name: str, embedding: List[float]):
        """
        Add or update a node's embedding index.
        """
        if node_name in self._known_nodes:
            return  # Optimization
            
        self.nodes_collection.add(
            documents=[node_name],
            embeddings=[embedding],
            ids=[f"node_{node_name}"],
            metadatas=[{"name": node_name}]
        )
        self._known_nodes.add(node_name)

    def search_nodes(self, query_vector: List[float], k: int = 5) -> List[str]:
        """
        Return list of node names closest to query vector.
        """
        results = self.nodes_collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        node_names = []
        if results and results.get('ids') and len(results['ids']) > 0:
            metadatas = results.get('metadatas', [[]])[0]
            if metadatas:
                for m in metadatas:
                    # m is dict or None
                    if m and 'name' in m:
                        node_names.append(m['name'])
        
        # Fallback to pure document text if metadata fail?
        # Usually metadata works.
        return node_names

    def add_graph_edge(self, edge: GraphEdge) -> None:
        """
        Add edge to NetworkX graph. 
        We use edge.id as the key in MultiDiGraph to allow updates.
        """
        attr = edge.dict(exclude={'source', 'target', 'id'})
        # Use edge.id as the key. This ensures updates if ID exists, or new if unique ID.
        self.graph.add_edge(edge.source, edge.target, key=edge.id, **attr)
        logger.info(f"Added/Updated graph edge: {edge.source} -[{edge.relation}]-> {edge.target} (key={edge.id})")
        
        # Persist graph
        self._save_graph()

    def get_graph_neighbors(self, node_id: str) -> List[GraphEdge]:
        edges = []
        if self.graph.has_node(node_id):
            # Outgoing edges
            # self.graph[node_id] is a dict of neighbors
            for nbr, datadict in self.graph[node_id].items():
                # datadict is {key: {attrs}}
                for key, attr in datadict.items():
                    try:
                        edge = GraphEdge(
                            id=str(key),
                            source=node_id,
                            target=nbr,
                            **attr
                        )
                        edges.append(edge)
                    except Exception as e:
                        logger.warning(f"Skipping malformed edge {node_id}->{nbr}: {e}")
        return edges

    def get_node_edges(self, source: str, relation: Optional[str] = None) -> List[GraphEdge]:
        """Get all outgoing edges from source, optionally filtered by relation type."""
        all_edges = self.get_graph_neighbors(source)
        if relation:
            return [e for e in all_edges if e.relation == relation]
        return all_edges
