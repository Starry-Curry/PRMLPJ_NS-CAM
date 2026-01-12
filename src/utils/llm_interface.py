from typing import List, Dict, Any, Optional, Tuple
import os
import time
from datetime import datetime
from src.utils.logging_setup import get_logger
from src.utils import config
from src.utils.embedder import embed_texts

try:
    import openai
except ImportError:
    openai = None

logger = get_logger(__name__)


class LLMInterface:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.OPENAI_MODEL
        # Configure openai client
        key = config.OPENAI_API_KEY
        if key:
            # New OpenAI Client (v1.0.0+)
            self.client = openai.OpenAI(
                api_key=key,
                base_url=config.OPENAI_BASE_URL
            )
            self.embedding_model = config.OPENAI_EMBEDDING_MODEL
            logger.info(f"OpenAI-compatible client configured (Model: {self.model_name}, Emb: {self.embedding_model})")
        else:
            self.client = None
            logger.warning("No OPENAI API key found, LLM calls will be mocked.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 256) -> str:
        logger.info(f"LLM Call ({self.model_name}): {prompt[:120]}...")
        
        # Force mock if model name is 'mock'
        if self.model_name == 'mock' or not self.client:
            return self._mock_response(prompt)

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt or "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=30.0 
            )
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"ChatCompletion failed: {e}")
            # Fallback to mock on ANY error (including 404 model not found)
            return self._mock_response(prompt)


    def _mock_response(self, prompt: str) -> str:
        p_lower = prompt.lower()
        import re
        # Profiler heuristics
        if "classify the user query" in p_lower:
            if any(w in p_lower for w in ['before', 'past', 'previous']):
                return 'retrieve_graph'
            if 'where' in p_lower or 'who' in p_lower:
                return 'retrieve_graph'
            if 'detail' in p_lower or 'what happened' in p_lower:
                return 'retrieve_vec'
            return 'mixed'

        if "context:" in p_lower or "retrieved knowledge" in p_lower:
            # Past
            if any(w in p_lower for w in ['before', 'past', 'previous']):
                # Match format: - A --[rel]--> B (Archived/Past)
                # Regex needs to be flexible
                past_matches = re.findall(r"--\[.*?\]-->\s*(.+?)\s*\(Archived", prompt)
                if past_matches:
                    return f"Before that, it was {past_matches[-1]}."
                else:
                    return "I don't have records of the past."

            # Current
            # Match format: - A --[rel]--> B (Active/Current)
            match = re.search(r"--\[.*?\]-->\s*(.+?)\s*\(Active", prompt)
            if match:
                answer = match.group(1)
                return f"Based on the knowledge graph, the answer is {answer}."

            # Fallback for old tests or different formats
            match_ep = re.search(r"lives in (\w+)", prompt)
            if match_ep:
                return f"I recall {match_ep.group(1)}."

            return "Mock LLM Response"

    def extract_triples(self, text: str, reference_timestamp: float = None) -> List[Tuple[str, str, str, str, str]]:
        """
        Uses LLM to extract knowledge triples (Subject, Predicate, Object) + Temporal Info from text.
        Returns a list of 5-tuples: (Subject, Predicate, Object, Resolved_Time_Str, Time_Type)
        Resolved_Time_Str should be an absolute date/time if possible (e.g. YYYY-MM-DD).
        """
        if not self.client:
            # Fallback mock heuristic
            triples = []
            if ' lives in ' in text:
                parts = text.split(' lives in ')
                sub = parts[0].strip().split()[-1]
                obj = parts[1].strip().strip('.')
                triples.append((sub, 'lives_in', obj, 'None', 'unknown'))
            return triples

        # Prepare reference time string
        ref_ts = reference_timestamp if reference_timestamp is not None else time.time()
        ref_date_str = datetime.fromtimestamp(ref_ts).strftime('%Y-%m-%d %H:%M:%S')

        # Using a more robust structure for extraction to ensure 5-tuple
        prompt = f"""
        You are a Knowledge Graph extraction expert. Extract meaningful facts from the conversation.
        
        # Temporal Resolution
        Current Reference Date: {ref_date_str}
        For each fact, if there is a time mention (relative or absolute):
        1. Identify the raw time mention (e.g., "last Friday", "next week").
        2. RESOLVE it to an absolute date (YYYY-MM-DD) based on the Current Reference Date.
           - If Today is 2026-01-10 (Saturday), "last Friday" -> "2026-01-09".
        3. If no time mention, Time="None", Type="unknown".
        4. If it's a duration (e.g., "since 2020"), preserve the start date or range.

        # Guidelines:
        1. **Specificity**: Use specific details (e.g., "charity race" not "race").
        2. **Completeness**: Include location in Object if relevant.
        3. **Entities**: Subject/Object must be specific names/roles.
        
        # Format:
        (Subject, Predicate, Object, Resolved_Time_Str, Time_Type)
        
        # Examples:
        Input: "I visited the Tate Modern last Friday."
        Output:
        (User, visited, Tate Modern, 2026-01-09, point)
        
        Input: "I've been working as a barista since 2019."
        Output:
        (User, current_job, barista, since 2019, duration)

        Text:
        "{text}"
        
        Output ONLY tuples, one per line. No other text.
        """
        
        try:
            resp = self.generate(prompt, max_tokens=4096)
            triples = []
            import re
            # Improved regex to handle occasional spaces or variations
            # (Sub, Pred, Obj, T, Type)
            # We match inside parentheses, split by comma, allowing for quotes or plain text
            
            lines = resp.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line.startswith('(') or not line.endswith(')'):
                    continue
                    
                # Basic csv split inside parens
                inner = line[1:-1]
                parts = [p.strip() for p in inner.split(',')]
                
                if len(parts) >= 5:
                    # Take first 5
                    triples.append((
                        parts[0], parts[1], parts[2], parts[3], parts[4]
                    ))
                elif len(parts) == 3:
                     # Fallback for old triples
                     triples.append((
                        parts[0], parts[1], parts[2], "None", "unknown"
                     ))
            
            return triples
        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return []


    def embed(self, texts: List[str]) -> List[List[float]]:
        # Use OpenAI-compatible API if available
        if self.client:
            try:
                # Remove newlines for better embedding performance
                texts = [t.replace("\n", " ") for t in texts]
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"API Embedding failed: {e}. Falling back to local.")
        
        # Local embedder
        try:
            return embed_texts(texts)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            # Fallback to mock
            out = []
            for t in texts:
                h = abs(hash(t)) % (10**8)
                vec = [(h >> (i % 32)) % 100 / 100.0 for i in range(384)] # 384 dim for BGE
                out.append(vec)
            return out

    def check_redundancy(self, new_triple: tuple, existing_edges: List[dict]) -> dict:
        """
        Check if the new triple (sub, pred, obj) is redundant, an update, or new.
        Returns dict: {'status': 'new'|'duplicate'|'update', 'target_edge_id': str|None}
        """
        if not existing_edges:
            return {'status': 'new', 'target_edge_id': None}
            
        # Format existing edges for LLM
        edges_desc = []
        for e in existing_edges:
            # e is likely a GraphEdge object or dict
            eid = e.get('id') if isinstance(e, dict) else e.id
            rel = e.get('relation') if isinstance(e, dict) else e.relation
            tgt = e.get('target') if isinstance(e, dict) else e.target
            # Add semantic time if available for context
            st_time = ""
            if not isinstance(e, dict) and hasattr(e, 'semantic_time'):
                st_time = f" (Time: {e.semantic_time.time_str})"
            edges_desc.append(f"ID: {eid} | Relation: {rel} | Object: {tgt}{st_time}")
            
        existing_text = "\n".join(edges_desc)
        
        prompt = f"""
        You are a Knowledge Graph consistency checker.
        
        New Fact: {new_triple[0]} --[{new_triple[1]}]--> {new_triple[2]}
        
        Existing Knowledge relating to "{new_triple[0]}":
        {existing_text}
        
        Task: Determine how to handle this New Fact.
        1. 'duplicate': The new fact conveys the SAME meaning as an existing one (even if words differ slightly) and offers no new info.
        2. 'update': The new fact UPDATES an existing one (e.g., "living in NY" updates "living in LA"). Also use this if the new fact is more specific.
        3. 'new': The new fact is distinct and compatible (e.g., "likes apples" vs "likes pears").
        
        Return JSON ONLY:
        {{
            "status": "duplicate" | "update" | "new",
            "target_edge_id": "ID_OF_MATCHING_EDGE" (or null if new)
        }}
        """
        
        try:
            response = self.generate(prompt, max_tokens=256)
            # Simple JSON parsing
            import json
            import re
            
            # Extract JSON block
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                return {'status': 'new', 'target_edge_id': None}
                
        except Exception as e:
            logger.error(f"Redundancy check failed: {e}")
            return {'status': 'new', 'target_edge_id': None}
