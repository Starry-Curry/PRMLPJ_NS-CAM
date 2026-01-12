# NS-CAM Iteration 1.1: Neuro-Symbolic Logic & Temporal Refinement

**Version**: 1.1.0
**Date**: 2026-01-10
**Status**: Planned / Pending Implementation

## 1. Overview

This iteration focuses on elevating the cognitive fidelity of the system. We are moving from "hardcoded heuristics" to "semantic reasoning" in three key areas:

1. **Temporal Resolution**: Distinguishing between *recording time* (System Lifecycle) and *event time* (Semantic Time).
2. **State Logic**: Replacing static whitelist with LLM-based *Cardinality Inference* (Smart State Machine).
3. **Retrieval Robustness**: Replacing string matching with *Vector-Anchored Entity Linking*.

---

## 2. Detailed Design

### 2.1 Feature: Holographic Chrono-Edge (Time Refinement)

**Goal**: Decouple system logic (archiving versions) from world logic (history events).

#### Formal Definition

The Graph Hyper-Edge $e_{uv}$ is redefined as a 7-tuple:

$$
e_{uv} = \langle r, w, \mathcal{T}_{sys}, \mathcal{T}_{sem}, \sigma, s, \mathcal{A} \rangle
$$

* **$\mathcal{T}_{sys}$ (System Lifecycle)**: `[record_time, archive_time]`. The database valid time. Used for versioning and conflict checking.
* **$\mathcal{T}_{sem}$ (Semantic Time)**: `[start_str, end_str]`. The real-world event time (extracted from text). Used for reasoning.

#### Data Schema Updates (`src/models/data_models.py`)

1. **New Enums & Models**:

   ```python
   from enum import Enum
   from typing import Optional, Tuple
   from pydantic import BaseModel, Field

   class TimeType(str, Enum):
       POINT = "point"       # e.g., "visited museum at 2pm"
       DURATION = "duration" # e.g., "lived in London 2020-2022"
       UNKNOWN = "unknown"

   class SemanticTime(BaseModel):
       time_type: TimeType
       start_str: Optional[str] = None  # ISO8601 formatting preferred
       end_str: Optional[str] = None
       raw_desc: Optional[str] = None   # Backup natural language, e.g. "last summer"

   class GraphEdge(BaseModel):
       # ... existing ...
       relation: str

       # [UPDATED] System valid window (Unix Timestamps)
       # [record_time, archive_time]. If active, archive_time is infinity.
       system_window: Tuple[float, float] = Field(description="[record_time, archive_time]")

       # [NEW] Real-world event time
       semantic_time: SemanticTime

       status: Literal['active', 'archived']
       attributes: Dict = Field(default_factory=dict)
   ```
2. **Migration & Usage**:

   * Ingestion sets `system_window[0] = current_time`, `system_window[1] = inf`.
   * Conflict resolution sets `system_window[1] = current_time` for the OLD edge.
   * Reasoning agent uses `semantic_time` to answer "When did X happen?".

### 2.2 Feature: LLM-Driven Smart State Machine

**Goal**: Dynamic conflict handling based on common sense rather than a hardcoded `STATE_CHANGE_WHITELIST`.

#### Logic Flow (`src/logic/knowledge_manager.py`)

When adding a new triple $(S, P, O_{new})$:

1. **Check Cache**: Look up $P$ in `self.predicate_rules` (Dict mapping predicate -> Cardinality).
2. **LLM Inference (If Miss)**:
   * Query LLM with `PROMPT_CARDINALITY_CHECK`.
   * Determine if $P$ is **Exclusive (ToOne)** or **Concurrent (ToMany)**.
3. **Execute & Cache**:
   * **Exclusive** (e.g., `is_located_in`, `has_job`):
     Archive ANY existing active edge $(S, P, O_{old})$ where $O_{old} \neq O_{new}$.
   * **Concurrent** (e.g., `visited`, `met_with`):
     Do NOT archive. Simply append the new edge.
   * Store result in cache.

#### Prompt Design

```text
Analyze the relationship predicate: "{predicate}".

Determine its temporal exclusivity for a human subject based on common sense.
- "Exclusive": A person can typically only have ONE active value for this at a specific moment in time. (e.g., 'is_located_in', 'current_job', 'marriage_status').
- "Concurrent": A person can have MULTIPLE active values for this logic simultaneously or accumulate them over time. (e.g., 'visited', 'likes_food', 'friend_with', 'attended_event').

Predicate: "{predicate}"
Logic (Return only 'Exclusive' or 'Concurrent'):
```

### 2.3 Feature: Vector-Anchored Retrieval

**Goal**: Enable fuzzy matching for graph entry points (e.g., User asks about "art show" -> System finds node "Exhibition Center").

#### Architecture Updates (`src/storage/dual_memory_store.py`)

* **New Collection**: `nodes_collection` (ChromaDB collection).
* **Ingestion Hook**: Inside `add_graph_edge` (or `add_knowledge_triple`), check if Source or Target nodes effectively exist. If new, embed their string names and add to `nodes_collection`.

#### Retrieval Pipeline (`src/agents/agent_nodes.py`)

Refactor `associative_retriever_node` to:

1. **Entity Extraction**: Use LLM (or lightweight heuristics) to extract potential entities from the user's `Query`.
   * *Query*: "What happened at the art show?" -> *Entities*: `["art show"]`.
2. **Vector Search**: Query `nodes_collection` with the extracted entity strings.
   * *Result*: `["Exhibition_Center", "Art_Gallery"]` (Top-K).
3. **Seeding**: Use these Top-K result nodes as **Seeds**.
4. **Spreading Activation**: Perform standard random walk/BFS from seeds to gather context.

# NS-CAM Iteration 1.2: Retrieval Stabilization & Reasoning Injection

**Version**: 1.2.0
**Date**: 2026-01-11
**Status**: Implemented / In Testing

## 1. Overview

This iteration addresses critical failures observed during the LOCOMO dataset small-scale verification. The primary issues stemmed from "Chain of Thought" leakage in generation, rigid retrieval routing causing information loss, and environmental instability with local embeddings.

**Key Achievements:**

- **Recall Restoration**: Fixed logic that accidentally muted vector retrieval when graph retrieval failed, ensuring 100% coverage.
- **Hallucination Control**: Injected structured reasoning hints and enforced `<think>` block separation to prevent internal monologues from leaking into final answers.
- **Stability**: Migrated to a robust, soft-dependency architecture for embeddings to prevent crashes on non-GPU environments.

---

## 2. Key Optimizations

### 2.1 Logic: Unconditional Vector Fallback

**Problem**: The `episodic_retriever_node` contained a defensive check `if intent == 'retrieve_vec'`. When the Profiler classified a query as `retrieve_graph`, but the graph return was empty (common for abstract queries), the system skipped vector retrieval entirely, leading to "I don't have enough information."

**Solution**:

- **Strategy**: "Always-On" Vector Retrieval.
- **Implementation**: Removed conditional logic in `episodic_retriever_node`. It now executes whenever the workflow router points to it.
- **Routing Update**: Modified `langgraph_workflow.py` to route Graph failures or partial hits to Vector retrieval aggressively.

### 2.2 Generation: Structured Reasoning & Output Cleaning

**Problem**: The Generator model (DeepSeek) often outputted its thinking process ("Checking memory... Found X..."), which the evaluation metric (F1/ROUGE) penalized heavily as it wasn't the direct answer.

**Solution**:

- **Explicit `<think>` Block**: Updated system prompt to mandate:
  ```xml
  <think>
  [Reasoning Steps]
  </think>
  [Final Answer]
  ```
- **Post-Processing**: Added regex logic to strip `<think>...</think>` tags and "Answer:" prefixes, ensuring only the clean answer is returned to the user/evaluator.
- **Reasoner Injection**: The `reasoner_node` now pre-calculates `[System Reasoner]: The current active value is X` and injects it into the context, reducing the burden on the Generator to deduce state from raw edges.

### 2.3 Environment: Lazy-Loading Embedder

**Problem**: `sentence-transformers` caused script hangs or crashes on initialization in some environments due to `torch` conflicts.

**Solution**:

- **Soft Dependency**: Wrapped imports in `try/except` and `importlib` checks.
- **Lazy Loading**: The heavy model loading now happens only on the first `embed()` call, not at module level.
- **Mock Fallback**: If `torch` fails, the system seamlessly downgrades to deterministic hash-based embeddings for testing continuity.

### 2.4 Interface: Robust Triple Extraction

**Problem**: The LLM often returned natural language intros or malformatted tuples during knowledge extraction.

**Solution**:

- **Parsing logic**: Added a robust line-by-line parser that looks for `(A, B, C, D, E)` patterns, ignoring surrounding text.
- **Prompt Refinement**: Enhanced the extraction prompt to strictly demand 5-tuple format `(Subject, Predicate, Object, Time, Type)`.

---

## 3. Results (LOCOMO Sample 26)

After applying changes 2.1 and 2.2:

| Metric                         | Before Fix | After Fix              |
| :----------------------------- | :--------- | :--------------------- |
| **Category 4 F1**        | 0.12       | **0.22**         |
| **Category 2 LLM Score** | 2.9        | **Dynamic High** |
| **Retrieval Failures**   | High       | **Near Zero**    |

## 4. Next Steps (Iteration 1.3)

See below for detailed Iteration 1.3 log.

# NS-CAM Iteration 1.3: Dual-Brain Architecture (Mem0 Integration)

**Version**: 1.3.0
**Date**: 2026-01-11
**Status**: Implemented / In Verification

## 1. Overview

This iteration introduces the **Dual-Brain Architecture**, a hybrid system combining the rigorous temporal logic of **NS-CAM Graph** (Left Brain) with the rich, semantic, and adaptive memory of **Mem0** (Right Brain).

**Key Drivers:**

- **Signal-to-Noise Ratio**: Raw vector search (ChromaDB) on conversation chunks was retrieving too much noise. Mem0's "Fact Extraction" automatically distills user messages into atomic facts, improving context quality.
- **Statefulness in Unstructured Data**: Mem0 handles the deduping and updating of semantic facts (e.g., "User likes spicy food" -> "User no longer likes spicy food") which raw vectors missed.
- **Complementary Retrieval**: The Graph provides the "Skeleton" (Time, Location, Relationships), while Mem0 provides the "Flesh" (Preferences, History, Nuance).

---

## 2. Technical Design

### 2.1 Storage Layer: Mem0 Integration (`src/storage/dual_memory_store.py`)

- **Replacement**: The manual `episodic_collection` (ChromaDB) is deprecated.
- **New Engine**: `mem0.MemoryClient` is integrated as the core episodic engine.
- **Ingestion Pipeline**:
  1. **Preprocessing**: Relative time resolution (e.g., "yesterday" -> "2023-05-07") via LLM.
  2. **Fact Extraction**: Mem0 client extracts facts from the resolved text.
  3. **Storage**: Facts are stored in Mem0's vector store (Qdrant/Chroma compatible).
- **Configuration**:
  - `custom_instructions`: Injected to guide Mem0 to focus on personal details, emotions, and specific milestones (derived from LOCOMO optimization).

### 2.2 Processing Layer: Adaptive Retrieval (`src/agents/agent_nodes.py`)

- **Episodic Retriever**: Now calls `Mem0.search()` instead of raw vector query.
- **Generator**:
  - **Prompt Engineering**: Adopted the **Timeline-Based CoT (Chain of Thought)** prompt from Mem0's optimization suite.
  - **Constraint Enforcement**: Added strict rules for Date formatting ("D Month YYYY") and Title handling (double quotes) to maximize benchmark scores (F1/BLEU).

### 2.3 Workflow: The Dual-Path

User Query -> **Profiler** -> **Parallel Retrieval**
   |
   +--> **Graph Retrieval** (Left Brain) -> Structured Facts (Active/Archived)
   |
   +--> **Mem0 Search** (Right Brain) -> Semantic Memories (Preferences/Narratives)
   |
   v
**Reasoner & Generator** -> Synthesis -> Final Answer

---
