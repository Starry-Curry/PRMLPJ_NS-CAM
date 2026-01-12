# NS-CAM: Neuro-Symbolic Chrono-Agentic Memory

**NS-CAM** is a next-generation memory architecture for LLM agents designed to solve the "Long-Context Forgetfulness" problem. Unlike traditional RAG or simple vector stores, NS-CAM combines high-fidelity **Episodic Memory** (Vector) with a logical **Semantic Chrono-Graph** (Knowledge Graph) to provide agents with both intuitive recall and rigorous temporal reasoning capabilities.

---

## 🧠 Core Architecture

NS-CAM integrates two distinct memory systems mimicking the human brain's hippocampus and neocortex interaction:

### 1. Holographic Chrono-Memory (The Storage Layer)

- **Episodic Stream (Vector Store)**: Stores raw conversation slices ("Flashbulb Memories") with rich metadata (Time, Speaker, Sentiment). Powered by **Mem0 (Local)** and **ChromaDB**.
- **Semantic Chrono-Graph (Knowledge Graph)**: Stores entities and relationships as **Hyper-Edges** containing:
  - **Time Windows** ($[\tau_{start}, \tau_{end}]$) for conflict resolution (e.g., "Where *did* I live?" vs "Where *do* I live?").
  - **Confidence Scores** ($\sigma$) updated via Bayesian inference.

### 2. Agentic Cortex (The Reasoning Layer)

Built on **LangGraph**, the system employs specialized cognitive agents:

- **🕵️ The Profiler**: Routes queries to the appropriate memory subsystem (Recall vs. Reasoning).
- **🕸️ Associative Retriever**: Performs "Activation Spreading" on the graph to find logically connected context.
- **⏳ Temporal Reasoner**: Resolves factual conflicts by analyzing time windows instead of relying on LLM hallucination.

---

## 📂 Project Structure

```
NS-CAM/
├── docs/               # Architecture designs (plan_V2.md) and dev notes
├── src/
│   ├── agents/         # LangGraph workflows and cognitive agents
│   ├── logic/          # Knowledge Graph logic (NetworkX wrapper)
│   ├── storage/        # DualMemoryStore implementation
│   ├── models/         # Pydantic data models (MemoryObject, GraphEdge)
│   └── evaluation/     # Metrics (ROUGE, F1, GPT-4 Grading)
├── tests/              # Integration tests and evaluation scripts
├── data/               # LoCoMo benchmark dataset
└── run_demo.py         # Simple usage example
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (Recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/YourUsername/NS-CAM.git
   cd NS-CAM
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: This project includes a local version of `mem0` for customization.*
3. **Configuration**
   Create a `.env` file in the root directory with your API keys:

   ```env
   # LLM Provider (DeepSeek / OpenAI)
   OPENAI_API_KEY=sk-xxxx
   OPENAI_API_BASE=https://api.deepseek.com/v1  # or other compatible provider

   # Embedding Provider (Aliyun / OpenAI)
   EMBEDDING_API_KEY=sk-xxxx
   EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

   # Mem0 api
   MEM0_API_KEY=# your api
   MEM0_PROJECT_ID=# your project id
   MEM0_ORGANIZATION_ID=# your organization id
   ```

## 🧪 Running Benchmarks

NS-CAM is trained and evaluated on the **LoCoMo (Long Context Modeling)** benchmark.

To run the evaluation pipeline on a sample conversation:

```bash
python tests/base_v2/test_run_locomo_eval.py
```

Results will be saved to `results/locomo_eval/`.(the results out put: `~\results\locomo_eval\eval_result.json`)

## 🛠️ Usage Example

```python
from src.storage.dual_memory_store import DualMemoryStore
from src.agents.langgraph_workflow import NSCAMWorkflow

# Initialize
store = DualMemoryStore()
workflow = NSCAMWorkflow(store)

# Add Memory
store.add_user_memory(" I moved to Tokyo in 2022.")

# Query
response = workflow.run("Where do I live currently?")
print(response['final_answer'])
# Output: "You currently live in Tokyo (since 2022)."
```

## 📄 License

[MIT License](LICENSE)
