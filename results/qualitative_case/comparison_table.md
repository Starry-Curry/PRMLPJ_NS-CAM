# Output Comparison

### Output Comparison

| System | Response | Error Analysis |
| :--- | :--- | :--- |
| **RAG (Baseline)** | "You work at Google and OpenAI." | **Hallucination/Merge**: Retrieves all chunks without temporal awareness. |
| **Mem0 (Baseline)** | "You are a senior engineer at OpenAI." | **Recency Bias**: Overwrites old memory, loses history. |
| **NS-CAM (Ours)** | "You worked at Google right before joining OpenAI." | **Correct**: Utilizes graph state versions and temporal reasoning. |
