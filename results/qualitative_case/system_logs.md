# Qualitative Analysis Logs

### System Logs Simulation

**[Ingestion]** Input: "quit Google, moved to OpenAI"
**[Cardinality Check]** Predicate `work_at` is **EXCLUSIVE** (Max-Cardinality: 1).
**[Conflict Resolution]** Detected existing active edge: `(User, work_at, Google)`.
   -> Action: **ARCHIVE**. 
   -> Set `System_Time_End` = Now.
   -> State Update: `Google` [Active -> Archived].
**[Graph Update]** Created new edge: `(User, work_at, OpenAI)`.
   -> Status: **Active**.
   -> Semantic Time: `[T2, Now]`.

**[Retrieval]** Query: "Where did I work right before my current job?"
**[Query Analysis]** 
   -> Target: `work_at`
   -> Temporal Constraint: `before(current_job)`
**[Graph Search]** Found edges for `User` via `work_at`:
    1. Node: **OpenAI** | Status: Active   | Time: T2-Now
    2. Node: **Google** | Status: Archived | Time: T1-T2
**[Reasoning Injection]** 
<think> 
  Current job is identifier 'OpenAI' (from Active edge). 
  Query asks for "right before". 
  Comparing timestamps: Google (T1-T2) is immediately before OpenAI (T2-Now). 
  Answer: Google. 
</think>
