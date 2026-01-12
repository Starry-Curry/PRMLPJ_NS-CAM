为了让这张架构图达到 **ICLR / NeurIPS** 等顶级计算机视觉/AI 会议的标准，我们需要摒弃过于具象的“机械脑”、“齿轮”等隐喻（Metaphors），转而采用**计算思维（Computational）**和**模块化（Modular）**的表达方式。

顶会论文的架构图通常强调：**信息流向（Data Flow）**、**模块解耦（Decoupling）**以及**抽象逻辑（Abstraction）**。

以下是为您重新设计的 Prompt，旨在生成一张**学术级、扁平化、逻辑严密**的系统架构图。

---

### Figure 1: The Dual-Brain Architecture (Schematic Overview)

**设计核心：**

* **Top (Cortex):** 不再是齿轮，而是一个**抽象的控制单元（Control Unit）**或**路由节点（Router Node）**，体现“调度”和“分发”。
* **Left (Graph):** 强调**拓扑结构**和**时间刻度**，体现“逻辑严谨”。
* **Right (Stream):** 强调**流式数据**和**动态堆栈**，体现“语义适应”。
* **整体布局:** 采用经典的**分层架构（Hierarchical Layout）**，自上而下控制，左右并行处理，底部汇聚输出。

**Prompt:**

> **Subject:** A professional computer science system architecture diagram, hierarchical layout, white background.
>
> **1. Top Layer (The Orchestrator):** A centralized, abstract icon representing the "Agentic Cortex". Visualized as a **complex circuit hub** or a **neural network controller node** (not a biological brain). Arrows branch out downwards from this hub to the left and right.
>
> **2. Left Branch (The Logical Brain):** A defined rectangular region containing a **sparse knowledge graph**.
>
> * **Visuals:** Geometric nodes (circles) connected by straight lines.
> * **Detail:** On the connecting lines, draw tiny **perpendicular hash marks** or brackets `[ ]` to symbolize "Time Windows" or intervals.
> * **Color:** Clinical, deep science blue tones.
>
> **3. Right Branch (The Semantic Brain):** A defined rectangular region containing a **dynamic data stream**.
>
> * **Visuals:** A cascading stream of **rounded data cards** or text blocks, arranged in a way that suggests a continuous feed or log. Some cards appear to be merging or updating.
> * **Color:** Warm, energetic orange/amber tones.
>
> **4. Bottom Layer (Integration):** The outputs from both the Left Graph and Right Stream flow downwards into a unified processing block.
>
> * **Visuals:** A funnel-like or convergence symbol leading into a final rectangular block representing the "Reasoning Injection & Generator".
>
> **Style:** Flat academic vector art, schematic blueprint style. High contrast, clean thin lines, distinct separation of modules. No 3D rendering, no shadows, no gradients. Strictly 2D isometric or flat frontal view. Professional and minimalist.

---

### 🎨 生成后后期处理指南 (Post-Processing Labels)

AI 生成底图后，请使用 PPT 或 Illustrator 将图片中的伪文本覆盖，替换为以下标准的学术术语（与论文一致）：

1. **顶层模块 (Top Hub)**:

   * Label: **Agentic Cortex (Orchestrator)**
   * Sub-label: *Always-On Retrieval Policy*
2. **左侧区域 (Left Region)**:

   * Title: **Left Brain: Semantic Chrono-Graph**
   * Keywords to add near nodes: *Logic*, *Structure*, *$\mathcal{T}_{sys}$ / $\mathcal{T}_{sem}$*
3. **右侧区域 (Right Region)**:

   * Title: **Right Brain: Adaptive Semantic Stream**
   * Keywords to add near cards: *Mem0 Engine*, *Fact Extraction*, *Nuance*
4. **底部模块 (Bottom Block)**:

   * Label: **Generator with Reasoning Injection**
   * Action: *Context Synthesis $\rightarrow$ CoT Filtering $\rightarrow$ Final Output*

### 为什么这个 Prompt 更学术？

1. **去除了“Gear/Mechanical”**：在现代 AI 论文中，齿轮通常代表“硬编码”或“旧工业时代”，而“Circuit/Neural Hub”代表智能和算法。
2. **强调了“Data Stream”而非“Floating Documents”**：右脑使用的是 Mem0，本质上是流式数据的处理（Stream Processing）和向量检索，用“Cascading Cards”（级联卡片）比“漂浮的纸”更符合计算机科学对数据的表达。
3. **视觉区分明确**：蓝色（冷/逻辑） vs 橙色（暖/语义）的对比，是学术图表中展示 **Hybrid Architecture（混合架构）** 的经典配色方案，能帮助审稿人一眼看懂系统的双重性。
