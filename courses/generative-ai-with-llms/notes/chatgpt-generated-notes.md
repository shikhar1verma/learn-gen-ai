Below is a clean, icon-free set of **action-oriented reference notes** for each week of *Generative AI with LLMs*. I trimmed peripheral detail and kept only the concepts you’ll lean on when talking shop, designing systems, or prepping interviews. Paste straight into Notion and add personal highlights as you progress.

---

## Week 1 – Foundations & Pre-Training

### 1. Generative AI Lifecycle (“Idea → Impact”)

| Phase | Core Questions | Key Takeaways |
| --- | --- | --- |
| **Model Selection** | Domain? size? open vs proprietary? | Pick smallest model that meets quality; saves cost for later adaptation. |
| **Pre-Training** | Data scale and diversity? objective? | Self-supervised next-token on **hundreds B tokens**; follow **Chinchilla** compute-optimal rule (≈ 20–30 tokens per param). |
| **Adaptation** | Full vs PEFT? instruction data? | Decide early: *task breadth* → instruction tuning; *domain depth* → domain adaptive pre-training. |
| **Evaluation** | Which benchmarks & in-house tests? | Combine public leaderboards (MMLU, HELM) with private, task-specific evals. |
| **Deployment & Monitoring** | Latency? cost? safety? | Async batching, token streaming, safety filters, continuous drift detection. |

### 2. Transformer Architecture (⚙️ cheat-sheet)

```
Input → [Embedding + Positional Encoding] → N ×
  ┌─ Self-Attention (Q,K,V, softmax(QKᵀ/√dₖ)) ─┐
  └─ Feed-Forward (2-layer, GELU/SiLU)          ┘
Residual + LayerNorm wrap each sub-block

```

- **Self-attention** learns context in one hop (parallelizable vs RNNs).
- Complexity **O(n²)** → long-context tricks: RoPE, Flash-Attention, sliding windows.

### 3. Scaling Laws & Foundation Models

- *Kaplan 2020*: loss ∝ (model params)^-0.076 × (datapoints)^-0.095.
- **Chinchilla (DeepMind 2022)**: keep FLOPs fixed → better to shrink parameters and feed more tokens (e.g., 70 B params + 1.4 T tokens beats 175 B + 300 B).
- **BLOOM (176 B, open)** & **LLaMA (13 B)** show community scaling and efficient training recipes.

### 4. Vector Space Foundations

- Token → d-dim embedding; similarity via cosine.
- Underpins retrieval, RAG, adapters, LoRA low-rank matrices.

**Remember:** Everything downstream (fine-tuning, prompting, RAG) assumes you grasp this lifecycle + transformer core.

---

## Week 2 – Adaptation, Evaluation & Prompting Basics

### 1. Instruction & Multi-Task Fine-Tuning

| Path | When to Use | Typical Data |
| --- | --- | --- |
| **Supervised FT** | New task, plenty labels | 5–100 K examples |
| **Instruction FT (e.g., FLAN-T5)** | Wide task coverage, zero-shot | Mixed tasks + chain-of-thought traces |
| **Domain Adaptive PT** | Niche jargon | Unlabeled in-domain corpus |

> Data rule-of-thumb: 0.1–5 % of pre-training tokens often enough when adapting.
> 

### 2. Parameter-Efficient Fine-Tuning (PEFT)

| Method | Params Trained | Pros | Cons |
| --- | --- | --- | --- |
| **LoRA** | < 1 % (rank r matrices) | GPU friendly, merges offline | Adds latency unless merged |
| **QLoRA** | 4-bit base + LoRA | Fits 33 B on 24 GB VRAM | Slight quality hit from quant |
| **Adapters / Prefix / Prompt Tuning** | 0.1–3 % | Task-speciality libraries | Extra tokens at run-time |

**Mental model:** Freeze backbone; learn *delta* modules → swap in/out like plugins.

### 3. Model Evaluation Playbook

1. **Functional** – exact match, ROUGE/BLEU, word-error-rate.
2. **Holdout Benchmarks** – GLUE/SuperGLUE, MMLU, Big-Bench-Hard.
3. **Holistic & Slice-based** – HELM matrix: capability × scenario × metric × bias.
4. **Human Review** – especially for open-ended outputs.

> Tip: Track win rate vs GPT-4 baseline; it’s now the boardroom KPI.
> 

### 4. Prompt Engineering 101

- **Zero-shot** → “Explain like I’m 5…”
- **Few-shot** → in-context exemplars.
- **System / Persona** layer to steer tone.
- **Output constraints** via XML/JSON snippets for tool integration.

---

## Week 3 – Alignment, Advanced Reasoning & Production

### 1. RLHF & Alignment Stack

| Stage | Data | Loss |
| --- | --- | --- |
| **Supervised Fine-Tuning (SFT)** | Human demos | Cross-entropy |
| **Reward Modeling** | Ranked pairs | Pair-wise Bradley-Terry |
| **Policy Optimization (PPO / DPO)** | LM outputs + reward | RL objective or KL-regularized CE |
- **DPO** simplifies: directly maximize P(preferred|θ).
- **Constitutional AI** replaces humans with authored principles → bootstrap harmlessness.

### 2. Advanced Prompting → Reasoning + Tools

| Technique | Core Idea | When to Reach For It |
| --- | --- | --- |
| **CoT** | Insert “Let’s think step by step.” | Math, logic, multi-hop QA |
| **PAL** | LLM writes runnable code → execute | Precise arithmetic / data lookups |
| **ReAct** | Interleave “Thought, Action, Obs” | Tool-using agents |
| **RAG** | Inject retrieved docs via context | Up-to-date knowledge, domain expertise |

> Design Note: RAG is often cheaper and safer than full fine-tune when knowledge changes rapidly.
> 

### 3. Deployment Patterns

- **Hosted API (OpenAI, Anthropic)** – fastest to market; pay per token.
- **Self-Hosted GPU** – control + data sovereignty; watch *token TCO* (cap-ex vs op-ex).
- **Serverless Containers (SageMaker, Vertex)** – autoscale; good for spiky workloads.
- **On-Device / Edge** – 7 B-13 B quantized, metal acceleration (llama.cpp, GGUF).

**Ops Must-Dos:** batching, streaming tokens, KV cache reuse, safety middleware, canary rollout + offline eval alignment.

### 4. LLM-Powered App Stack (LangChain-style)

```
UI / API
   ↓
Controller / Agent  ←— Memory, State
   ↓
Chains (Prompt ↔ Model ↔ Tool)
   ↓
Retrievers + Vector DB (FAISS, Pinecone, PGVector)
   ↓
Foundation Model Endpoint

```

- Glue logic lives in **chains + agents**; keep prompts version-controlled.
- Log every input/ output pair for eval & regression testing.

### 5. Responsible AI Checklist

1. **Fairness & Bias** – dataset audits, demographic parity tests.
2. **Toxicity & Safety** – block-lists, refusal patterns, RLHF or SAFETY-fine-tune.
3. **Privacy** – PII scrubbing; differential privacy where needed.
4. **Transparency** – model cards, data cards, eval reports.
5. **Governance** – human-in-the-loop overrides, incident playbooks.

---

### Quick Reference Cheatsheet

| Concept | One-Liner |
| --- | --- |
| **Attention Score** | softmax(QKᵀ / √dₖ) V |
| **LoRA Update** | W ← W + Δ; Δ = BA (rank r) |
| **RAG Memory Budget** | ≤ (8 K – task-prompt tokens) / 2 *d_response |
| **Chinchilla Optimum** | N_tokens ≈ 20 × N_params |
| **PPO Clip** | r_t(θ) = π_θ(a |

---

### How to Use These Notes

- **Before coding:** skim the lifecycle & architecture tables to choose the right approach.
- **During builds:** refer to PEFT and deployment matrices for cost/time trade-offs.
- **For interviews & talks:** lean on the cheatsheet formulas and alignment stages.
- **Ongoing:** append personal experiments (hyper-params, latency numbers) under each heading.

*Happy building—these fundamentals will keep paying dividends as the GenAI landscape evolves.*