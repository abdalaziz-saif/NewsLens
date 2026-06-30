# NewsLens

**Structured News Understanding via LLM Knowledge Distillation and LoRA Fine-Tuning**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge&logoColor=black" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic"/>
  <img src="https://img.shields.io/badge/LLaMA--Factory-00A67E?style=for-the-badge" alt="LLaMA-Factory"/>
  <img src="https://img.shields.io/badge/LoRA-8A2BE2?style=for-the-badge" alt="LoRA"/>
  <img src="https://img.shields.io/badge/Qwen2.5-FF6F00?style=for-the-badge" alt="Qwen2.5"/>
  <img src="https://img.shields.io/badge/Weights%20%26%20Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black" alt="W&B"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
</p>

---

## Overview

**NewsLens** is an end-to-end pipeline that distills structured information-extraction capability from a large teacher LLM into a small, efficient student model that can be fine-tuned and deployed at low cost.

Given a raw news article, the fine-tuned model extracts a schema-validated JSON object containing:

- **Title** — concise, informative headline
- **Category** — politics, sports, technology, economy, health, entertainment, science, etc.
- **Summary** — key points (1–5 bullet points)
- **Entities** — people, organizations, locations, events, etc., with type labels
- **Keywords** — relevant terms for indexing/search

---

## Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Schema Design   │ ──> │ Knowledge         │ ──> │ Dataset Formatting │ ──> │ LoRA Fine-Tuning  │
│  (Pydantic)      │     │ Distillation      │     │ (LLaMA-Factory)    │     │ (Qwen2.5-1.5B)    │
│                  │     │ (Qwen2.5-3B)      │     │                    │     │                  │
└─────────────────┘     └──────────────────┘     └───────────────────┘     └──────────────────┘
```

| Stage | Description |
|---|---|
| **1. Schema Design** | Output structure defined and enforced with Pydantic models |
| **2. Knowledge Distillation** | `Qwen2.5-3B-Instruct` generates structured annotations over the CNN/DailyMail dataset via batched, schema-constrained prompting, with crash-safe checkpointing for long-running Colab sessions |
| **3. Dataset Formatting** | Distilled outputs converted into LLaMA-Factory's instruction-tuning format |
| **4. LoRA Fine-Tuning** | `Qwen2.5-1.5B-Instruct` fine-tuned via LoRA (rank 32, full-module targeting) under supervised fine-tuning, tracked in Weights & Biases |
| **5. Inference** | Trained adapter loaded onto the base student model for structured extraction on new articles |

---

## Why Distillation + LoRA?

Running a 3B+ parameter model for every inference call is expensive and slow. NewsLens uses the large model **once**, offline, purely as a labeling engine — then transfers that capability into a 1.5B model that's cheap to serve and fast to run, via parameter-efficient fine-tuning (LoRA) rather than full fine-tuning.


---

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | Qwen2.5 (3B teacher / 1.5B student), Transformers, PyTorch |
| Fine-Tuning | LoRA, LLaMA-Factory |
| Data Validation | Pydantic |
| Experiment Tracking | Weights & Biases |
| Dataset | CNN/DailyMail (Hugging Face Datasets) |
| Environment | Google Colab |

---

## Status

🚧 Work in progress 

---

## License

MIT
