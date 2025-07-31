# LLM

[简体中文](README.md)

# 🌟 LLM Study Journey

> 👤 Author: [@shangguanjiannan](https://github.com/shangguanjiannan)  
> 📅 Started: 2025  
> 🛠️ Last Updated: 2025-07-31  
> 🧷 Keywords: LLM, Transformer, Fine-tuning, RAG, Inference, Tokenizer

## 🧠 Background

With the boom of Large Language Models (LLMs), understanding their mechanisms and implementation has become essential. This document records my insights, hands-on projects, and troubleshooting during the learning journey.

## 🗂️ Table of Contents

1. Fundamentals  
2. Training  
3. Inference & Deployment  
4. RAG (Retrieval-Augmented Generation)  
5. Projects  
6. Troubleshooting Notes  
7. References  

## 📘 Fundamentals

### 🔸 Transformer Architecture

- Original paper: *Attention Is All You Need*  
- Key modules: Multi-head attention, Feedforward layers, Residual + LayerNorm  
- Model types:  
  - Decoder-only (e.g., GPT)  
  - Encoder-only (e.g., BERT)  
  - Encoder-Decoder (e.g., T5/BART)

### 🔸 Tokenization

- Methods: BPE, Unigram, WordPiece, SentencePiece  
- Tools: Huggingface Tokenizers, SentencePiece, Tiktoken  
- For Chinese: `bge`, `text2vec`, `QwenTokenizer`

## 🧪 Training

### 🔹 Pretraining

- Preprocessing: deduplication, cleaning, formatting  
- Tools: `transformers`, `OLMo`, `DeepSpeed`, `Axolotl`  
- Config: `config.json`, `tokenizer_config.json`

### 🔹 Supervised Fine-tuning (SFT)

```json
{"instruction": "Who are you?", "input": "", "output": "I'm an AI assistant."}
```

- Format styles: Alpaca, ChatML, OpenAI  
- Tips: Use multiple templates and scripting to generate training data

## ⚙️ Inference & Deployment

### 🔹 Serving with vLLM

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server   --served-model-name qwen3-14b   --model /path/to/qwen3-14b   --tensor-parallel-size 2   --max-model-len 32000   --port 8051
```

- Memory tuning:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export VLLM_DISABLE_TORCH_COMPILE=1
  export TORCHDYNAMO_DISABLE=1
  ```

### 🔹 Convert OLMo to HF Format

- Path: `/mnt/.../Pretrain/CNIT-1B-TDDRSIMPLE`  
- Output: sharded `.safetensors`, `config.json`, and custom `tokenizer.json`

### 🔹 Tokenizer Conflict Fix

- Problem: `AutoConfig.register("aimv2", AIMv2Config)` causes duplicate entry  
- Fix: comment out the conflicting line

## 🔍 RAG System

- Embedding Models: `all-MiniLM-L6-v2`, `bge`, `text2vec`  
- Milvus Workflow: schema → insert → index → search  

```python
from pymilvus import Collection
collection = Collection("mcp-errors")
collection.search(...)
```

## 🛠️ Projects

### MCP-RAG

- Goal: AI assistant for error log retrieval and reasoning  
- Stack: Milvus + SentenceTransformer + FastAPI + Instruction Tuning

## 🐛 Troubleshooting Notes

| Issue | Solution |
|-------|----------|
| np.float deprecated | Use np.float64 |
| PY_SSIZE_T_CLEAN error | Add macro and recompile |
| AIMv2 register conflict | Comment out register line |
| torch.compile crash | Disable torch.compile for vLLM |

## 📚 References

- [Transformers Docs](https://huggingface.co/docs/transformers)  
- [OLMo](https://github.com/allenai/OLMo)  
- [vLLM](https://github.com/vllm-project/vllm)  
- [Milvus](https://milvus.io/)  
- [Qwen](https://github.com/QwenLM)