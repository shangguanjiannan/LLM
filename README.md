
<!-- 中英切换按钮样式与脚本 -->
<style>
.toggle-button {
  margin: 1rem 0;
  padding: 6px 12px;
  background-color: #3b82f6;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
.lang-cn { display: block; }
.lang-en { display: none; }
</style>

<button class="toggle-button" onclick="toggleLang()">🌐 切换语言 / Switch Language</button>

<script>
let showCN = true;
function toggleLang() {
  showCN = !showCN;
  document.querySelectorAll('.lang-cn').forEach(el => el.style.display = showCN ? 'block' : 'none');
  document.querySelectorAll('.lang-en').forEach(el => el.style.display = showCN ? 'none' : 'block');
}
</script>

---

<div class="lang-cn">

# 🌟 大模型学习历程

> 👤 作者: [@shangguanjiannan](#)  
> 📅 起始时间: 2025  
> 🛠️ 最近更新: 2025-07-31  
> 🧷 关键词: LLM, Transformer, 微调, 检索增强生成, 推理部署, Tokenizer  

## 🧠 学习背景

随着大语言模型（LLM）的爆发，理解其原理并掌握落地方法成为 AI 工程师的核心能力。记录了我在 LLM 学习与实战过程中的思考与技术积累。

## 🗂️ 学习目录

1. 基础理论  
2. 模型训练  
3. 推理部署  
4. RAG 检索增强生成  
5. 实战项目  
6. 踩坑笔记  
7. 参考资源  

## 📘 基础理论

### 🔸 Transformer 架构

- 原始论文: *Attention Is All You Need*  
- 核心模块: 多头注意力、前馈网络、残差连接+归一化  
- 模型类型:  
  - Decoder-only（如 GPT）  
  - Encoder-only（如 BERT）  
  - Encoder-Decoder（如 T5/BART）

### 🔸 分词机制

- 常用方法: BPE、Unigram、SentencePiece  
- 工具推荐: Huggingface Tokenizers、SentencePiece、Tiktoken  
- 中文支持: `bge`, `text2vec`, `QwenTokenizer`

## 🧪 模型训练

### 🔹 预训练

- 数据处理：去重、清洗、格式标准化  
- 使用工具：`transformers`、`OLMo`、`DeepSpeed`、`Axolotl`  
- 配置管理：`config.json`, `tokenizer_config.json`

### 🔹 指令微调 SFT

```json
{"instruction": "你是谁", "input": "", "output": "我是AI助手"}
```

- 格式支持：OpenAI、Alpaca、ChatML  
- 推荐写法：多模板支持 + 批量生成脚本  

## ⚙️ 推理部署

### 🔹 使用 vLLM 部署

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server   --served-model-name qwen3-14b   --model /path/to/qwen3-14b   --tensor-parallel-size 2   --max-model-len 32000   --port 8051
```

- 显存优化：  
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export VLLM_DISABLE_TORCH_COMPILE=1
  export TORCHDYNAMO_DISABLE=1
  ```

- Torch >= 2.3 时需关闭 torch.compile，防止 kernel 报错  

### 🔹 模型格式转换（OLMo → HF）

- 示例路径: `/mnt/.../Pretrain/CNIT-1B-TDDRSIMPLE`  
- 转换输出：分片 `.safetensors` + `config.json` + 自定义 `tokenizer.json`

### 🔹 Tokenizer 冲突修复

- 报错位置：`AutoConfig.register("aimv2", AIMv2Config)`  
- 修复方式：注释该行代码

## 🔍 RAG 检索增强生成

- 特征模型推荐：`all-MiniLM-L6-v2`, `bge`, `text2vec`  
- Milvus 检索系统使用步骤：schema → 插入 → 索引 → 查询  

```python
from pymilvus import Collection
collection = Collection("mcp-errors")
collection.search(...)
```

## 🛠️ 实战项目

### MCP-RAG

- 核心目标：构建错误日志知识库 + 自动问答  
- 关键组件：向量索引、语义检索、指令微调回复

## 🐛 踩坑笔记

| 问题 | 解决方法 |
|------|----------|
| np.float 被废弃 | 改为 np.float64 |
| PY_SSIZE_T_CLEAN 报错 | 增加宏定义重新编译 |
| AIMv2 注册冲突 | 注释掉注册项 |
| torch.compile 报错 | 关闭 vLLM 中的 compile |

## 📚 参考资源

- [Transformers 文档](https://huggingface.co/docs/transformers)
- [OLMo 项目](https://github.com/allenai/OLMo)
- [vLLM 项目](https://github.com/vllm-project/vllm)
- [Milvus 数据库](https://milvus.io/)
- [Qwen 项目](https://github.com/QwenLM)

</div>

<div class="lang-en">

# 🌟 LLM Study Journey

> 👤 Author: [@shangguanjiannan](#)  
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

</div>
