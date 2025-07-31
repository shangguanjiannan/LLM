# LLM

[English](README_en.md)

# 🌟 大模型学习历程

> 👤 作者: [@shangguanjiannan](https://github.com/shangguanjiannan)  
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