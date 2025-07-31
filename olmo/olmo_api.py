from fastapi import FastAPI, Request
import torch
import yaml
from transformers import AutoTokenizer
from olmo.model import OLMo  # 按你的OLMo源码实际路径导入
from omegaconf import OmegaConf
from pydantic import BaseModel

app = FastAPI()

# 路径配置
MODEL_DIR = "xxx/Pretrain/1B-TDDRSIMPLE/step55682-unsharded"
CONFIG_PATH = f"{MODEL_DIR}/config.yaml"
MODEL_PATH = f"{MODEL_DIR}/model.pt"

# 1. 加载配置
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
model_args = config["model"]
# 如果没有 effective_n_kv_heads，就补充
if "effective_n_kv_heads" not in model_args:
    model_args["effective_n_kv_heads"] = model_args.get("n_heads", 16)
model_args = OmegaConf.create(model_args)  # 关键修正

# 2. 加载分词器（HuggingFace格式）
tokenizer_path = 'xxx/qwen2.5-7B_tokenizer'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# 3. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OLMo(model_args)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 简单贪婪解码生成函数
def generate(model, input_ids, max_new_tokens=128, eos_token_id=0):
    model.eval()
    device = next(model.parameters()).device
    generated = input_ids.to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            generated = generated.to(device)
            logits = model(generated)[0][:, -1, :]  # 关键修正
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break
    return generated


class ChatRequest(BaseModel):
    input: str

@app.post("/chat")
async def chat(req: ChatRequest):
    user_input = req.input

    # 文本转token
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    # 推理生成
    output_ids = generate(model, input_ids, max_new_tokens=128, eos_token_id=model_args.get("eos_token_id", 0))
    response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    return {"response": response}