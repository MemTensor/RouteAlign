"""用 router_21K_Qwen3-0.6B 自身提取 query 的"需求空间"嵌入。

取与任务画像训练数据逐字一致的 sft_v3 提示词（system+user），
用 chat template 渲染后，取最后一个 prompt token 的末层 hidden state
作为该 query 的需求表征（模型开始生成画像 JSON 前的内部分析状态）。

输出 L2 归一化嵌入缓存 emb_router_{train,test}.npy。

用法：
    CUDA_VISIBLE_DEVICES=1 python -m adapter.compute_router_embeddings
"""

import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import (
    TASK_PROFILE_SFT_V3_SYSTEM_PROMPT,
    TASK_PROFILE_SFT_V3_USER_TEMPLATE,
)

MODEL_PATH = os.environ.get("PROFILER_MODEL_PATH", "/mnt/afs/chending_zjh/models/router_21K_Qwen3-0.6B")
DATA = os.environ.get("ROUTER_DATA_DIR", "data/seed42_split0.7")
BATCH = 64
MAX_LEN = 4096


def load_queries(path):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line)["query"])
    return qs


@torch.no_grad()
def embed(model, tok, queries, device):
    outs = []
    for i in range(0, len(queries), BATCH):
        batch = queries[i:i + BATCH]
        texts = []
        for q in batch:
            q = q[:20000]
            messages = [
                {"role": "system", "content": TASK_PROFILE_SFT_V3_SYSTEM_PROMPT},
                {"role": "user", "content": TASK_PROFILE_SFT_V3_USER_TEMPLATE.format(query=q)},
            ]
            texts.append(
                tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        enc = tok(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_LEN, padding_side="left",
        ).to(device)
        # 只过 backbone，跳过 lm_head（避免 B×T×vocab 的巨大 logits）
        out = model.model(**enc)
        hidden = out.last_hidden_state  # (B, T, H)
        lengths = enc["attention_mask"].sum(dim=1)  # (B,)
        last = hidden[torch.arange(len(batch)), lengths - 1]  # 左填充后最后一个非 pad token
        outs.append(last.float().cpu().numpy())
        if (i // BATCH) % 20 == 0:
            print(f"  {i}/{len(queries)}", flush=True)
    arr = np.concatenate(outs, axis=0)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def main():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(device).eval()

    for split in ("train", "test"):
        queries = load_queries(f"{DATA}/{split}.jsonl")
        print(f"{split}: {len(queries)} queries", flush=True)
        arr = embed(model, tok, queries, device)
        np.save(f"{DATA}/emb_router_{split}.npy", arr)
        print(f"saved emb_router_{split}.npy {arr.shape}", flush=True)


if __name__ == "__main__":
    main()
