"""
chat.py — 학습된 모델로 텍스트 이어쓰기

사용:
    python chat.py "your prompt"

주의:
    아래 CFG는 학습 시 사용한 하이퍼파라미터와 정확히 일치해야 합니다.
    학습 명령을 바꾸셨다면 CFG도 같이 수정하세요.
"""

import os
import sys
import torch
import tiktoken
from train import LLM


MODEL_DIR = "/workspace/custom-llm-out"

# 학습 시와 동일하게 — 다른 sweep 결과로 학습했다면 여기도 같이 수정
CFG = dict(
    dim=768,
    depth=12,
    heads=8,
    dim_head=64,
    mlp_dim=2048,
    block_size=512,
    rope_base=10000,
)


def load_model(device):
    enc = tiktoken.get_encoding("r50k_base")

    model = LLM(
        dim=CFG["dim"], depth=CFG["depth"], max_len=CFG["block_size"],
        mlp_dim=CFG["mlp_dim"], heads=CFG["heads"], dim_head=CFG["dim_head"],
        vocab_size=enc.n_vocab, padding_idx=enc.eot_token,
        base=CFG["rope_base"], dropout=0.0,
    )

    sf = os.path.join(MODEL_DIR, "model.safetensors")
    pt = os.path.join(MODEL_DIR, "pytorch_model.bin")
    if os.path.exists(sf):
        from safetensors.torch import load_file
        sd = load_file(sf)
    elif os.path.exists(pt):
        sd = torch.load(pt, map_location="cpu")
    else:
        sys.exit(f"가중치 파일을 찾을 수 없습니다: {MODEL_DIR}")

    model.load_state_dict(sd)
    return model.to(device).eval(), enc


@torch.no_grad()
def generate(model, enc, prompt, max_new=200, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = ids.size(1)

    for _ in range(max_new):
        ctx = ids[:, -CFG["block_size"]:]
        logits = model(ctx).logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == enc.eot_token:
            break

    return enc.decode(ids[0, prompt_len:].tolist())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("사용법: python chat.py \"your prompt\"")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, enc = load_model(device)
    print(generate(model, enc, sys.argv[1]))