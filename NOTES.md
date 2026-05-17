#!/usr/bin/env bash
# =====================================================================
# run.sh — vast.ai 시나리오별 실행 커맨드 (복붙용)
# 가정: PyTorch 인스턴스, 작업 디렉토리는 /workspace
# =====================================================================

# 0. 1회 셋업
# ---------------------------------------------------------------------
# pip install -U wandb transformers datasets tiktoken accelerate einops numpy
# wandb login    # 토큰 직접 입력 (수동)
# # 학습 코퍼스를 /workspace 에 배치

# A. 단일 GPU 학습 (sweep 없이)
# ---------------------------------------------------------------------
# python train.py --project my-llm-project --run_name "single-gpu-baseline"

# B. 멀티 GPU DDP 학습 (sweep 없이)
# ---------------------------------------------------------------------
# torchrun --standalone --nproc_per_node=2 train.py \
#     --project my-llm-project --run_name "final-best" \
#     --epochs 5 \
#     --lr 0.0004233275962272715 --dropout 0.2 --batch_size 4 --grad_accum 1 \
#     --depth 12 --dim 512 --warmup_steps 100

# C. sweep 등록 + agent 실행 (DDP 포함)
# ---------------------------------------------------------------------
# DDP 설정은 sweep.yaml 의 nproc_per_node 값으로 결정됨 (현재 2)
#
# wandb sweep --project my-llm-project sweep.yaml
# # → 출력: wandb agent myname/my-llm-project/abc123xy
#
# wandb agent choijiwan1229-hansung-science-high-school/my-llm-project/u7z29q2i --count 10
#
# # 다른 GPU 그룹으로 병렬 agent 추가 (각 agent마다 자기 GPU만 보이게):
# # CUDA_VISIBLE_DEVICES=2,3 nohup wandb agent myname/my-llm-project/abc123xy > agent2.log 2>&1 &

##############################
#AWS 실행

