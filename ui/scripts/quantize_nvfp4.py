#!/usr/bin/env python3
"""Quantize an HF model to NVFP4 with NVIDIA TensorRT Model Optimizer.

Runs INSIDE the anima-vllm:thor-latest container (torch 2.11 / CUDA 13, sm_110a).
Prints `[STAGE] <name> <msg>` markers that anima-thor-ui parses for the progress UI.

Usage:
  python quantize_nvfp4.py --model <hf_repo> --out <dir> [--calib 64]
"""
from __future__ import annotations

import argparse
import sys


def log_stage(name: str, msg: str = ""):
    print(f"[STAGE] {name} {msg}", flush=True)


# A small, diverse, offline calibration set — enough for PTQ activation scales.
CALIB_PROMPTS = [
    "Explain how a transformer attention head works, step by step.",
    "Write a Python function that merges two sorted lists.",
    "Summarize the causes of the 2008 financial crisis.",
    "Translate to French: 'The weather is beautiful today.'",
    "What are the trade-offs between TCP and UDP?",
    "Give me a recipe for a simple tomato pasta.",
    "Describe the plot of Romeo and Juliet in three sentences.",
    "How does photosynthesis convert light into chemical energy?",
    "Write a SQL query to find the top 5 customers by total spend.",
    "Explain the difference between supervised and unsupervised learning.",
    "What is the capital of Japan and why is it significant?",
    "Refactor this loop to be more efficient: for i in range(len(a)): b.append(a[i]*2)",
    "Outline the steps to deploy a Docker container.",
    "Compare REST and GraphQL APIs.",
    "Explain quantum entanglement to a high-school student.",
    "Draft a polite email declining a meeting invitation.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=int, default=64)
    args = ap.parse_args()

    log_stage("load", "importing torch + modelopt")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    # pick an NVFP4 config robustly across modelopt versions
    cfg = None
    for name in ("NVFP4_DEFAULT_CFG", "NVFP4_FP8_KV_CFG", "NVFP4_AWQ_LITE_CFG"):
        cfg = getattr(mtq, name, None)
        if cfg is not None:
            print(f"[INFO] using mtq.{name}", flush=True)
            break
    if cfg is None:
        print("[ERROR] no NVFP4 config found in this modelopt build", flush=True)
        return 3

    log_stage("load", f"loading {args.model} in bf16")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    prompts = (CALIB_PROMPTS * ((args.calib // len(CALIB_PROMPTS)) + 1))[: args.calib]

    def forward_loop(m):
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").to(m.device)
            with torch.no_grad():
                m(**ids)
            if i % 8 == 0:
                print(f"[STAGE] calibrate sample {i+1}/{len(prompts)}", flush=True)

    log_stage("calibrate", f"calibrating on {len(prompts)} samples")
    log_stage("quantize", "applying NVFP4 PTQ")
    mtq.quantize(model, cfg, forward_loop)

    log_stage("export", f"writing NVFP4 checkpoint to {args.out}")
    import os
    os.makedirs(args.out, exist_ok=True)
    export_hf_checkpoint(model, export_dir=args.out)
    tok.save_pretrained(args.out)

    log_stage("done", "export complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
