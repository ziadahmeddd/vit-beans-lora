"""Quantize the merged LoRA ViT (downloaded from Kaggle) to INT8 and compare it with FP32.

Usage: python quantize.py   (expects models/vit-beans-merged from the Kaggle notebook output)
"""
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

sys.path.insert(0, "app")
from quant import load_int8, quantize  # noqa: E402

SRC, DST = "models/vit-beans-merged", "models/vit-beans-int8"

processor = AutoImageProcessor.from_pretrained(SRC)
fp32 = ViTForImageClassification.from_pretrained(SRC).eval()
int8 = quantize(ViTForImageClassification.from_pretrained(SRC))

os.makedirs(DST, exist_ok=True)
torch.save(int8.state_dict(), f"{DST}/model_int8.pt")
int8.config.save_pretrained(DST)
processor.save_pretrained(DST)
torch.save(fp32.state_dict(), "fp32.tmp.pt")
fp32_mb, int8_mb = os.path.getsize("fp32.tmp.pt") / 2**20, os.path.getsize(f"{DST}/model_int8.pt") / 2**20
os.remove("fp32.tmp.pt")

# Reload from disk exactly like the API does, so the benchmark covers the deployed artifact.
int8 = load_int8(DST)

test = load_dataset("AI-Lab-Makerere/beans", split="test")
images = [im.convert("RGB") for im in test["image"]]
labels = torch.tensor(test["labels"])


@torch.inference_mode()
def bench(model):
    preds = torch.cat([model(**processor(images[i:i + 32], return_tensors="pt")).logits.argmax(-1)
                       for i in range(0, len(images), 32)])
    x = processor(images[0], return_tensors="pt")
    model(**x)  # warm-up
    t = time.perf_counter()
    for _ in range(20):
        model(**x)
    return preds, (preds == labels).float().mean().item(), (time.perf_counter() - t) / 20 * 1000


p32, acc32, ms32 = bench(fp32)
p8, acc8, ms8 = bench(int8)
print(f"{'model':<6}{'size MB':>10}{'test acc':>10}{'ms/img':>10}")
print(f"{'fp32':<6}{fp32_mb:>10.1f}{acc32:>10.4f}{ms32:>10.1f}")
print(f"{'int8':<6}{int8_mb:>10.1f}{acc8:>10.4f}{ms8:>10.1f}")
print(f"size reduction {fp32_mb / int8_mb:.2f}x, speedup {ms32 / ms8:.2f}x, "
      f"prediction agreement {(p32 == p8).float().mean():.4f}")
assert acc8 >= acc32 - 0.03, "INT8 lost more than 3 points of accuracy"
