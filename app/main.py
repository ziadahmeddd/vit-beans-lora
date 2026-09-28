import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor

from quant import load_int8

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/vit-beans-int8"))
torch.set_num_threads(os.cpu_count() or 1)


app = FastAPI(title="ViT Beans Classifier (LoRA + INT8)")
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = load_int8(MODEL_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "labels": list(model.config.id2label.values())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "file is not a valid image")
    with torch.inference_mode():
        probs = model(**processor(image, return_tensors="pt")).logits.softmax(-1)[0]
    top = int(probs.argmax())
    return {
        "label": model.config.id2label[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {model.config.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }
