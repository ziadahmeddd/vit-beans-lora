# Assignment 14 — ViT + LoRA + INT8 Quantization + FastAPI + Docker

Bean-leaf disease classifier (`AI-Lab-Makerere/beans`: angular_leaf_spot, bean_rust, healthy)
built on `google/vit-base-patch16-224-in21k`.

```
kaggle/train.ipynb          LoRA fine-tuning (runs on Kaggle T4 GPU)
kaggle/kernel-metadata.json Kaggle kernel config (GPU + internet)
quantize.py                 FP32 -> INT8 dynamic quantization + benchmark (local, CPU)
app/quant.py                quantize / load INT8 model helpers
app/main.py                 FastAPI service (/health, /predict)
Dockerfile, requirements.txt
models/vit-beans-lora/      best LoRA adapter (2.3 MB)
models/vit-beans-merged/    adapter merged into base, FP32 (Kaggle output)
models/vit-beans-int8/      quantized model served by the API
samples/                    one test image per class
```

## 1. Fine-tune with LoRA (Kaggle)
LoRA `r=16, alpha=16, dropout=0.1` on attention `query`/`value`, classifier head fully trained
→ **592K trainable params (0.69 %)** of 86.4M. AdamW lr 5e-3, batch 32, fp16, up to 10 epochs
with early stopping; `load_best_model_at_end` on validation accuracy keeps the best checkpoint.

```bash
kaggle kernels push -p kaggle
kaggle kernels output <user>/vit-beans-lora -p models
```

| split | accuracy |
|---|---|
| validation | 0.9925 |
| test | 0.9375 |

## 2. Save best model
`models/vit-beans-lora` = best LoRA adapter; `models/vit-beans-merged` = adapter merged
(`merge_and_unload`) into the base weights for deployment.

## 3. Quantization
PyTorch dynamic INT8 quantization of all `nn.Linear` layers (`python quantize.py`), CPU, test set:

| model | size | test acc | latency / image |
|---|---|---|---|
| FP32 | 327.4 MB | 0.9375 | 115.2 ms |
| INT8 | **84.4 MB** | **0.9531** | **50.8 ms** |

3.88× smaller, 2.27× faster, 96.9 % prediction agreement with FP32.

## 4–6. FastAPI + Docker + inference
```bash
docker build -t vit-beans-api .
docker run -d --name vit-beans -p 8000:8000 vit-beans-api
curl localhost:8000/health
curl -F "file=@samples/bean_rust.jpg" localhost:8000/predict
```
```json
{"label":"bean_rust","confidence":0.9979,"probabilities":{"angular_leaf_spot":0.0017,"bean_rust":0.9979,"healthy":0.0004}}
```
Non-image uploads return `400`. Container RAM at idle after loading: ~525 MiB. Interactive docs: http://localhost:8000/docs

## Local setup
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt datasets kaggle --extra-index-url https://download.pytorch.org/whl/cpu
```
