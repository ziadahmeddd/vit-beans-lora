from pathlib import Path

import torch
from transformers import ViTConfig, ViTForImageClassification


def quantize(model):
    """Dynamic INT8 quantization of every nn.Linear (weights int8, activations quantized on the fly)."""
    return torch.ao.quantization.quantize_dynamic(model.eval(), {torch.nn.Linear}, dtype=torch.qint8)


def load_int8(model_dir):
    # Rebuild the architecture, quantize the empty shell, then load the saved int8 weights into it.
    model = quantize(ViTForImageClassification(ViTConfig.from_pretrained(model_dir)))
    model.load_state_dict(torch.load(Path(model_dir) / "model_int8.pt", weights_only=False))
    return model.eval()
