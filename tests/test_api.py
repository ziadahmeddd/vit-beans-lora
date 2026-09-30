import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODEL_DIR", str(ROOT / "models/vit-beans-int8"))
sys.path.insert(0, str(ROOT / "app"))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)
LABELS = ["angular_leaf_spot", "bean_rust", "healthy"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "labels": LABELS}


@pytest.mark.parametrize("label", LABELS)
def test_predict_sample(label):
    with open(ROOT / f"samples/{label}.jpg", "rb") as f:
        r = client.post("/predict", files={"file": (f"{label}.jpg", f, "image/jpeg")})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] == label
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_predict_rejects_non_image():
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
