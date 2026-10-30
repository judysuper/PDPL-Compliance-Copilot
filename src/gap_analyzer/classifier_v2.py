from __future__ import annotations
from pathlib import Path
import json
import numpy as np


class XLMRClassifierAdapter:
    """
    Loads the local fine-tuned XLM-R model produced by E4.
    If the model cannot be loaded, pass classifier=None to the analyzer
    and the Gap Analyzer will still run with the PDPL rule layer.
    """

    def __init__(self, model_dir: str | Path, thresholds_path: str | Path | None = None, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.id2label = {}
        if getattr(cfg, "id2label", None):
            self.id2label = {int(k): str(v) for k, v in cfg.id2label.items()}

        # Some training pipelines save generic LABEL_0 names. Try local label maps.
        if not self.id2label or all(v.startswith("LABEL_") for v in self.id2label.values()):
            for name in ("label_mapping.json", "label2id.json", "labels.json"):
                p = self.model_dir / name
                if not p.exists():
                    continue
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    # label -> int
                    if all(isinstance(v, int) for v in obj.values()):
                        self.id2label = {int(v): str(k) for k, v in obj.items()}
                    # int-like -> label
                    elif all(str(k).isdigit() for k in obj.keys()):
                        self.id2label = {int(k): str(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    self.id2label = {i: str(v) for i, v in enumerate(obj)}
                if self.id2label:
                    break

        n = int(getattr(cfg, "num_labels", len(self.id2label) or 10))
        if not self.id2label:
            self.id2label = {i: f"LABEL_{i}" for i in range(n)}

        self.thresholds = {label: 0.5 for label in self.id2label.values()}
        if thresholds_path:
            p = Path(thresholds_path)
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            self.thresholds[str(k)] = float(v)
                        except Exception:
                            pass

    def predict_all_scores(self, text: str) -> dict[str, float]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0]
            probs = self.torch.sigmoid(logits).detach().cpu().numpy()

        return {
            self.id2label.get(i, f"LABEL_{i}"): float(probs[i])
            for i in range(len(probs))
        }
