from __future__ import annotations

from pathlib import Path
import json


DEFAULT_MODEL = "ItsJudy/pdpl-opp115-xlm-roberta"


class XLMRClassifierAdapter:
    """
    Adapter for the fine-tuned XLM-RoBERTa classifier used in Experiment E4.

    The model can be loaded from either:

    1. A local model directory, for example:
       models/xlm_roberta_best

    2. A Hugging Face model repository, for example:
       ItsJudy/pdpl-opp115-xlm-roberta

    By default, the classifier is downloaded automatically from Hugging Face:

        ItsJudy/pdpl-opp115-xlm-roberta

    If the classifier is unavailable, the Gap Analyzer can still be run
    with classifier=None and use the PDPL rule-based layer only.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL,
        thresholds_path: str | Path | None = None,
        device: str | None = None,
    ):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch = torch
        self.model_source = str(model_dir)

        # ---------------------------------------------------------
        # Determine whether model_dir is:
        # 1. a local path, or
        # 2. a Hugging Face repository ID
        # ---------------------------------------------------------

        local_path = Path(self.model_source)

        if local_path.exists():
            model_source = str(local_path)
            self.model_dir = local_path
            self.is_local_model = True
        else:
            model_source = self.model_source
            self.model_dir = None
            self.is_local_model = False

        # ---------------------------------------------------------
        # Load tokenizer and model
        #
        # If model_source is a Hugging Face repo ID,
        # transformers downloads the model automatically.
        # ---------------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source
        )

        # ---------------------------------------------------------
        # Select device
        # ---------------------------------------------------------

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------
        # Load label mapping from model config
        # ---------------------------------------------------------

        cfg = self.model.config

        self.id2label: dict[int, str] = {}

        if getattr(cfg, "id2label", None):
            self.id2label = {
                int(k): str(v)
                for k, v in cfg.id2label.items()
            }

        # ---------------------------------------------------------
        # Some training pipelines save generic labels such as:
        #
        # LABEL_0
        # LABEL_1
        # ...
        #
        # If that happens and the model is local, try loading
        # the real label names from local JSON files.
        # ---------------------------------------------------------

        if (
            not self.id2label
            or all(
                str(label).startswith("LABEL_")
                for label in self.id2label.values()
            )
        ):
            if self.model_dir is not None:

                possible_label_files = (
                    "label_mapping.json",
                    "label2id.json",
                    "labels.json",
                )

                for filename in possible_label_files:

                    path = self.model_dir / filename

                    if not path.exists():
                        continue

                    obj = json.loads(
                        path.read_text(encoding="utf-8")
                    )

                    # -------------------------------------------------
                    # Case 1:
                    #
                    # {
                    #   "First Party Collection": 0,
                    #   "Third Party Sharing": 1
                    # }
                    # -------------------------------------------------

                    if isinstance(obj, dict):

                        if all(
                            isinstance(v, int)
                            for v in obj.values()
                        ):
                            self.id2label = {
                                int(v): str(k)
                                for k, v in obj.items()
                            }

                        # ---------------------------------------------
                        # Case 2:
                        #
                        # {
                        #   "0": "First Party Collection",
                        #   "1": "Third Party Sharing"
                        # }
                        # ---------------------------------------------

                        elif all(
                            str(k).isdigit()
                            for k in obj.keys()
                        ):
                            self.id2label = {
                                int(k): str(v)
                                for k, v in obj.items()
                            }

                    # -------------------------------------------------
                    # Case 3:
                    #
                    # [
                    #   "First Party Collection",
                    #   "Third Party Sharing"
                    # ]
                    # -------------------------------------------------

                    elif isinstance(obj, list):
                        self.id2label = {
                            i: str(label)
                            for i, label in enumerate(obj)
                        }

                    if self.id2label:
                        break

        # ---------------------------------------------------------
        # Number of labels
        # ---------------------------------------------------------

        num_labels = int(
            getattr(
                cfg,
                "num_labels",
                len(self.id2label) or 10,
            )
        )

        # ---------------------------------------------------------
        # Final fallback
        # ---------------------------------------------------------

        if not self.id2label:
            self.id2label = {
                i: f"LABEL_{i}"
                for i in range(num_labels)
            }

        # ---------------------------------------------------------
        # Default thresholds
        #
        # Each class uses 0.5 unless a custom thresholds file
        # is provided.
        # ---------------------------------------------------------

        self.thresholds = {
            label: 0.5
            for label in self.id2label.values()
        }

        # ---------------------------------------------------------
        # Optional custom thresholds file
        # ---------------------------------------------------------

        if thresholds_path is not None:

            thresholds_file = Path(thresholds_path)

            if thresholds_file.exists():

                obj = json.loads(
                    thresholds_file.read_text(
                        encoding="utf-8"
                    )
                )

                if isinstance(obj, dict):
                    for key, value in obj.items():

                        try:
                            self.thresholds[str(key)] = float(value)

                        except (TypeError, ValueError):
                            pass

    def predict_all_scores(
        self,
        text: str,
    ) -> dict[str, float]:
        """
        Return probability score for every classifier label.

        Example output:

        {
            "First Party Collection": 0.91,
            "Third Party Sharing": 0.14,
            ...
        }
        """

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with self.torch.no_grad():

            logits = self.model(
                **encoded
            ).logits[0]

            probabilities = (
                self.torch.sigmoid(logits)
                .detach()
                .cpu()
                .numpy()
            )

        return {
            self.id2label.get(
                i,
                f"LABEL_{i}",
            ): float(probabilities[i])

            for i in range(len(probabilities))
        }

    def predict_labels(
        self,
        text: str,
    ) -> list[str]:
        """
        Return labels whose probability is greater than or equal
        to their configured threshold.
        """

        scores = self.predict_all_scores(text)

        predicted_labels = []

        for label, score in scores.items():

            threshold = self.thresholds.get(
                label,
                0.5,
            )

            if score >= threshold:
                predicted_labels.append(label)

        return predicted_labels

    def predict(
        self,
        text: str,
    ) -> dict:
        """
        Convenience method that returns both:

        - predicted labels
        - probability scores
        """

        scores = self.predict_all_scores(text)

        labels = []

        for label, score in scores.items():

            threshold = self.thresholds.get(
                label,
                0.5,
            )

            if score >= threshold:
                labels.append(label)

        return {
            "labels": labels,
            "scores": scores,
            "model_source": self.model_source,
            "device": self.device,
        }


if __name__ == "__main__":

    print("Loading classifier...")

    classifier = XLMRClassifierAdapter()

    print()
    print("Model loaded successfully")
    print("Model source:", classifier.model_source)
    print("Device:", classifier.device)
    print("Number of labels:", len(classifier.id2label))

    sample_text = (
        "We collect personal information to provide our services "
        "and may share information with service providers."
    )

    print()
    print("Sample prediction:")
    print(classifier.predict(sample_text))