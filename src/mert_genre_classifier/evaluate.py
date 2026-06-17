from __future__ import annotations

from typing import Any

from .config import AppConfig
from .io import write_json
from .models import MODEL_NAMES, load_classifier, load_embeddings
from .paths import confusion_matrix_path, metrics_path


def evaluate_classifiers(
    config: AppConfig,
    split: str | None = None,
    model_name: str = "all",
    max_samples: int | None = None,
) -> dict[str, dict[str, Any]]:
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    config.ensure_artifact_dirs()
    eval_split = split or config.dataset.eval_split
    data = load_embeddings(config, eval_split, max_samples=max_samples)
    x_eval = data["embeddings"]
    y_eval = data["labels"]
    labels = data["label_names"]

    requested = MODEL_NAMES if model_name == "all" else (model_name,)
    results: dict[str, dict[str, Any]] = {}
    for name in requested:
        bundle = load_classifier(config, name)
        pipeline = bundle["pipeline"]
        predictions = pipeline.predict(x_eval)
        macro_f1 = float(f1_score(y_eval, predictions, average="macro", zero_division=0))
        accuracy = float(accuracy_score(y_eval, predictions))
        report = classification_report(
            y_eval,
            predictions,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_eval, predictions, labels=list(range(len(labels))))

        metrics = {
            "model": name,
            "split": eval_split,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "num_examples": int(len(y_eval)),
            "labels": labels,
            "classification_report": report,
            "embedding_path": str(data["path"]),
        }
        write_json(metrics_path(config, eval_split, name), metrics)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            confusion_matrix_path(config, eval_split, name)
        )
        results[name] = metrics
    return results

