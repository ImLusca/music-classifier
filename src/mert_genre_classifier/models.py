from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .paths import classifier_path, embeddings_path


MODEL_NAMES = ("logistic", "mlp")


def load_embeddings(config: AppConfig, split: str, max_samples: int | None = None):
    import numpy as np

    path = embeddings_path(config, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings nao encontrados em {path}. Rode `extract-embeddings --split {split}`."
        )
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    label_names = [str(label) for label in data["label_names"]]
    item_ids = [str(item_id) for item_id in data["item_ids"]]

    if max_samples is not None:
        limit = min(max_samples, len(labels))
        embeddings = embeddings[:limit]
        labels = labels[:limit]
        item_ids = item_ids[:limit]

    return {
        "embeddings": embeddings,
        "labels": labels,
        "label_names": label_names,
        "item_ids": item_ids,
        "path": path,
        "label_distribution": label_distribution(labels, label_names),
    }


def label_distribution(labels: Any, label_names: list[str]) -> dict[str, int]:
    from collections import Counter

    counts = Counter(int(label) for label in labels)
    return {
        label_names[index]: int(counts.get(index, 0))
        for index in range(len(label_names))
        if counts.get(index, 0) > 0
    }


def train_classifiers(
    config: AppConfig,
    train_split: str | None = None,
    model_name: str = "all",
    max_samples: int | None = None,
) -> dict[str, Path]:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    config.ensure_artifact_dirs()
    split = train_split or config.dataset.train_split
    data = load_embeddings(config, split, max_samples=max_samples)
    x_train = data["embeddings"]
    y_train = data["labels"]
    if len(set(int(label) for label in y_train)) < 2:
        raise ValueError("Treinamento precisa de pelo menos duas classes.")

    requested = MODEL_NAMES if model_name == "all" else (model_name,)
    saved: dict[str, Path] = {}
    for name in requested:
        if name == "logistic":
            estimator = LogisticRegression(
                max_iter=config.classifier.logistic_max_iter,
                class_weight=config.classifier.logistic_class_weight,
                random_state=config.seed,
                n_jobs=None,
            )
        elif name == "mlp":
            estimator = MLPClassifier(
                hidden_layer_sizes=config.classifier.mlp_hidden_layer_sizes,
                max_iter=config.classifier.mlp_max_iter,
                alpha=config.classifier.mlp_alpha,
                early_stopping=config.classifier.mlp_early_stopping,
                random_state=config.seed,
            )
        else:
            raise ValueError(f"Modelo desconhecido: {name}. Use um de {MODEL_NAMES} ou `all`.")

        pipeline = make_pipeline(StandardScaler(), estimator)
        pipeline.fit(x_train, y_train)
        output_path = classifier_path(config, name)
        bundle = {
            "model_name": name,
            "pipeline": pipeline,
            "labels": data["label_names"],
            "train_split": split,
            "embedding_model": config.mert.model_name,
            "config": config.to_dict(),
        }
        joblib.dump(bundle, output_path)
        saved[name] = output_path
    return saved


def load_classifier(config: AppConfig, model_name: str) -> dict[str, Any]:
    import joblib

    path = classifier_path(config, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Classificador nao encontrado em {path}. Rode `train` primeiro.")
    bundle = joblib.load(path)
    if "pipeline" not in bundle or "labels" not in bundle:
        raise ValueError(f"Arquivo de classificador invalido: {path}")
    return bundle


def demo_status(config: AppConfig) -> dict[str, Any]:
    splits = [config.dataset.train_split, config.dataset.eval_split]
    embeddings = {
        split: {
            "path": str(embeddings_path(config, split)),
            "exists": embeddings_path(config, split).exists(),
        }
        for split in splits
    }
    classifiers = {
        name: {
            "path": str(classifier_path(config, name)),
            "exists": classifier_path(config, name).exists(),
        }
        for name in MODEL_NAMES
    }
    ready_for_train = embeddings[config.dataset.train_split]["exists"]
    ready_for_evaluate = (
        embeddings[config.dataset.eval_split]["exists"]
        and any(item["exists"] for item in classifiers.values())
    )
    ready_for_predict = any(item["exists"] for item in classifiers.values())
    return {
        "run_name": config.run_name,
        "ready_for_train": ready_for_train,
        "ready_for_evaluate": ready_for_evaluate,
        "ready_for_predict": ready_for_predict,
        "embeddings": embeddings,
        "classifiers": classifiers,
        "notes": [
            "Para treino/avaliacao local, copie os embeddings do Colab para data/processed/.",
            "Para predizer uma musica nova ao vivo, deixe um classificador treinado e o MERT baixado/cacheado.",
        ],
    }
