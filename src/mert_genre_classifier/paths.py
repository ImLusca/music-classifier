from __future__ import annotations

from pathlib import Path

from .config import AppConfig


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    chars = []
    for char in lowered:
        if char.isalnum():
            chars.append(char)
        elif char in {".", "-", "_"}:
            chars.append(char)
        else:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "value"


def model_slug(config: AppConfig) -> str:
    return slugify(config.mert.model_name.split("/")[-1])


def split_slug(split: str) -> str:
    return slugify(split.replace("/", "_"))


def labels_path(config: AppConfig) -> Path:
    return config.artifacts.processed_dir / f"{config.run_name}_labels.json"


def prepare_summary_path(config: AppConfig) -> Path:
    return config.artifacts.processed_dir / f"{config.run_name}_prepare_summary.json"


def embeddings_path(config: AppConfig, split: str) -> Path:
    return (
        config.artifacts.processed_dir
        / f"{config.run_name}_{split_slug(split)}_{model_slug(config)}_embeddings.npz"
    )


def partial_embeddings_path(config: AppConfig, split: str) -> Path:
    final_path = embeddings_path(config, split)
    return final_path.with_name(final_path.stem + ".partial.npz")


def classifier_path(config: AppConfig, model_name: str) -> Path:
    return config.artifacts.models_dir / f"{config.run_name}_{slugify(model_name)}.joblib"


def metrics_path(config: AppConfig, split: str, model_name: str) -> Path:
    return (
        config.artifacts.reports_dir
        / f"{config.run_name}_{split_slug(split)}_{slugify(model_name)}_metrics.json"
    )


def confusion_matrix_path(config: AppConfig, split: str, model_name: str) -> Path:
    return (
        config.artifacts.reports_dir
        / f"{config.run_name}_{split_slug(split)}_{slugify(model_name)}_confusion_matrix.csv"
    )

