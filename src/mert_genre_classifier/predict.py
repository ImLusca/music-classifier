from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .embeddings import embed_audio_file
from .models import load_classifier


def predict_audio(
    config: AppConfig,
    audio_path: str | Path,
    model_name: str = "mlp",
    top_k: int = 5,
) -> dict[str, Any]:
    import numpy as np

    bundle = load_classifier(config, model_name)
    labels = [str(label) for label in bundle["labels"]]
    embedding = embed_audio_file(config, audio_path).reshape(1, -1)
    pipeline = bundle["pipeline"]
    predicted_index = int(pipeline.predict(embedding)[0])

    probabilities = _probabilities_for_all_labels(pipeline, embedding, len(labels))
    ranked = sorted(
        (
            {"label": labels[index], "probability": float(probabilities[index])}
            for index in range(len(labels))
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    top = ranked[: max(1, top_k)]
    return {
        "audio_path": str(audio_path),
        "model": model_name,
        "prediction": labels[predicted_index],
        "prediction_index": predicted_index,
        "top_k": top,
        "embedding_model": config.mert.model_name,
    }


def _probabilities_for_all_labels(pipeline: Any, embedding: Any, num_labels: int):
    import numpy as np

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(embedding)[0]
        classes = getattr(pipeline, "classes_", np.arange(len(proba)))
        result = np.zeros(num_labels, dtype=np.float64)
        for column, class_index in enumerate(classes):
            result[int(class_index)] = float(proba[column])
        return result

    prediction = int(pipeline.predict(embedding)[0])
    result = np.zeros(num_labels, dtype=np.float64)
    result[prediction] = 1.0
    return result

