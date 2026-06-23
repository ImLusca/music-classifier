from __future__ import annotations

from collections.abc import Iterable
import sys
from typing import Any

from .audio import audio_from_dataset_value
from .config import AppConfig
from .io import read_json, write_json
from .paths import labels_path, prepare_summary_path


def configured_splits(config: AppConfig) -> list[str]:
    splits = [config.dataset.train_split, config.dataset.eval_split]
    unique: list[str] = []
    for split in splits:
        if split not in unique:
            unique.append(split)
    return unique


def load_dataset_split(
    config: AppConfig,
    split: str,
    max_samples: int | None = None,
    decode_audio: bool = True,
):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "A biblioteca `datasets` nao esta instalada. Rode "
            '`python -m pip install -e ".[dev]"` antes de acessar o dataset.'
        ) from exc

    requested_split = _split_with_limit(split, max_samples)
    dataset = load_dataset(config.dataset.name, split=requested_split)
    if decode_audio:
        dataset = cast_audio_for_decoding(dataset, config)
    else:
        dataset = disable_audio_decoding(dataset, config)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def validate_dataset_columns(dataset: Any, config: AppConfig) -> None:
    columns = set(dataset.column_names)
    required = {config.dataset.audio_column, config.dataset.label_column}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            f"Dataset split nao possui as colunas obrigatorias: {missing}. "
            f"Colunas encontradas: {sorted(columns)}"
        )


def disable_audio_decoding(dataset: Any, config: AppConfig):
    try:
        from datasets import Audio
    except ModuleNotFoundError:
        return dataset
    return dataset.cast_column(config.dataset.audio_column, Audio(decode=False))


def cast_audio_for_decoding(dataset: Any, config: AppConfig):
    if config.dataset.audio_column not in dataset.column_names:
        return dataset
    try:
        from datasets import Audio
    except ModuleNotFoundError:
        return dataset

    try:
        return dataset.cast_column(
            config.dataset.audio_column,
            Audio(sampling_rate=config.mert.sample_rate),
        )
    except Exception as exc:
        print(
            "AVISO: nao foi possivel converter a coluna de audio com datasets.Audio; "
            "o pipeline tentara decodificar o valor bruto. "
            f"Motivo: {exc}",
            file=sys.stderr,
        )
        return dataset


def build_label_map(rows: Iterable[dict[str, Any]], label_column: str, label_id_column: str | None = None):
    id_to_label: dict[int, str] = {}
    labels_without_ids: set[str] = set()

    for row in rows:
        if label_column not in row:
            raise ValueError(f"Linha sem coluna de genero `{label_column}`.")
        label = str(row[label_column])
        if label_id_column and label_id_column in row and row[label_id_column] is not None:
            source_id = int(row[label_id_column])
            previous = id_to_label.get(source_id)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Mesmo {label_id_column}={source_id} aponta para `{previous}` e `{label}`."
                )
            id_to_label[source_id] = label
        else:
            labels_without_ids.add(label)

    if id_to_label:
        ordered_pairs = sorted(id_to_label.items())
        labels = [label for _, label in ordered_pairs]
        source_ids = {label: source_id for source_id, label in ordered_pairs}
    else:
        labels = sorted(labels_without_ids)
        source_ids = {label: index for index, label in enumerate(labels)}

    if not labels:
        raise ValueError("Nenhum label de genero encontrado no dataset.")

    label_to_index = {label: index for index, label in enumerate(labels)}
    source_id_to_index = {str(source_id): label_to_index[label] for label, source_id in source_ids.items()}

    return {
        "labels": labels,
        "label_to_index": label_to_index,
        "source_ids": source_ids,
        "source_id_to_index": source_id_to_index,
    }


def label_index_for_row(row: dict[str, Any], label_map: dict[str, Any], config: AppConfig) -> int:
    label_id = row.get(config.dataset.label_id_column)
    if label_id is not None:
        key = str(int(label_id))
        if key in label_map["source_id_to_index"]:
            return int(label_map["source_id_to_index"][key])

    label = row.get(config.dataset.label_column)
    if label is not None:
        label_text = str(label)
        if label_text in label_map["label_to_index"]:
            return int(label_map["label_to_index"][label_text])

    raise ValueError(f"Nao foi possivel mapear label para a linha: {row}")


def load_label_map(config: AppConfig) -> dict[str, Any]:
    path = labels_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Mapa de labels nao encontrado em {path}. Rode `prepare-data` primeiro."
        )
    return read_json(path)


def prepare_data(config: AppConfig, split: str | None = None, max_samples: int | None = None) -> dict[str, Any]:
    config.ensure_artifact_dirs()
    splits = [split] if split else configured_splits(config)
    effective_max_samples = (
        max_samples if max_samples is not None else config.embedding.max_samples_per_split
    )
    label_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset": config.dataset.name,
        "splits": {},
        "config": config.to_dict(),
    }

    for split_name in splits:
        dataset = load_dataset_split(
            config,
            split_name,
            max_samples=effective_max_samples,
            decode_audio=False,
        )
        validate_dataset_columns(dataset, config)
        summary["splits"][split_name] = {
            "rows": len(dataset),
            "columns": list(dataset.column_names),
            "max_samples": effective_max_samples,
        }

        for row in dataset:
            label_rows.append(
                {
                    config.dataset.label_column: row[config.dataset.label_column],
                    config.dataset.label_id_column: row.get(config.dataset.label_id_column),
                }
            )

    label_map = build_label_map(
        label_rows,
        label_column=config.dataset.label_column,
        label_id_column=config.dataset.label_id_column,
    )
    summary["labels"] = label_map["labels"]
    summary["num_labels"] = len(label_map["labels"])

    write_json(labels_path(config), label_map)
    write_json(prepare_summary_path(config), summary)
    return summary


def load_or_prepare_label_map(config: AppConfig) -> dict[str, Any]:
    path = labels_path(config)
    if path.exists():
        return read_json(path)
    prepare_data(config)
    return read_json(path)


def inspect_audio_sample(config: AppConfig, split: str | None = None, index: int = 0) -> dict[str, Any]:
    split_name = split or config.dataset.train_split
    dataset = load_dataset_split(config, split_name, max_samples=index + 1)
    validate_dataset_columns(dataset, config)
    if index >= len(dataset):
        raise IndexError(f"Indice {index} fora do split `{split_name}` com {len(dataset)} linhas.")

    row = dataset[index]
    audio_value = row[config.dataset.audio_column]
    audio_array, sample_rate = audio_from_dataset_value(
        audio_value,
        fallback_sample_rate=config.mert.sample_rate,
    )
    shape = getattr(audio_array, "shape", None)
    if shape is not None:
        shape = [int(value) for value in shape]
    elif hasattr(audio_array, "__len__"):
        shape = [len(audio_array)]

    return {
        "split": split_name,
        "index": index,
        "song_id": row.get("song_id"),
        "raw_audio_type": type(audio_value).__name__,
        "raw_audio_module": type(audio_value).__module__,
        "decoded_sample_rate": int(sample_rate),
        "decoded_shape": shape,
        "genre": row.get(config.dataset.label_column),
        "genre_id": row.get(config.dataset.label_id_column),
    }


def _split_with_limit(split: str, max_samples: int | None) -> str:
    if max_samples is None:
        return split
    if "[" in split:
        return split
    return f"{split}[:{max_samples}]"
