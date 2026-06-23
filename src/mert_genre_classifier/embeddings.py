from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from .audio import audio_from_dataset_value, load_audio_file, resample_to_mono
from .config import AppConfig
from .dataset import label_index_for_row, load_dataset_split, load_or_prepare_label_map, validate_dataset_columns
from .paths import embeddings_path, partial_embeddings_path


@dataclass
class MertRuntime:
    model: Any
    processor: Any
    device: str
    batch_size: int


def resolve_device(config: AppConfig) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("`torch` nao esta instalado.") from exc

    requested = config.runtime.device.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config pediu CUDA, mas `torch.cuda.is_available()` retornou False.")
    return requested


def batch_size_for_device(config: AppConfig, device: str) -> int:
    return config.runtime.batch_size_cuda if device.startswith("cuda") else config.runtime.batch_size_cpu


def load_mert_runtime(config: AppConfig) -> MertRuntime:
    try:
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "`torch` e `transformers` sao necessarios para carregar o MERT. "
            'Instale com `python -m pip install -e ".[dev]"`.'
        ) from exc

    device = resolve_device(config)
    batch_size = batch_size_for_device(config, device)
    model = AutoModel.from_pretrained(config.mert.model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config.mert.model_name,
        trust_remote_code=True,
    )
    torch.set_grad_enabled(False)
    return MertRuntime(model=model, processor=processor, device=device, batch_size=batch_size)


def embed_audio_arrays(runtime: MertRuntime, config: AppConfig, audio_arrays: list[Any]):
    import torch

    inputs = runtime.processor(
        audio_arrays,
        sampling_rate=config.mert.sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = runtime.model(**inputs, output_hidden_states=True)

    if config.mert.layer == "last":
        hidden = outputs.hidden_states[-1]
    else:
        hidden = outputs.hidden_states[int(config.mert.layer)]

    if config.mert.pooling != "mean":
        raise ValueError(f"Pooling nao suportado: {config.mert.pooling}")

    return hidden.mean(dim=1).detach().cpu().numpy()


def embed_audio_file(config: AppConfig, audio_path: str | Path):
    runtime = load_mert_runtime(config)
    audio_array, sample_rate = load_audio_file(audio_path)
    prepared = resample_to_mono(audio_array, sample_rate, config.mert.sample_rate)
    return embed_audio_arrays(runtime, config, [prepared])[0]


def extract_embeddings(
    config: AppConfig,
    split: str,
    max_samples: int | None = None,
    resume: bool | None = None,
) -> Path:
    import numpy as np
    from tqdm import tqdm

    config.ensure_artifact_dirs()
    effective_max_samples = (
        max_samples if max_samples is not None else config.embedding.max_samples_per_split
    )
    effective_resume = config.embedding.resume if resume is None else resume
    final_path = embeddings_path(config, split)
    partial_path = partial_embeddings_path(config, split)

    if effective_resume and final_path.exists():
        print(f"Embeddings ja existem: {final_path}")
        return final_path

    label_map = load_or_prepare_label_map(config)
    dataset = load_dataset_split(config, split, max_samples=effective_max_samples)
    validate_dataset_columns(dataset, config)
    runtime = load_mert_runtime(config)
    _warn_if_cpu_full(config, runtime.device, len(dataset), effective_max_samples)

    embeddings: list[Any] = []
    labels: list[int] = []
    item_ids: list[str] = []
    start_index = 0

    if effective_resume and partial_path.exists():
        partial = np.load(partial_path, allow_pickle=True)
        embeddings = [row for row in partial["embeddings"]]
        labels = [int(value) for value in partial["labels"]]
        item_ids = [str(value) for value in partial["item_ids"]]
        start_index = len(labels)
        print(f"Retomando cache parcial de {partial_path} a partir do item {start_index}.")

    batch_audio: list[Any] = []
    batch_labels: list[int] = []
    batch_ids: list[str] = []
    save_every = max(1, int(config.embedding.save_every_batches))
    batches_since_save = 0

    iterator = range(start_index, len(dataset))
    for index in tqdm(iterator, desc=f"Extracting {split}", unit="track"):
        row = dataset[index]
        try:
            audio_array, sample_rate = audio_from_dataset_value(
                row[config.dataset.audio_column],
                fallback_sample_rate=config.mert.sample_rate,
            )
        except Exception as exc:
            raise ValueError(
                f"Falha ao ler audio no split `{split}`, indice {index}, "
                f"song_id={row.get('song_id', 'desconhecido')}. "
                "Confira se a coluna de audio foi baixada/decodificada corretamente."
            ) from exc
        batch_audio.append(resample_to_mono(audio_array, sample_rate, config.mert.sample_rate))
        batch_labels.append(label_index_for_row(row, label_map, config))
        batch_ids.append(str(row.get("song_id", index)))

        if len(batch_audio) >= runtime.batch_size:
            batch_embeddings = embed_audio_arrays(runtime, config, batch_audio)
            embeddings.extend(batch_embeddings)
            labels.extend(batch_labels)
            item_ids.extend(batch_ids)
            batch_audio, batch_labels, batch_ids = [], [], []
            batches_since_save += 1
            if batches_since_save >= save_every:
                _save_npz(partial_path, embeddings, labels, item_ids, label_map, split, config)
                batches_since_save = 0

    if batch_audio:
        batch_embeddings = embed_audio_arrays(runtime, config, batch_audio)
        embeddings.extend(batch_embeddings)
        labels.extend(batch_labels)
        item_ids.extend(batch_ids)

    _save_npz(final_path, embeddings, labels, item_ids, label_map, split, config)
    if partial_path.exists():
        partial_path.unlink()
    print(f"Embeddings salvos em {final_path}")
    return final_path


def repair_embedding_labels(config: AppConfig, split: str) -> dict[str, Any]:
    import numpy as np

    path = embeddings_path(config, split)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings nao encontrados em {path}.")

    existing = np.load(path, allow_pickle=True)
    embeddings = existing["embeddings"]
    item_ids = [str(value) for value in existing["item_ids"]]
    old_labels = existing["labels"]
    label_map = load_or_prepare_label_map(config)
    dataset = load_dataset_split(
        config,
        split,
        max_samples=len(old_labels),
        decode_audio=False,
    )
    validate_dataset_columns(dataset, config)
    if len(dataset) != len(old_labels):
        raise ValueError(
            f"Split `{split}` tem {len(dataset)} linhas, mas o arquivo de embeddings tem "
            f"{len(old_labels)} labels. Nao e seguro reparar automaticamente."
        )

    new_labels = [label_index_for_row(dataset[index], label_map, config) for index in range(len(dataset))]
    backup_path = path.with_name(path.stem + ".before-label-repair.npz")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    _save_npz(path, embeddings, new_labels, item_ids, label_map, split, config)
    old_distribution = _label_distribution(old_labels, label_map["labels"])
    new_distribution = _label_distribution(new_labels, label_map["labels"])
    return {
        "path": str(path),
        "backup_path": str(backup_path),
        "num_examples": int(len(new_labels)),
        "old_distribution": old_distribution,
        "new_distribution": new_distribution,
    }


def _save_npz(
    path: Path,
    embeddings: list[Any],
    labels: list[int],
    item_ids: list[str],
    label_map: dict[str, Any],
    split: str,
    config: AppConfig,
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        item_ids=np.asarray(item_ids, dtype=str),
        label_names=np.asarray(label_map["labels"], dtype=str),
        split=np.asarray(split),
        model_name=np.asarray(config.mert.model_name),
        sample_rate=np.asarray(config.mert.sample_rate),
    )


def _warn_if_cpu_full(
    config: AppConfig,
    device: str,
    rows: int,
    effective_max_samples: int | None,
) -> None:
    if not device.startswith("cpu"):
        return
    if effective_max_samples is not None:
        return
    if rows < config.runtime.cpu_full_warning_threshold:
        return
    print(
        "\nAVISO: voce esta extraindo embeddings MERT em CPU para um split grande. "
        "Essa etapa pode demorar bastante localmente. Para execucao completa, prefira CUDA:\n"
        f"  python -m mert_genre_classifier -c configs/full_cuda.yaml extract-embeddings "
        f"--split {config.dataset.train_split} --resume\n"
        f"  python -m mert_genre_classifier -c configs/full_cuda.yaml extract-embeddings "
        f"--split {config.dataset.eval_split} --resume\n",
        file=sys.stderr,
    )


def _label_distribution(labels: Any, label_names: list[str]) -> dict[str, int]:
    from collections import Counter

    counts = Counter(int(label) for label in labels)
    return {
        label_names[index]: int(counts.get(index, 0))
        for index in range(len(label_names))
        if counts.get(index, 0) > 0
    }
