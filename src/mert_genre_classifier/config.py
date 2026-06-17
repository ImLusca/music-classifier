from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    name: str = "lewtun/music_genres"
    train_split: str = "train"
    eval_split: str = "test"
    audio_column: str = "audio"
    label_column: str = "genre"
    label_id_column: str = "genre_id"


@dataclass
class MertConfig:
    model_name: str = "m-a-p/MERT-v1-95M"
    sample_rate: int = 24000
    layer: str = "last"
    pooling: str = "mean"


@dataclass
class RuntimeConfig:
    device: str = "auto"
    batch_size_cpu: int = 1
    batch_size_cuda: int = 8
    cpu_full_warning_threshold: int = 1000


@dataclass
class EmbeddingConfig:
    max_samples_per_split: int | None = None
    resume: bool = True
    save_every_batches: int = 10


@dataclass
class ClassifierConfig:
    logistic_max_iter: int = 2000
    logistic_class_weight: str | None = "balanced"
    mlp_hidden_layer_sizes: tuple[int, ...] = (256, 64)
    mlp_max_iter: int = 500
    mlp_alpha: float = 0.0001
    mlp_early_stopping: bool = True


@dataclass
class ArtifactsConfig:
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


@dataclass
class AppConfig:
    run_name: str = "local_smoke"
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mert: MertConfig = field(default_factory=MertConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source_path: Path | None = None) -> "AppConfig":
        classifier_data = dict(data.get("classifier", {}))
        if "mlp_hidden_layer_sizes" in classifier_data:
            classifier_data["mlp_hidden_layer_sizes"] = tuple(
                int(value) for value in classifier_data["mlp_hidden_layer_sizes"]
            )

        artifacts_data = dict(data.get("artifacts", {}))
        for key in ("processed_dir", "models_dir", "reports_dir"):
            if key in artifacts_data and artifacts_data[key] is not None:
                artifacts_data[key] = Path(artifacts_data[key])

        return cls(
            run_name=str(data.get("run_name", "local_smoke")),
            seed=int(data.get("seed", 42)),
            dataset=DatasetConfig(**dict(data.get("dataset", {}))),
            mert=MertConfig(**dict(data.get("mert", {}))),
            runtime=RuntimeConfig(**dict(data.get("runtime", {}))),
            embedding=EmbeddingConfig(**dict(data.get("embedding", {}))),
            classifier=ClassifierConfig(**classifier_data),
            artifacts=ArtifactsConfig(**artifacts_data),
            source_path=source_path,
        )

    def ensure_artifact_dirs(self) -> None:
        self.artifacts.processed_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.reports_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML nao esta instalado. Instale as dependencias com "
            '`python -m pip install -e ".[dev]"`.'
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config invalida em {config_path}: esperado um mapa YAML.")
    return AppConfig.from_mapping(raw, source_path=config_path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value

