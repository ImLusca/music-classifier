from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any


def audio_from_dataset_value(audio_value: Any, fallback_sample_rate: int | None = None) -> tuple[Any, int]:
    if isinstance(audio_value, dict):
        return _audio_from_mapping(audio_value, fallback_sample_rate=fallback_sample_rate)

    if isinstance(audio_value, (str, Path)):
        return load_audio_file(audio_value)

    if isinstance(audio_value, (bytes, bytearray)):
        return load_audio_bytes(bytes(audio_value))

    if _looks_like_waveform(audio_value):
        if fallback_sample_rate is None:
            raise ValueError(
                "Audio do dataset veio como waveform sem `sampling_rate`. "
                "Defina um sample rate de fallback antes de processar."
            )
        return audio_value, int(fallback_sample_rate)

    raise ValueError(
        "Valor de audio inesperado. Esperado dict do Hugging Face Audio, caminho, "
        f"bytes ou waveform; recebido {_describe_value(audio_value)}."
    )


def load_audio_file(path: str | Path) -> tuple[Any, int]:
    try:
        import torch
        import torchaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "`torch` e `torchaudio` sao necessarios para carregar audio local."
        ) from exc

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    return waveform.detach().cpu().numpy(), int(sample_rate)


def load_audio_bytes(audio_bytes: bytes) -> tuple[Any, int]:
    try:
        import soundfile as sf
    except ModuleNotFoundError:
        sf = None

    if sf is not None:
        try:
            array, sample_rate = sf.read(BytesIO(audio_bytes), always_2d=False)
            return array, int(sample_rate)
        except Exception:
            pass

    try:
        import torchaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "`soundfile` ou `torchaudio` sao necessarios para decodificar bytes de audio."
        ) from exc

    waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    return waveform.detach().cpu().numpy(), int(sample_rate)


def resample_to_mono(audio_array: Any, source_rate: int, target_rate: int):
    try:
        import numpy as np
        import torch
        import torchaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "`numpy`, `torch` e `torchaudio` sao necessarios para preparar audio."
        ) from exc

    audio = torch.as_tensor(np.asarray(audio_array), dtype=torch.float32)
    if audio.ndim == 2:
        if audio.shape[0] <= audio.shape[1]:
            audio = audio.mean(dim=0)
        else:
            audio = audio.mean(dim=1)
    if audio.ndim != 1:
        audio = audio.flatten()

    if source_rate != target_rate:
        audio = torchaudio.transforms.Resample(source_rate, target_rate)(audio)
    return audio.detach().cpu().numpy()


def _audio_from_mapping(audio_value: dict[str, Any], fallback_sample_rate: int | None) -> tuple[Any, int]:
    if "array" in audio_value and audio_value["array"] is not None:
        sample_rate = audio_value.get("sampling_rate", fallback_sample_rate)
        if sample_rate is None:
            raise ValueError("Audio do dataset contem `array`, mas nao contem `sampling_rate`.")
        return audio_value["array"], int(sample_rate)

    if "bytes" in audio_value and audio_value["bytes"] is not None:
        return load_audio_bytes(audio_value["bytes"])

    if "path" in audio_value and audio_value["path"]:
        return load_audio_file(audio_value["path"])

    raise ValueError(
        "Audio do dataset veio como dict, mas sem `array`, `bytes` ou `path` utilizavel. "
        f"Chaves recebidas: {sorted(audio_value.keys())}"
    )


def _looks_like_waveform(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return True
    module = type(value).__module__.split(".")[0]
    return module in {"numpy", "torch", "array"}


def _describe_value(value: Any) -> str:
    type_name = type(value).__name__
    if isinstance(value, dict):
        return f"dict com chaves {sorted(value.keys())}"
    if hasattr(value, "shape"):
        return f"{type_name} com shape {getattr(value, 'shape')}"
    return type_name
