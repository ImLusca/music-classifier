from __future__ import annotations

from pathlib import Path
from typing import Any


def audio_from_dataset_value(audio_value: Any) -> tuple[Any, int]:
    if not isinstance(audio_value, dict):
        raise ValueError("Valor de audio inesperado; esperado dict com `array` e `sampling_rate`.")
    if "array" not in audio_value or "sampling_rate" not in audio_value:
        raise ValueError("Audio do dataset precisa conter `array` e `sampling_rate`.")
    return audio_value["array"], int(audio_value["sampling_rate"])


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

