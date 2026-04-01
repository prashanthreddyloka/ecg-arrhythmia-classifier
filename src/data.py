from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import wfdb
except ImportError as exc:  # pragma: no cover - import guard for setup issues
    wfdb = None
    WFDB_IMPORT_ERROR = exc
else:
    WFDB_IMPORT_ERROR = None


DS1_RECORDS = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
]

DS2_RECORDS = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]

NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}
SUPPORTED_BEAT_SYMBOLS = NORMAL_BEAT_SYMBOLS | {
    "A",
    "a",
    "J",
    "S",
    "V",
    "E",
    "F",
    "/",
    "f",
    "Q",
}


@dataclass
class ProcessedSplit:
    sequence: np.ndarray
    fft_features: np.ndarray
    labels: np.ndarray


def ensure_wfdb_installed() -> None:
    if wfdb is None:
        raise ImportError(
            "wfdb is required for MIT-BIH loading. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from WFDB_IMPORT_ERROR


def download_mit_bih(dataset_dir: Path) -> None:
    ensure_wfdb_installed()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=str(dataset_dir))


def sliding_window_normalize(signal: np.ndarray, window_size: int = 31, eps: float = 1e-6) -> np.ndarray:
    normalized = np.empty_like(signal, dtype=np.float32)
    half_window = max(window_size // 2, 1)

    for idx in range(signal.shape[0]):
        start = max(0, idx - half_window)
        end = min(signal.shape[0], idx + half_window + 1)
        window = signal[start:end]
        normalized[idx] = (signal[idx] - window.mean()) / (window.std() + eps)

    return normalized


def extract_fft_features(beat: np.ndarray, fft_bins: int) -> np.ndarray:
    spectrum = np.fft.rfft(beat)
    magnitude = np.abs(spectrum)[1 : fft_bins + 1]
    if magnitude.shape[0] < fft_bins:
        magnitude = np.pad(magnitude, (0, fft_bins - magnitude.shape[0]))
    return magnitude.astype(np.float32)


def beat_symbol_to_label(symbol: str) -> int | None:
    if symbol not in SUPPORTED_BEAT_SYMBOLS:
        return None
    return 0 if symbol in NORMAL_BEAT_SYMBOLS else 1


def load_record_beats(
    dataset_dir: Path,
    record_name: str,
    window_size: int,
    fft_bins: int,
    pre_samples: int,
    post_samples: int,
    lead_index: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    ensure_wfdb_installed()
    record_path = str(dataset_dir / record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")
    signal = record.p_signal[:, lead_index].astype(np.float32)

    sequences: list[np.ndarray] = []
    fft_features: list[np.ndarray] = []
    labels: list[int] = []

    for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
        label = beat_symbol_to_label(symbol)
        if label is None:
            continue

        start = sample_idx - pre_samples
        end = sample_idx + post_samples
        if start < 0 or end > signal.shape[0]:
            continue

        beat = signal[start:end]
        beat = sliding_window_normalize(beat, window_size=window_size)
        sequences.append(beat[:, None])
        fft_features.append(extract_fft_features(beat, fft_bins=fft_bins))
        labels.append(label)

    return sequences, fft_features, labels


def build_split(
    dataset_dir: Path,
    records: list[str],
    window_size: int,
    fft_bins: int,
    pre_samples: int,
    post_samples: int,
    lead_index: int,
) -> ProcessedSplit:
    sequence_parts: list[np.ndarray] = []
    fft_parts: list[np.ndarray] = []
    label_parts: list[int] = []

    for record_name in records:
        sequences, fft_features, labels = load_record_beats(
            dataset_dir=dataset_dir,
            record_name=record_name,
            window_size=window_size,
            fft_bins=fft_bins,
            pre_samples=pre_samples,
            post_samples=post_samples,
            lead_index=lead_index,
        )
        sequence_parts.extend(sequences)
        fft_parts.extend(fft_features)
        label_parts.extend(labels)

    return ProcessedSplit(
        sequence=np.asarray(sequence_parts, dtype=np.float32),
        fft_features=np.asarray(fft_parts, dtype=np.float32),
        labels=np.asarray(label_parts, dtype=np.float32)[:, None],
    )


def save_processed_splits(output_path: Path, train_split: ProcessedSplit, test_split: ProcessedSplit) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_sequence=train_split.sequence,
        train_fft=train_split.fft_features,
        train_labels=train_split.labels,
        test_sequence=test_split.sequence,
        test_fft=test_split.fft_features,
        test_labels=test_split.labels,
    )


def load_processed_splits(input_path: Path) -> tuple[ProcessedSplit, ProcessedSplit]:
    loaded = np.load(input_path)
    train_split = ProcessedSplit(
        sequence=loaded["train_sequence"],
        fft_features=loaded["train_fft"],
        labels=loaded["train_labels"],
    )
    test_split = ProcessedSplit(
        sequence=loaded["test_sequence"],
        fft_features=loaded["test_fft"],
        labels=loaded["test_labels"],
    )
    return train_split, test_split


class ECGBeatDataset(Dataset):
    def __init__(self, sequence: np.ndarray, fft_features: np.ndarray, labels: np.ndarray) -> None:
        self.sequence = torch.tensor(sequence, dtype=torch.float32)
        self.fft_features = torch.tensor(fft_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.sequence.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequence[index], self.fft_features[index], self.labels[index]
