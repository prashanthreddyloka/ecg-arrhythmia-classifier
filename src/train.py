from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from data import ECGBeatDataset, load_processed_splits
from model import ECGArrhythmiaClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ECG arrhythmia detector on MIT-BIH beats.")
    parser.add_argument("--data-path", type=Path, default=Path("data/processed/mitbih_beats.npz"))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positives = float(labels.sum())
    negatives = float(labels.shape[0] - positives)
    return torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for sequence, fft_features, label in dataloader:
            sequence = sequence.to(device)
            fft_features = fft_features.to(device)
            logits = model(sequence, fft_features)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            probabilities.extend(probs.tolist())
            targets.extend(label.numpy().ravel().tolist())

    y_true = np.asarray(targets, dtype=np.int32)
    y_score = np.asarray(probabilities, dtype=np.float32)
    y_pred = (y_score >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)

    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_split, test_split = load_processed_splits(args.data_path)
    train_dataset = ECGBeatDataset(train_split.sequence, train_split.fft_features, train_split.labels)
    test_dataset = ECGBeatDataset(test_split.sequence, test_split.fft_features, test_split.labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGArrhythmiaClassifier(
        input_size=train_split.sequence.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        fft_feature_size=train_split.fft_features.shape[1],
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=build_pos_weight(train_split.labels).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_auc = 0.0

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_ecg_arrhythmia_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for sequence, fft_features, label in train_loader:
            sequence = sequence.to(device)
            fft_features = fft_features.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(sequence, fft_features)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * sequence.size(0)

        avg_loss = running_loss / len(train_dataset)
        metrics = evaluate(model, test_loader, device, threshold=args.threshold)
        print(
            f"Epoch {epoch:02d} | "
            f"loss={avg_loss:.4f} | "
            f"auc={metrics['auc']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"sensitivity={metrics['sensitivity']:.4f} | "
            f"specificity={metrics['specificity']:.4f} | "
            f"precision={metrics['precision']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "metrics": metrics,
                },
                best_model_path,
            )

    print(f"Saved best model to {best_model_path}")


if __name__ == "__main__":
    main()
