from __future__ import annotations

import torch
from torch import nn


class ECGArrhythmiaClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        fft_feature_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + fft_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sequence: torch.Tensor, fft_features: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm(sequence)
        forward_hidden = hidden_state[-2]
        backward_hidden = hidden_state[-1]
        temporal_embedding = torch.cat([forward_hidden, backward_hidden], dim=1)
        combined = torch.cat([temporal_embedding, fft_features], dim=1)
        return self.classifier(combined)
