import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerNet(nn.Module):
    def __init__(
        self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1
    ):
        super().__init__()
        self.l1 = nn.Linear(1, d_model)
        self.l2 = nn.Linear(1, d_model)
        self.positional_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.l3 = nn.Linear(d_model, 1)

    def forward(self, x, t):
        mask = nn.Transformer.generate_square_subsequent_mask(t.shape[-1]).cuda()
        x = x.unsqueeze(-1)
        x = self.l1(x)
        x = self.positional_encoder(x)
        t = t.unsqueeze(-1)
        t = self.l2(t)
        t = self.positional_encoder(t)
        x = self.transformer(x, t, tgt_mask=mask)
        x = self.l3(x).squeeze(-1)
        return x
