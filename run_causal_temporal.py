#!/usr/bin/env python3
"""
CausalTemporalNetwork - Stock Price Forecasting with Visualization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ============================================================================
# Core Components
# ============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class DilatedCausalBlock(nn.Module):
    def __init__(self, channels, kernel_size=2, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalConv1d(channels, channels, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        self.activations = nn.ModuleList([nn.GELU() for _ in range(num_layers)])

    def forward(self, x):
        for conv, act in zip(self.layers, self.activations):
            residual = x
            x = act(conv(x)) + residual
        return x


class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class ScaleInvariantNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class CausalTemporalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=4,
                 n_heads=4, kernel_size=3, forecast_horizon=1, dropout=0.1):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv_stack = DilatedCausalBlock(hidden_dim, kernel_size=kernel_size, num_layers=num_layers)
        self.norm1 = ScaleInvariantNorm(hidden_dim)
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                CausalAttention(hidden_dim, n_heads, dropout),
                ScaleInvariantNorm(hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(2)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * forecast_horizon)
        )

    def forward(self, x):
        B, T, F = x.shape
        h = self.input_proj(x)
        h = self.conv_stack(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm1(h)
        for attn_layer in self.attention_layers:
            h = h + attn_layer(h)
        out = self.output_proj(h[:, -1, :])
        return out.view(B, self.forecast_horizon, self.output_dim)


# ============================================================================
# Data & Training
# ============================================================================

def generate_causal_timeseries(n_samples=1000, seq_len=100):
    """True model: y[t] = 0.7*y[t-1] + 0.2*x[t-1] + noise"""
    data = []
    for _ in range(n_samples):
        x = np.random.randn(seq_len)
        y = np.zeros(seq_len)
        for t in range(1, seq_len):
            y[t] = 0.7 * y[t-1] + 0.2 * x[t-1] + 0.1 * np.random.randn()
        data.append(np.stack([y, x], axis=-1))
    return np.array(data)


def train_and_visualize():
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate data
    data = generate_causal_timeseries(n_samples=1000, seq_len=100)
    train_data = torch.FloatTensor(data[:800])
    test_data = torch.FloatTensor(data[800:])

    lookback, horizon = 90, 10
    X_train = train_data[:, :lookback, :]
    y_train = train_data[:, lookback:lookback+horizon, 0:1]
    X_test = test_data[:, :lookback, :]
    y_test = test_data[:, lookback:lookback+horizon, 0:1]

    # Model
    model = CausalTemporalNetwork(input_dim=2, hidden_dim=32, output_dim=1,
                                   num_layers=4, forecast_horizon=horizon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # Training
    train_losses, test_losses = [], []
    print("Training CausalTemporalNetwork...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            test_losses.append(criterion(model(X_test), y_test).item())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train={train_losses[-1]:.4f}, Test={test_losses[-1]:.4f}")

    # Predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    actuals = y_test.numpy()

    # =========================================================================
    # Visualization
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CausalTemporalNetwork - Stock Price Forecasting Results', fontsize=14, fontweight='bold')

    # 1. Training curves
    ax1 = axes[0, 0]
    ax1.plot(train_losses, label='Train Loss', color='#2563eb', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', color='#dc2626', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Sample predictions (4 test sequences)
    ax2 = axes[0, 1]
    for i in range(4):
        offset = i * 2.5
        ax2.plot(range(horizon), actuals[i, :, 0] + offset, 'o-', color='#2563eb',
                 label='Actual' if i == 0 else '', markersize=4, linewidth=1.5)
        ax2.plot(range(horizon), predictions[i, :, 0] + offset, 's--', color='#dc2626',
                 label='Predicted' if i == 0 else '', markersize=4, linewidth=1.5)
        ax2.axhline(y=offset, color='gray', linestyle=':', alpha=0.3)
        ax2.text(-0.5, offset, f'Seq {i+1}', fontsize=9, va='center')
    ax2.set_xlabel('Forecast Step')
    ax2.set_ylabel('Value (offset for clarity)')
    ax2.set_title('Sample Predictions vs Actuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Full sequence context
    ax3 = axes[1, 0]
    seq_idx = 0
    full_seq = test_data[seq_idx, :, 0].numpy()
    time = np.arange(len(full_seq))

    ax3.plot(time[:lookback], full_seq[:lookback], '-', color='#2563eb', linewidth=1.5, label='History (input)')
    ax3.plot(time[lookback:], full_seq[lookback:], 'o-', color='#16a34a', linewidth=2, markersize=5, label='Actual future')
    ax3.plot(time[lookback:lookback+horizon], predictions[seq_idx, :, 0], 's--', color='#dc2626',
             linewidth=2, markersize=5, label='Predicted')
    ax3.axvline(x=lookback, color='gray', linestyle='--', alpha=0.7, label='Forecast start')
    ax3.fill_between(time[lookback:lookback+horizon],
                     predictions[seq_idx, :, 0] - 0.3,
                     predictions[seq_idx, :, 0] + 0.3,
                     color='#dc2626', alpha=0.2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Value')
    ax3.set_title('Full Sequence: 90 Steps History → 10 Steps Forecast')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 4. Error distribution
    ax4 = axes[1, 1]
    errors = (predictions - actuals).flatten()
    ax4.hist(errors, bins=50, color='#8b5cf6', edgecolor='white', alpha=0.8)
    ax4.axvline(x=0, color='#dc2626', linestyle='--', linewidth=2, label=f'Zero error')
    ax4.axvline(x=errors.mean(), color='#2563eb', linestyle='-', linewidth=2,
                label=f'Mean: {errors.mean():.3f}')
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution (RMSE: {np.sqrt((errors**2).mean()):.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('forecast_results.png', dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: forecast_results.png")
    plt.show()


if __name__ == "__main__":
    train_and_visualize()
