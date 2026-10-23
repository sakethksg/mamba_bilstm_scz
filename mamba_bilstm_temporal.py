"""
Mamba + Bidirectional LSTM (BiLSTM) Temporal Encoder for fMRI BOLD Timeseries.

Combines:
  1. Gated Multi-Scale Dilated State-Space Modeling (Mamba/S4 equivalent):
     Fast continuous-time state-space dynamics across scales (1, 2, 4, 8 TRs).
  2. Bidirectional LSTM:
     Captures forward and backward recurrent temporal dependencies,
     phase relationships, and long-range hemodynamic co-activation profiles.
  3. Temporal Self-Attention Pooling:
     Dynamically weights salient BOLD transient activations rather than naive averaging.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelTemporalSSMBlock(nn.Module):
    """
    Parallel State-Space Sequence Modeling block.
    Captures multi-scale temporal dynamics across windowed BOLD signals.
    """
    def __init__(self, d_model: int = 32, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand

        self.in_proj = nn.Conv1d(d_model, d_inner, kernel_size=1)

        # Multi-scale dilated state filters (dilations: 1, 2, 4, 8 TRs)
        self.conv_d1 = nn.Conv1d(d_inner, d_inner * 2, kernel_size=5, padding=2, groups=d_inner)
        self.conv_d2 = nn.Conv1d(d_inner, d_inner * 2, kernel_size=5, padding=4, dilation=2, groups=d_inner)
        self.conv_d4 = nn.Conv1d(d_inner, d_inner * 2, kernel_size=5, padding=8, dilation=4, groups=d_inner)
        self.conv_d8 = nn.Conv1d(d_inner, d_inner * 2, kernel_size=5, padding=16, dilation=8, groups=d_inner)

        self.norm = nn.GroupNorm(num_groups=4, num_channels=d_inner)
        self.act = nn.SiLU()
        self.out_proj = nn.Conv1d(d_inner, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_nodes, L, d_model)
        Returns:
            (N_nodes, L, d_model)
        """
        h = self.in_proj(x.transpose(1, 2))  # (N, d_inner, L)

        # Multi-scale gated filtering with residual accumulations
        p1, g1 = self.conv_d1(h).chunk(2, dim=1)
        h = h + p1 * torch.sigmoid(g1)

        p2, g2 = self.conv_d2(h).chunk(2, dim=1)
        h = h + p2 * torch.sigmoid(g2)

        p3, g3 = self.conv_d4(h).chunk(2, dim=1)
        h = h + p3 * torch.sigmoid(g3)

        p4, g4 = self.conv_d8(h).chunk(2, dim=1)
        h = h + p4 * torch.sigmoid(g4)

        h = self.act(self.norm(h))
        out = self.out_proj(h)
        return out.transpose(1, 2)  # (N_nodes, L, d_model)


class TemporalAttentionPooling(nn.Module):
    """
    Learned attention-weighted pooling over temporal timesteps.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_nodes, L, feature_dim)
        Returns:
            weighted_pool: (N_nodes, feature_dim)
        """
        scores = self.attn(x)  # (N, L, 1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # (N, feature_dim)
        return pooled


class MambaBiLSTMTemporalEncoder(nn.Module):
    """
    Hybrid Mamba-BiLSTM temporal feature extractor for fMRI parcels.
    Transforms raw windowed BOLD signals into expressive static parcel embeddings.

    Pipeline:
      1. Linear input embedding: 1 -> d_model
      2. Mamba multi-scale SSM block: (N, L, d_model)
      3. Bidirectional LSTM: (N, L, 2 * lstm_hidden)
      4. Residual skip-connection + LayerNorm
      5. Dual pooling: Temporal Attention + Mean pooling -> output_dim
    """
    def __init__(
        self,
        in_features: int = 1,
        d_model: int = 32,
        mamba_expand: int = 2,
        lstm_hidden: int = 32,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.15
    ):
        super().__init__()
        self.d_model = d_model
        self.lstm_out_dim = lstm_hidden * 2  # Bidirectional

        # 1. Input embedding
        self.input_embed = nn.Linear(in_features, d_model)

        # 2. Mamba Selective SSM
        self.ssm_block = ParallelTemporalSSMBlock(d_model=d_model, expand=mamba_expand)
        self.ssm_norm = nn.LayerNorm(d_model)

        # 3. Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )
        self.lstm_proj = nn.Linear(self.lstm_out_dim, d_model)
        self.bilstm_norm = nn.LayerNorm(d_model)

        # 4. Temporal Readout: Dual Attention + Mean Pooling
        self.attn_pool = TemporalAttentionPooling(d_model)
        # Concatenation of attention-pooled + mean-pooled gives 2 * d_model
        self.out_dim = d_model * 2
        self.out_proj = nn.Linear(self.out_dim, d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_nodes, L)  e.g. (total_nodes, 100)
        Returns:
            node_embed: (N_nodes, d_model)
        """
        # (N, L, 1) -> (N, L, d_model)
        x_seq = self.input_embed(x.unsqueeze(-1))

        # --- Stage 1: Mamba State-Space Modeling ---
        h_ssm = self.ssm_block(x_seq)
        h_ssm = self.ssm_norm(x_seq + h_ssm)

        # --- Stage 2: Bidirectional LSTM ---
        h_lstm_raw, _ = self.bilstm(h_ssm)  # (N, L, 2*lstm_hidden)
        h_lstm = self.lstm_proj(h_lstm_raw)  # (N, L, d_model)
        h_temporal = self.bilstm_norm(h_ssm + h_lstm)  # Residual connection

        # --- Stage 3: Dual Temporal Readout ---
        attn_pooled = self.attn_pool(h_temporal)      # (N, d_model)
        mean_pooled = torch.mean(h_temporal, dim=1)   # (N, d_model)

        combined = torch.cat([attn_pooled, mean_pooled], dim=-1)  # (N, 2*d_model)
        out = self.final_norm(self.out_proj(combined))            # (N, d_model)

        return out
