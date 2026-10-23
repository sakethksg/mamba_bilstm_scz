"""
Connectome-Enriched Spatio-Temporal Mamba-BiLSTM Classifier (No GAT).

Combines:
  1. Temporal Mamba SSM: Gated multi-scale dilated convolutions (1, 2, 4, 8 TRs).
  2. Temporal BiLSTM: Forward & backward recurrent dynamics along 100 TRs.
  3. Dynamic Functional Connectome (dFC) Projection: Direct linear projection
     of each ROI's whole-brain functional connectivity profile (200 -> d_model).
     Captures connectome topography WITHOUT graph convolutions or GAT.
  4. Spatial BiLSTM: Recurrent sequential modeling across the 200 cortical parcels.
  5. Multi-Scale Readout: ROI Attention Pooling + Global Mean + Global Max.
  6. Regularized Classification MLP Head.
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
        x: (N_nodes, L, d_model) -> transpose to (N_nodes, d_model, L)
        """
        h = self.in_proj(x.transpose(1, 2))

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


class AttentionPooling(nn.Module):
    """
    Learned attention-weighted pooling over a sequence dimension.
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
        x: (B, Seq_Len, feature_dim) -> (B, feature_dim)
        """
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)


class ROIDropout(nn.Module):
    """
    Randomly drops entire ROI parcel feature vectors during training
    to prevent the model from overfitting to idiosyncratic parcel co-activations.
    """
    def __init__(self, p: float = 0.08):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > self.p).float()
        return (x * mask) / (1.0 - self.p)


class MambaBiLSTMClassifier(nn.Module):
    """
    Connectome-Enriched Spatio-Temporal Mamba-BiLSTM Classifier (No GAT).
    """
    def __init__(
        self,
        n_rois: int = 200,
        window_size: int = 100,
        mamba_d_model: int = 48,
        mamba_expand: int = 2,
        temporal_lstm_hidden: int = 48,
        temporal_lstm_layers: int = 1,
        temporal_dropout: float = 0.20,
        spatial_lstm_hidden: int = 48,
        spatial_lstm_layers: int = 1,
        spatial_dropout: float = 0.25,
        roi_dropout: float = 0.08,
        feature_dropout: float = 0.25,
        classifier_dropout: float = 0.35,
        classifier_hidden: int = 96,
        num_classes: int = 2
    ):
        super().__init__()
        self.n_rois = n_rois
        self.window_size = window_size
        self.d_model = mamba_d_model
        d_spatial = spatial_lstm_hidden * 2  # Bidirectional
        self.roi_drop = ROIDropout(p=roi_dropout)

        # ─── 1. Temporal Mamba SSM Module ───
        self.temporal_input_proj = nn.Linear(1, mamba_d_model)
        self.mamba_ssm = ParallelTemporalSSMBlock(d_model=mamba_d_model, expand=mamba_expand)
        self.mamba_norm = nn.LayerNorm(mamba_d_model)

        # ─── 2. Temporal BiLSTM Module ───
        self.temporal_bilstm = nn.LSTM(
            input_size=mamba_d_model,
            hidden_size=temporal_lstm_hidden,
            num_layers=temporal_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=temporal_dropout if temporal_lstm_layers > 1 else 0.0
        )
        self.temporal_lstm_proj = nn.Linear(temporal_lstm_hidden * 2, mamba_d_model)
        self.temporal_lstm_norm = nn.LayerNorm(mamba_d_model)

        # ─── 3. Temporal Readout ───
        self.temporal_attn_pool = AttentionPooling(mamba_d_model)
        self.temporal_out_proj = nn.Linear(mamba_d_model * 2, mamba_d_model)
        self.temporal_final_norm = nn.LayerNorm(mamba_d_model)

        # ─── 4. Functional Connectome (dFC) Linear Injection (NO GAT) ───
        # Injects each ROI's whole-brain functional connectivity profile + graph summary statistics directly
        fc_in_dim = n_rois + 3  # 200 raw FC + degree centrality + pos FC + neg FC
        self.fc_proj = nn.Sequential(
            nn.Linear(fc_in_dim, mamba_d_model),
            nn.GELU(),
            nn.LayerNorm(mamba_d_model),
            nn.Dropout(feature_dropout),
            nn.Linear(mamba_d_model, mamba_d_model)
        )
        self.fusion_gate = nn.Linear(mamba_d_model * 2, mamba_d_model)
        self.fusion_norm = nn.LayerNorm(mamba_d_model)
        self.feature_drop = nn.Dropout(feature_dropout)

        # ─── 5. Spatial BiLSTM Module ───
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, n_rois, mamba_d_model) * 0.02)
        self.spatial_bilstm = nn.LSTM(
            input_size=mamba_d_model,
            hidden_size=spatial_lstm_hidden,
            num_layers=spatial_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=spatial_dropout if spatial_lstm_layers > 1 else 0.0
        )
        self.spatial_res_proj = nn.Linear(mamba_d_model, d_spatial)
        self.spatial_norm = nn.LayerNorm(d_spatial)
        self.spatial_drop = nn.Dropout(spatial_dropout)

        # ─── 6. Multi-Scale Brain Readout ───
        self.spatial_attn_pool = AttentionPooling(d_spatial)
        pooled_dim = d_spatial * 4  # mean, max, std, attention

        # ─── 7. Classifier MLP Head (LayerNorm for stability across all batch sizes) ───
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.LayerNorm(classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout * 0.5),
            nn.Linear(classifier_hidden // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (B, 200, 100) raw windowed BOLD signals
            fc: (B, 200, 200) windowed Fisher-z functional connectivity matrix
        Returns:
            logits: (B, num_classes)
        """
        B = x.size(0)

        # Reshape to (B * 200, 100, 1) for temporal sequence modeling
        x_flat = x.view(B * self.n_rois, self.window_size, 1)

        # --- Stage 1: Temporal Mamba SSM ---
        h_mamba_in = self.temporal_input_proj(x_flat)
        h_ssm = self.mamba_ssm(h_mamba_in)
        h_ssm = self.mamba_norm(h_mamba_in + h_ssm)

        # --- Stage 2: Temporal BiLSTM ---
        h_lstm_raw, _ = self.temporal_bilstm(h_ssm)
        h_lstm = self.temporal_lstm_proj(h_lstm_raw)
        h_temporal = self.temporal_lstm_norm(h_ssm + h_lstm)

        # --- Stage 3: Temporal Dual Readout ---
        attn_pool = self.temporal_attn_pool(h_temporal)
        mean_pool = torch.mean(h_temporal, dim=1)
        h_temporal_roi = torch.cat([attn_pool, mean_pool], dim=-1)
        h_temporal_roi = self.temporal_final_norm(self.temporal_out_proj(h_temporal_roi))
        h_temporal_roi = h_temporal_roi.view(B, self.n_rois, self.d_model)

        # --- Stage 4: Functional Connectome (dFC) Injection (NO GAT) ---
        # Topological feature extraction: Degree centrality, pos & neg connectivity
        fc_abs = fc.abs().mean(dim=-1, keepdim=True)
        fc_pos = torch.clamp(fc, min=0.0).mean(dim=-1, keepdim=True)
        fc_neg = torch.clamp(fc, max=0.0).abs().mean(dim=-1, keepdim=True)
        fc_enriched = torch.cat([fc, fc_abs, fc_pos, fc_neg], dim=-1)  # (B, 200, 203)

        h_fc = self.fc_proj(fc_enriched)  # (B, 200, d_model)

        # Gated Residual Fusion: dynamically blends temporal dynamics and connectome topology
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_temporal_roi, h_fc], dim=-1)))
        h_fused = self.fusion_norm(gate * h_temporal_roi + (1.0 - gate) * h_fc + h_temporal_roi)
        h_fused = self.feature_drop(h_fused)

        # --- Stage 5: Spatial BiLSTM ---
        # Apply ROIDropout to prevent overfitting to specific parcels
        h_spatial_in = self.roi_drop(h_fused) + self.spatial_pos_embed
        h_spatial_lstm, _ = self.spatial_bilstm(h_spatial_in)  # (B, 200, d_spatial)
        h_res = self.spatial_res_proj(h_spatial_in)
        h_brain = self.spatial_norm(h_spatial_lstm + h_res)
        h_brain = self.spatial_drop(h_brain)

        # --- Stage 6: Multi-Scale Brain Readout ---
        brain_mean = torch.mean(h_brain, dim=1)
        brain_max, _ = torch.max(h_brain, dim=1)
        brain_std = torch.std(h_brain, dim=1)
        brain_attn = self.spatial_attn_pool(h_brain)
        brain_embed = torch.cat([brain_mean, brain_max, brain_std, brain_attn], dim=-1)  # (B, 4 * d_spatial)

        # --- Stage 7: Classifier Head ---
        logits = self.classifier(brain_embed)
        return logits
