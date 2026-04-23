import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAlignment(nn.Module):
    def __init__(
        self,
        text_dim: int,
        visual_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, (
            f'hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})'
        )

        self.W_qt = nn.Linear(text_dim, hidden_dim)
        self.W_kt = nn.Linear(text_dim, hidden_dim)
        self.W_vv = nn.Linear(visual_dim, hidden_dim)
        self.W_qv = nn.Linear(visual_dim, hidden_dim)
        self.W_kv = nn.Linear(visual_dim, hidden_dim)
        self.W_tt = nn.Linear(text_dim, hidden_dim)
        self.fc_merge = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    @staticmethod
    def _adapt_seq_len(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        if tensor.size(1) == target_len:
            return tensor
        tensor = tensor.transpose(1, 2)
        tensor = F.adaptive_avg_pool1d(tensor, target_len)
        tensor = tensor.transpose(1, 2)
        return tensor

    def _multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~mask.unsqueeze(1).unsqueeze(2).bool(), float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        return attn_output

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        seq_len = text_features.size(1)
        num_regions = visual_features.size(1)

        Q_t = self.W_qt(text_features)
        K_t = self.W_kt(text_features)
        V_v = self.W_vv(visual_features)
        V_v = self._adapt_seq_len(V_v, seq_len)

        Y_t_star = self._multi_head_attention(Q_t, K_t, V_v, mask=text_mask)

        Q_v = self.W_qv(visual_features)
        K_v = self.W_kv(visual_features)
        V_t = self.W_tt(text_features)
        V_t = self._adapt_seq_len(V_t, num_regions)

        Y_v_star = self._multi_head_attention(Q_v, K_v, V_t, mask=visual_mask)
        Y_v_star = self._adapt_seq_len(Y_v_star, seq_len)

        merged = torch.cat([Y_t_star, Y_v_star], dim=-1)
        aligned_repr = self.fc_merge(merged)

        return aligned_repr, Y_t_star, Y_v_star


class TNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatingMechanism(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gate_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_linear(x))
        return gate * x


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiEvidenceFusionAlignmentModule(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 2048,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_heads: int = 8,
        num_alignment_pairs: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_alignment_pairs = num_alignment_pairs
        self.num_fusion_pairs = 2

        self.alignment_modules = nn.ModuleList(
            [
                CrossModalAlignment(
                    text_dim=text_dim,
                    visual_dim=visual_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_alignment_pairs)
            ]
        )

        self.alignment_ffns = nn.ModuleList(
            [
                FFN(
                    input_dim=hidden_dim * 2,
                    hidden_dim=hidden_dim * 2,
                    output_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_alignment_pairs)
            ]
        )

        self.visual_projectors = nn.ModuleList(
            [nn.Linear(visual_dim, hidden_dim) for _ in range(self.num_fusion_pairs)]
        )

        self.text_projector_sentence_emotion = nn.Linear(text_dim, hidden_dim)
        self.text_projector_sentence_interp = nn.Linear(text_dim, hidden_dim)
        self.text_projector_entity = nn.Linear(text_dim, hidden_dim)

        self.tnet = TNet(
            input_dim=visual_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim * 3,
        )

        mlp_input_dim = hidden_dim * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

        self.gate = GatingMechanism(output_dim)

    def forward(
        self,
        psi_T: torch.Tensor,
        psi_T_Em: torch.Tensor,
        psi_T_E: torch.Tensor,
        psi_T_EI: torch.Tensor,
        psi_V_O: torch.Tensor,
        psi_V_G: torch.Tensor,
        psi_V_E: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = psi_T.size(0)
        seq_len = psi_T.size(1)

        alignment_pairs = [
            (psi_T, psi_V_G),
            (psi_T_E, psi_V_E),
        ]

        all_Y_t_stars = []
        all_Y_v_stars = []

        for i, (text_feat, visual_feat) in enumerate(alignment_pairs):
            _, Y_t_star, Y_v_star = self.alignment_modules[i](
                text_features=text_feat,
                visual_features=visual_feat,
                text_mask=text_mask,
                visual_mask=visual_mask,
            )
            all_Y_t_stars.append(Y_t_star)
            all_Y_v_stars.append(Y_v_star)

        U_M = torch.zeros(batch_size, seq_len, self.hidden_dim, device=psi_T.device)
        for i in range(self.num_alignment_pairs):
            concat_aligned = torch.cat([all_Y_t_stars[i], all_Y_v_stars[i]], dim=-1)
            U_M = U_M + self.alignment_ffns[i](concat_aligned)

        X_T_sentence = (
            self.text_projector_sentence_emotion(psi_T_Em)
            + self.text_projector_sentence_interp(psi_T_EI)
        )
        X_T_entity = self.text_projector_entity(psi_T_E)

        text_paired = [X_T_sentence, X_T_entity]
        visual_paired = [psi_V_G, psi_V_E]

        fusion_sum = torch.zeros(batch_size, seq_len, self.hidden_dim * 3, device=psi_T.device)

        for i in range(self.num_fusion_pairs):
            X_T_i = text_paired[i]
            X_V_i = self.visual_projectors[i](visual_paired[i])
            if X_V_i.size(1) != seq_len:
                X_V_i = X_V_i.transpose(1, 2)
                X_V_i = F.adaptive_avg_pool1d(X_V_i, seq_len)
                X_V_i = X_V_i.transpose(1, 2)

            pair_concat = torch.cat([U_M, X_V_i, X_T_i], dim=-1)
            fusion_sum = fusion_sum + pair_concat

        tnet_output = self.tnet(psi_V_O)
        if tnet_output.size(1) != seq_len:
            tnet_output = tnet_output.transpose(1, 2)
            tnet_output = F.adaptive_avg_pool1d(tnet_output, seq_len)
            tnet_output = tnet_output.transpose(1, 2)

        fusion_result = fusion_sum + tnet_output
        mlp_output = self.fusion_mlp(fusion_result)
        h_star = self.gate(mlp_output)

        return h_star
