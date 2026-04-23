from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from modelscope.models.builder import MODELS
from modelscope.utils.config import ConfigDict

from adaseq.data.constant import PAD_LABEL_ID
from adaseq.metainfo import Models, Pipelines, Tasks
from adaseq.models.base import Model
from adaseq.modules.decoders import CRF, PartialCRF
from adaseq.modules.dropouts import WordDropout
from adaseq.modules.embedders import Embedder
from adaseq.modules.encoders import Encoder
from adaseq.modules.hgmaf_modules import MultiEvidenceFusionAlignmentModule
from adaseq.modules.util import get_tokens_mask


@MODELS.register_module(Tasks.named_entity_recognition, module_name=Models.hgmaf_model)
class HGMAFModel(Model):
    pipeline = Pipelines.sequence_labeling_pipeline

    def __init__(
        self,
        id_to_label: Dict[int, str],
        embedder: Union[Embedder, ConfigDict],
        encoder: Optional[Union[Encoder, ConfigDict]] = None,
        dropout: float = 0.0,
        word_dropout: bool = False,
        use_crf: Optional[bool] = True,
        partial: Optional[bool] = False,
        visual_dim: int = 2048,
        alignment_hidden_dim: int = 256,
        num_heads: int = 8,
        num_visual_regions: int = 49,
        hgmaf_dropout: float = 0.1,
        fallback_to_text_when_no_evidence: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.id_to_label = id_to_label
        self.num_labels = len(id_to_label)

        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder.from_config(embedder)
        hidden_size = self.embedder.get_output_dim()

        if encoder is None:
            self.encoder = None
        else:
            if isinstance(encoder, Encoder):
                self.encoder = encoder
            else:
                self.encoder = Encoder.from_config(encoder)
            assert hidden_size == self.encoder.get_input_dim()
            hidden_size = self.encoder.get_output_dim()

        self.text_dim = hidden_size
        self.visual_dim = visual_dim
        self.num_visual_regions = num_visual_regions
        self.fallback_to_text_when_no_evidence = fallback_to_text_when_no_evidence

        self.mefa = MultiEvidenceFusionAlignmentModule(
            text_dim=self.text_dim,
            visual_dim=visual_dim,
            hidden_dim=alignment_hidden_dim,
            output_dim=self.text_dim,
            num_heads=num_heads,
            num_alignment_pairs=2,
            dropout=hgmaf_dropout,
        )

        self.linear = nn.Linear(self.text_dim, self.num_labels)
        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            if word_dropout:
                self.dropout = WordDropout(dropout)
            else:
                self.dropout = nn.Dropout(dropout)

        self.use_crf = use_crf
        if use_crf:
            if partial:
                self.crf = PartialCRF(self.num_labels, batch_first=True)
            else:
                self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=PAD_LABEL_ID)

    def _create_zero_evidence(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> dict:
        return {
            'psi_T_Em': torch.zeros(batch_size, seq_len, self.text_dim, device=device),
            'psi_T_E': torch.zeros(batch_size, seq_len, self.text_dim, device=device),
            'psi_T_EI': torch.zeros(batch_size, seq_len, self.text_dim, device=device),
            'psi_V_O': torch.zeros(
                batch_size, self.num_visual_regions, self.visual_dim, device=device
            ),
            'psi_V_G': torch.zeros(
                batch_size, self.num_visual_regions, self.visual_dim, device=device
            ),
            'psi_V_E': torch.zeros(
                batch_size, self.num_visual_regions, self.visual_dim, device=device
            ),
        }

    def _collect_evidence(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        evidence_text_emotion: Optional[torch.Tensor] = None,
        evidence_text_entity: Optional[torch.Tensor] = None,
        evidence_text_entity_interp: Optional[torch.Tensor] = None,
        evidence_visual_original: Optional[torch.Tensor] = None,
        evidence_visual_generated: Optional[torch.Tensor] = None,
        evidence_visual_entity: Optional[torch.Tensor] = None,
    ) -> dict:
        evidence = self._create_zero_evidence(batch_size, seq_len, device)
        if evidence_text_emotion is not None:
            evidence['psi_T_Em'] = evidence_text_emotion
        if evidence_text_entity is not None:
            evidence['psi_T_E'] = evidence_text_entity
        if evidence_text_entity_interp is not None:
            evidence['psi_T_EI'] = evidence_text_entity_interp
        if evidence_visual_original is not None:
            evidence['psi_V_O'] = evidence_visual_original
        if evidence_visual_generated is not None:
            evidence['psi_V_G'] = evidence_visual_generated
        if evidence_visual_entity is not None:
            evidence['psi_V_E'] = evidence_visual_entity
        return evidence

    @staticmethod
    def _has_any_evidence_input(*evidence_tensors: Optional[torch.Tensor]) -> bool:
        return any(tensor is not None for tensor in evidence_tensors)

    def forward(
        self,
        tokens: Dict[str, Any],
        label_ids: Optional[torch.LongTensor] = None,
        meta: Optional[Dict[str, Any]] = None,
        origin_tokens: Optional[Dict[str, Any]] = None,
        origin_mask: Optional[torch.Tensor] = None,
        evidence_text_emotion: Optional[torch.Tensor] = None,
        evidence_text_entity: Optional[torch.Tensor] = None,
        evidence_text_entity_interp: Optional[torch.Tensor] = None,
        evidence_visual_original: Optional[torch.Tensor] = None,
        evidence_visual_generated: Optional[torch.Tensor] = None,
        evidence_visual_entity: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        evidence_inputs = (
            evidence_text_emotion,
            evidence_text_entity,
            evidence_text_entity_interp,
            evidence_visual_original,
            evidence_visual_generated,
            evidence_visual_entity,
        )
        use_text_fallback = (
            self.fallback_to_text_when_no_evidence
            and not self._has_any_evidence_input(*evidence_inputs)
        )

        if use_text_fallback:
            hidden_states = self._encode_text(tokens, apply_dropout=self.use_dropout)
        else:
            bert_output = self._encode_text(tokens)
            batch_size, seq_len, _ = bert_output.shape
            device = bert_output.device

            collected_evidence = self._collect_evidence(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                evidence_text_emotion=evidence_text_emotion,
                evidence_text_entity=evidence_text_entity,
                evidence_text_entity_interp=evidence_text_entity_interp,
                evidence_visual_original=evidence_visual_original,
                evidence_visual_generated=evidence_visual_generated,
                evidence_visual_entity=evidence_visual_entity,
            )

            h_star = self.mefa(
                psi_T=bert_output,
                psi_T_Em=collected_evidence['psi_T_Em'],
                psi_T_E=collected_evidence['psi_T_E'],
                psi_T_EI=collected_evidence['psi_T_EI'],
                psi_V_O=collected_evidence['psi_V_O'],
                psi_V_G=collected_evidence['psi_V_G'],
                psi_V_E=collected_evidence['psi_V_E'],
                text_mask=get_tokens_mask(tokens, seq_len),
                visual_mask=None,
            )

            hidden_states = h_star
            if self.use_dropout:
                hidden_states = self.dropout(hidden_states)

        logits = self.linear(hidden_states)
        crf_mask = get_tokens_mask(tokens, logits.size(1)) if origin_mask is None else origin_mask

        if self.training and label_ids is not None:
            loss = self._calculate_loss(logits, label_ids, crf_mask)
            outputs = {'logits': logits, 'loss': loss}
        else:
            predicts = self.decode(logits, crf_mask)
            outputs = {'logits': logits, 'predicts': predicts}

        return outputs

    def _encode_text(self, tokens: Dict[str, Any], apply_dropout: bool = False) -> torch.Tensor:
        x = self.embedder(**tokens)

        if apply_dropout and self.use_dropout:
            x = self.dropout(x)

        if self.encoder is not None:
            mask = get_tokens_mask(tokens, x.size(1))
            x = self.encoder(x, mask)

            if apply_dropout and self.use_dropout:
                x = self.dropout(x)

        return x

    def _calculate_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.use_crf:
            targets = targets * mask
            loss = -self.crf(logits, targets, reduction='mean', mask=mask)
        else:
            loss = self.loss_fn(logits.transpose(1, 2), targets)
        return loss

    def decode(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> Union[List, torch.LongTensor]:
        if self.use_crf:
            predicts = self.crf.decode(logits, mask=mask).squeeze(0)
        else:
            predicts = logits.argmax(-1)
        return predicts
