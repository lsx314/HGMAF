# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch
from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg, default_group
from transformers import PreTrainedTokenizerBase

from adaseq.data.batch import DataBatch
from adaseq.data.constant import HGMAF_EVIDENCE_FIELDS, HGMAF_META_FIELDS
from adaseq.metainfo import DataCollators

DATA_COLLATORS = Registry('data_collators')


def build_data_collator(
    tokenizer: PreTrainedTokenizerBase, cfg: ConfigDict, default_args: Optional[dict] = None
):
    """build data collator from config."""
    if default_args is None:
        default_args = {}
    default_args['tokenizer'] = tokenizer
    return build_from_cfg(cfg, DATA_COLLATORS, group_key=default_group, default_args=default_args)


@DATA_COLLATORS.register_module(module_name=DataCollators.data_collator_with_padding)
class DataCollatorWithPadding:
    """
    A `DataCollator` support padding some fields to same length.
    Support padding encoder related fields: input_ids, token_type_ids, mask,
    and padding other fields with `default_pad_id`.
    `no_pad_fields` will be skipped.
    """

    tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        tokenizer,
        default_pad_id: int = 0,
        no_pad_fields: Optional[Set[str]] = None,
        keep_fields: Optional[Set[str]] = None,
        tensor_pad_fields: Optional[Set[str]] = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.default_pad_id = default_pad_id
        self.keep_fields: Set[str] = {'tokens', 'origin_tokens', 'meta'}
        if keep_fields is not None:
            self.keep_fields |= set(keep_fields)
        self.no_pad_fields = no_pad_fields or set()
        self.tensor_pad_fields: Set[str] = set(HGMAF_EVIDENCE_FIELDS)
        if tensor_pad_fields is not None:
            self.tensor_pad_fields |= set(tensor_pad_fields)
        # Keep raw multimodal metadata as python objects when these fields are
        # explicitly passed through by the preprocessor.
        self.keep_fields |= set(HGMAF_META_FIELDS)

    @staticmethod
    def _get_pad_func(padding_side: str) -> Callable:
        if padding_side == 'right':

            def _pad(array, size: int, pad_value):
                return array + [pad_value] * size

        elif padding_side == 'left':

            def _pad(array, size: int, pad_value):
                return [pad_value] * size + array

        else:
            raise ValueError('Invalid padding strategy:' + str(padding_side))
        return _pad

    def padding_token(self, batch: Dict[str, Any], padding_side: str) -> Dict[str, Any]:
        """pad token related fields (hf.transformers style)"""
        _pad = self._get_pad_func(padding_side)
        batch_size = len(batch['meta'])
        for field in [f for f in batch.keys() if f.endswith('tokens')]:
            sub_field_pair = [
                ('input_ids', self.tokenizer.pad_token_id),
                ('attention_mask', False),
                ('mask', False),
            ]
            if 'token_type_ids' in batch[field][0]:
                sub_field_pair.append(('token_type_ids', self.tokenizer.pad_token_type_id))
            if 'offsets' in batch[field][0]:
                sub_field_pair.append(('offsets', (0, 0)))

            sub = 'has_special_tokens'
            try:
                padded_tokens = {sub: [batch[field][i][sub] for i in range(batch_size)]}
            except KeyError:
                padded_tokens = {}

            for sub, pad_value in sub_field_pair:
                if sub not in batch[field][0]:
                    continue
                padded_field = list()
                max_length = max(len(i[sub]) for i in batch[field])
                for i in range(batch_size):
                    difference = max_length - len(batch[field][i][sub])
                    if difference > 0:
                        padded_field.append(_pad(batch[field][i][sub], difference, pad_value))
                    else:
                        padded_field.append(batch[field][i][sub])
                padded_tokens[sub] = padded_field
            batch[field] = padded_tokens

        field = 'origin_mask'
        if field in batch:
            max_length = max(len(i) for i in batch[field])
            for i in range(batch_size):
                difference = max_length - len(batch[field][i])
                if difference > 0:
                    batch[field][i] = _pad(batch[field][i], difference, False)

        return batch

    def padding(
        self,
        batch: Dict[str, Any],
        padding_side: str,
        fields: Optional[Union[Set[str], str]] = None,
        pad_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """pad other fields."""
        pad_id = self.default_pad_id if pad_id is None else pad_id
        _pad = self._get_pad_func(padding_side)

        if fields is None:
            fields = set(batch.keys())
        elif isinstance(fields, str):
            fields = {fields}
        fields -= self.keep_fields.union(self.no_pad_fields).union(self.tensor_pad_fields)

        for field in fields:
            if batch[field][0] is None:
                continue
            if not isinstance(batch[field][0], list):
                continue
            max_length = max(len(i) for i in batch[field])
            for i in range(len(batch[field])):
                difference = max_length - len(batch[field][i])
                if difference > 0:
                    batch[field][i] = _pad(batch[field][i], difference, pad_id)
        return batch

    def padding_tensor_fields(self, batch: Dict[str, Any], padding_side: str) -> Dict[str, Any]:
        """Pad tensor-like optional fields along the first dimension.

        This is mainly used for future HGMAF evidence tensors, whose first
        dimension can vary with sequence length.
        """
        for field in list(self.tensor_pad_fields):
            if field not in batch:
                continue

            values = batch[field]
            if all(value is None for value in values):
                del batch[field]
                continue

            reference = None
            tensors = []
            for value in values:
                if value is None:
                    tensors.append(None)
                    continue
                tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
                if reference is None:
                    reference = tensor
                else:
                    if tensor.dim() != reference.dim():
                        raise ValueError(
                            f'Inconsistent rank for field `{field}`: '
                            f'{tensor.dim()} vs {reference.dim()}'
                        )
                    if tuple(tensor.shape[1:]) != tuple(reference.shape[1:]):
                        raise ValueError(
                            f'Inconsistent trailing shape for field `{field}`: '
                            f'{tuple(tensor.shape[1:])} vs {tuple(reference.shape[1:])}'
                        )
                tensors.append(tensor)

            if reference is None:
                del batch[field]
                continue

            if reference.dim() == 0:
                batch[field] = [
                    tensor if tensor is not None else reference.new_zeros(())
                    for tensor in tensors
                ]
                continue

            max_length = max(tensor.size(0) if tensor is not None else 0 for tensor in tensors)
            padded_tensors = []
            for tensor in tensors:
                if tensor is None:
                    padded_tensors.append(reference.new_zeros((max_length,) + tuple(reference.shape[1:])))
                    continue

                if tensor.size(0) == max_length:
                    padded_tensors.append(tensor)
                    continue

                pad_tensor = tensor.new_zeros((max_length - tensor.size(0),) + tuple(tensor.shape[1:]))
                if padding_side == 'left':
                    padded_tensors.append(torch.cat([pad_tensor, tensor], dim=0))
                else:
                    padded_tensors.append(torch.cat([tensor, pad_tensor], dim=0))

            batch[field] = padded_tensors

        return batch

    def __call__(self, instances: List[Dict[str, Any]]) -> DataBatch:
        """pad list of instances to batch"""
        ordered_keys = []
        seen = set()
        for instance in instances:
            for key in instance.keys():
                if key not in seen:
                    ordered_keys.append(key)
                    seen.add(key)

        batch = {
            key: [instance.get(key, None) for instance in instances]
            for key in ordered_keys
            if any(key in instance for instance in instances)
        }
        padding_side = self.tokenizer.padding_side
        padded_batch = self.padding_token(batch, padding_side)
        padded_batch = self.padding_tensor_fields(padded_batch, padding_side)
        padded_batch = self.padding(padded_batch, padding_side=padding_side)
        batch = DataBatch(padded_batch, self.keep_fields)
        return batch
