# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os.path as osp
from typing import Dict
import datasets
from datasets import Features, Value
from adaseq.data.constant import PAD_LABEL
from .base import CustomDatasetBuilder


class NamedEntityRecognitionDatasetBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for Named Entity Recognition datasets"""

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super().__init__(data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class NamedEntityRecognitionDatasetBuilder(CustomDatasetBuilder):
    """Builder for entity typing datasets.

    features:
        id: string, data record id.
        tokens: list[str] input tokens.
        spans: List[Dict],  mentions like: [{'start': 0, 'end': 2, 'type': 'PER'}]
        mask: bool, mention mask.
    """

    BUILDER_CONFIG_CLASS = NamedEntityRecognitionDatasetBuilderConfig

    def stub():  # noqa: D102
        pass

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features(
                {
                    'id': Value('string'),
                    'tokens': [Value('string')],
                    'spans': [
                        {
                            'start': Value('int32'),  # close
                            'end': Value('int32'),  # open
                            'type': Value('string'),
                        }
                    ],
                    'mask': [Value('bool')],
                    'image_id': Value('string'),
                    'image_path': Value('string'),
                    'aux_data': Value('string'),
                }
            )
        )
        return info

    @staticmethod
    def _resolve_image_path(image_id: str, corpus_config: Dict) -> str:
        """Resolve image path from image_id when an image directory is provided."""
        if not image_id:
            return ''

        image_dir = corpus_config.get('image_dir') or corpus_config.get('image_root')
        if not image_dir:
            return ''

        filename = image_id
        image_ext = corpus_config.get('image_ext')
        if image_ext:
            image_ext = image_ext if str(image_ext).startswith('.') else f'.{image_ext}'
            if not str(filename).lower().endswith(str(image_ext).lower()):
                filename = f'{filename}{image_ext}'

        return osp.join(image_dir, str(filename))

    @staticmethod
    def _load_aux_mapping(corpus_config: Dict):
        """Load optional sidecar metadata for multimodal/HGMAF interfaces.

        Supported formats:
        - JSONL: one dict per line
        - JSON list: [{...}, {...}]
        - JSON dict: {"key": {...}, ...}
        """
        aux_info_file = corpus_config.get('aux_info_file')
        if not aux_info_file:
            return {}, corpus_config.get('aux_info_key', 'image_id')

        aux_info_key = corpus_config.get('aux_info_key', 'image_id')
        with open(aux_info_file, encoding='utf-8') as f:
            if aux_info_file.endswith('.jsonl'):
                raw_items = [json.loads(line) for line in f if line.strip()]
            else:
                raw_items = json.load(f)

        if isinstance(raw_items, dict):
            if all(isinstance(v, dict) for v in raw_items.values()):
                return {str(k): v for k, v in raw_items.items()}, aux_info_key
            raw_items = raw_items.get('items', [])

        mapping = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            lookup_key = item.get(aux_info_key)
            if lookup_key is None:
                continue
            mapping[str(lookup_key)] = item
        return mapping, aux_info_key

    @staticmethod
    def _split_metadata(example: Dict, reserved_keys):
        """Split explicit metadata fields from arbitrary extra keys.

        Unknown keys are serialized into aux_data so future multimodal inputs
        can be added without changing the builder schema again.
        """
        image_id = str(example.get('image_id', '') or example.get('img_id', '') or '')
        image_path = str(example.get('image_path', '') or '')

        aux_payload = {}
        aux_data = example.get('aux_data')
        if isinstance(aux_data, dict):
            aux_payload.update(aux_data)
        elif isinstance(aux_data, str) and aux_data:
            try:
                parsed_aux = json.loads(aux_data)
                if isinstance(parsed_aux, dict):
                    aux_payload.update(parsed_aux)
            except json.JSONDecodeError:
                aux_payload['aux_data_raw'] = aux_data

        for key, value in example.items():
            if key in reserved_keys or key in {'image_id', 'img_id', 'image_path', 'aux_data'}:
                continue
            aux_payload[key] = value

        return image_id, image_path, aux_payload

    @staticmethod
    def _compose_record(
        guid: int,
        tokens,
        spans,
        mask,
        image_id: str = '',
        image_path: str = '',
        aux_payload: Dict = None,
    ):
        return {
            'id': str(guid),
            'tokens': tokens,
            'spans': spans,
            'mask': mask,
            'image_id': image_id or '',
            'image_path': image_path or '',
            'aux_data': json.dumps(aux_payload or {}, ensure_ascii=False),
        }

    def _generate_examples(self, filepath):
        """
        NER reader supports:
        1. data_type: conll
            ```
            duck B-PER
            duck I-PER
            duck O
            ```

        2. data_type: json_tags
        ```
        {
            'text': 'duck duck duck duck',
            'labels': ['B-PER', 'O', ...]
        }
        ```

        3. data_type: json_spans
        ```
        {
            'text': 'duck duck',
            'spans': [
                {'start': 0, 'end': 1, 'type': 'PER'},
                ...
            ]
        ```

        4. data_type: cluener
        ```
        {
            'text': 'duck duck',
            'label': {
                'LOC': [[0, 1], ...]
            }
        }
        ```
        """
        corpus_config = self.config.corpus_config
        if corpus_config['data_type'] == 'conll':
            return self._load_conll_file(filepath, corpus_config)
        elif corpus_config['data_type'] == 'json_tags':
            return self._load_json_tags_file(filepath, corpus_config)
        elif corpus_config['data_type'] == 'json_spans':
            return self._load_json_spans_file(filepath, corpus_config)
        elif corpus_config['data_type'] == 'cluener':
            return self._load_cluener_file(filepath, corpus_config)
        else:
            raise ValueError('Unknown corpus format type [%s]' % corpus_config['data_type'])

    @classmethod
    def _load_conll_file(cls, file_path, corpus_config: Dict):
        delimiter = corpus_config.get('delimiter', None)
        aux_mapping, aux_info_key = cls._load_aux_mapping(corpus_config)
        with open(file_path, encoding='utf-8') as f:
            guid = 0
            tokens = []
            labels = []
            image_id = ''
            for line in f:
                if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                    if tokens:
                        spans = cls._labels_to_spans(labels)
                        mask = cls._labels_to_mask(labels)
                        image_path = cls._resolve_image_path(image_id, corpus_config)
                        aux_payload = {}
                        lookup_candidates = [str(image_id), str(guid)]
                        for lookup_key in lookup_candidates:
                            if lookup_key and lookup_key in aux_mapping:
                                aux_payload = dict(aux_mapping[lookup_key])
                                break
                        image_id = str(aux_payload.pop('image_id', image_id) or image_id or '')
                        image_path = str(aux_payload.pop('image_path', image_path) or image_path or '')
                        aux_payload.pop(aux_info_key, None)
                        yield guid, cls._compose_record(
                            guid=guid,
                            tokens=tokens,
                            spans=spans,
                            mask=mask,
                            image_id=image_id,
                            image_path=image_path,
                            aux_payload=aux_payload,
                        )
                        guid += 1
                        tokens = []
                        labels = []
                        image_id = ''
                elif line.startswith('IMGID:'):
                    image_id = line.split(':', 1)[1].strip()
                else:
                    splits = line.split(delimiter)
                    tokens.append(splits[0])
                    labels.append(splits[-1].rstrip())
            if tokens:
                spans = cls._labels_to_spans(labels)
                mask = cls._labels_to_mask(labels)
                image_path = cls._resolve_image_path(image_id, corpus_config)
                aux_payload = {}
                lookup_candidates = [str(image_id), str(guid)]
                for lookup_key in lookup_candidates:
                    if lookup_key and lookup_key in aux_mapping:
                        aux_payload = dict(aux_mapping[lookup_key])
                        break
                image_id = str(aux_payload.pop('image_id', image_id) or image_id or '')
                image_path = str(aux_payload.pop('image_path', image_path) or image_path or '')
                aux_payload.pop(aux_info_key, None)
                yield guid, cls._compose_record(
                    guid=guid,
                    tokens=tokens,
                    spans=spans,
                    mask=mask,
                    image_id=image_id,
                    image_path=image_path,
                    aux_payload=aux_payload,
                )

    @classmethod
    def _load_json_tags_file(cls, filepath, corpus_config):
        tags_key = corpus_config.get('tags_key', 'labels')
        text_key = corpus_config.get('text_key', 'text')
        tokenizer = corpus_config.get('tokenizer', 'char')
        with open(filepath, encoding='utf-8') as f:
            guid = 0
            for line in f:
                example = json.loads(line)
                image_id, image_path, aux_payload = cls._split_metadata(
                    example, reserved_keys={tags_key, text_key}
                )
                text = example[text_key]
                if isinstance(text, list):
                    tokens = text
                elif isinstance(text, str):
                    if tokenizer == 'char':
                        tokens = list(text)
                    elif tokenizer == 'blank':
                        tokens = text.split(' ')
                    else:
                        raise NotImplementedError
                else:
                    raise ValueError('Unsupported text input.')
                labels = example[tags_key]
                assert len(tokens) == len(labels)
                spans = cls._labels_to_spans(labels)
                mask = cls._labels_to_mask(labels)
                yield guid, cls._compose_record(
                    guid=guid,
                    tokens=tokens,
                    spans=spans,
                    mask=mask,
                    image_id=image_id,
                    image_path=image_path,
                    aux_payload=aux_payload,
                )
                guid += 1

    @classmethod
    def _load_json_spans_file(cls, filepath, corpus_config):
        # {'text': 'aaa', 'labels': [{'start': 0, 'end': 1, type: 'LOC'}, ...]}
        # {'tokens': ['a', 'aa', ...], 'spans': [{'start': 0, 'end': 1, type: 'LOC'}, ...]}
        spans_key = corpus_config.get('spans_key', 'spans')
        text_key = corpus_config.get('text_key', 'text')
        tokenizer = corpus_config.get('tokenizer', 'char')
        is_end_included = corpus_config.get('is_end_included', False)

        with open(filepath, encoding='utf-8') as f:
            guid = 0
            for line in f:
                if line.strip() == '':
                    continue
                example = json.loads(line)
                image_id, image_path, aux_payload = cls._split_metadata(
                    example, reserved_keys={spans_key, text_key}
                )
                text = example[text_key]
                if isinstance(text, list):
                    tokens = text
                elif isinstance(text, str):
                    if tokenizer == 'char':
                        tokens = list(text)
                    elif tokenizer == 'blank':
                        tokens = text.split(' ')
                    else:
                        raise RuntimeError
                else:
                    raise ValueError('Unsupported text input.')
                spans = []
                for span in example[spans_key]:
                    if is_end_included:
                        span['end'] += 1
                    if 'word' in span:
                        del span['word']
                    spans.append(span)
                mask = [True] * len(tokens)
                yield guid, cls._compose_record(
                    guid=guid,
                    tokens=tokens,
                    spans=spans,
                    mask=mask,
                    image_id=image_id,
                    image_path=image_path,
                    aux_payload=aux_payload,
                )
                guid += 1

    @classmethod
    def _load_cluener_file(cls, filepath, corpus_config):
        is_end_included = corpus_config.get('is_end_included', False)

        with open(filepath, encoding='utf-8') as f:
            guid = 0
            for line in f:
                example = json.loads(line)
                image_id, image_path, aux_payload = cls._split_metadata(
                    example, reserved_keys={'label', 'text'}
                )
                text = example['text']
                if isinstance(text, list):
                    tokens = text
                elif isinstance(text, str):
                    if corpus_config['tokenizer'] == 'char':
                        tokens = list(text)
                    elif corpus_config['tokenizer'] == 'blank':
                        tokens = text.split(' ')
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                entities = list()
                for entity_type, span_list in example['label'].items():
                    for name, span in span_list.items():
                        end_offset = int(is_end_included)
                        span = dict(start=span[0][0], end=span[0][1] + end_offset, type=entity_type)
                        entities.append(span)
                mask = [True] * len(tokens)
                yield guid, cls._compose_record(
                    guid=guid,
                    tokens=tokens,
                    spans=entities,
                    mask=mask,
                    image_id=image_id,
                    image_path=image_path,
                    aux_payload=aux_payload,
                )
                guid += 1

    @classmethod
    def _labels_to_spans(cls, labels):
        spans = []
        in_entity = False
        start = -1
        for i, tag in enumerate(labels):
            # fix label error
            if tag[0] in 'IE' and not in_entity:
                tag = 'B' + tag[1:]
            if tag[0] in 'BS':
                if i + 1 < len(labels) and labels[i + 1][0] in 'IE':
                    start = i
                else:
                    spans.append(dict(start=i, end=i + 1, type=tag[2:]))
            elif tag[0] in 'IE':
                if i + 1 >= len(labels) or labels[i + 1][0] not in 'IE':
                    assert start >= 0, 'Invalid label sequence found: {}'.format(labels)
                    spans.append(dict(start=start, end=i + 1, type=tag[2:]))
                    start = -1
            if tag[0] in 'B':
                in_entity = True
            elif tag[0] in 'OES':
                in_entity = False
        return spans

    @classmethod
    def _labels_to_mask(cls, labels):
        mask = []
        for label in labels:
            mask.append(label != PAD_LABEL)
        return mask
