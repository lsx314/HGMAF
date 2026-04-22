import importlib.util
import inspect
import logging
import os.path as osp
import sys
from functools import partial
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DownloadManager
from datasets.utils.file_utils import is_remote_url

from adaseq.metainfo import Tasks, get_member_set

from .utils import COUNT_LABEL_FUNCTIONS, DATASET_TRANSFORMS

BUILTIN_TASKS = get_member_set(Tasks)
logger = logging.getLogger(__name__)


def _load_dataset_from_script(script_path, name=None, **kwargs):
    import datasets as hf_datasets

    builders_dir = osp.dirname(script_path)
    pkg_root = osp.dirname(builders_dir)

    pkg_name = 'adaseq.data.dataset_builders'
    if pkg_name not in sys.modules:
        _ensure_package_in_sys_modules('adaseq', osp.dirname(osp.dirname(pkg_root)))
        _ensure_package_in_sys_modules('adaseq.data', pkg_root)
        init_path = osp.join(builders_dir, '__init__.py')
        _load_module_to_sys(pkg_name, init_path)

    base_mod_name = pkg_name + '.base'
    if base_mod_name not in sys.modules:
        _load_module_to_sys(base_mod_name, osp.join(builders_dir, 'base.py'))

    script_mod_name = pkg_name + '.' + osp.basename(script_path)[:-3]
    _load_module_to_sys(script_mod_name, script_path)
    module = sys.modules[script_mod_name]

    builder_cls = None
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            obj.__module__ == script_mod_name
            and issubclass(obj, hf_datasets.GeneratorBasedBuilder)
            and obj is not hf_datasets.GeneratorBasedBuilder
            and not inspect.isabstract(obj)
        ):
            builder_cls = obj
            break

    if builder_cls is None:
        raise RuntimeError(f'No valid DatasetBuilder found in {script_path}')

    builder_kwargs = {k: v for k, v in kwargs.items() if k in ('data_dir', 'data_files')}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in ('data_dir', 'data_files')}
    builder = builder_cls(
        cache_dir=None,
        name=name or 'default',
        **builder_kwargs,
        **extra_kwargs,
    )

    builder.download_and_prepare()
    hf_ds = builder.as_dataset()
    if isinstance(hf_ds, hf_datasets.Dataset):
        return {'train': hf_ds}
    return dict(hf_ds)


def _ensure_package_in_sys_modules(pkg_name, pkg_dir):
    if pkg_name not in sys.modules:
        try:
            importlib.import_module(pkg_name)
        except ImportError:
            init_path = osp.join(pkg_dir, '__init__.py')
            _load_module_to_sys(pkg_name, init_path if osp.exists(init_path) else None)


def _load_module_to_sys(mod_name, file_path):
    if file_path and osp.exists(file_path):
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
    else:
        spec = importlib.util.spec_from_loader(mod_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = mod_name.rsplit('.', 1)[0] if '.' in mod_name else mod_name
    sys.modules[mod_name] = module
    if spec.loader:
        spec.loader.exec_module(module)


class DatasetManager:
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        labels: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    ) -> None:
        self.datasets = datasets

        if labels is None:
            labels = None
        elif isinstance(labels, list):
            pass
        elif isinstance(labels, str):
            if is_remote_url(labels):
                labels = DownloadManager().download(labels)
                labels = [line.strip() for line in open(labels)]
        elif isinstance(labels, dict):
            labels = labels.copy()
            label_set = set()
            counter_type = labels.pop('type')
            func = COUNT_LABEL_FUNCTIONS[counter_type]
            counter = partial(func, labels=label_set, **labels)
            kwargs = dict(desc=f'Counting labels by {counter_type}', load_from_cache_file=False)
            datasets['train'].map(counter, **kwargs)
            if 'valid' in datasets:
                datasets['valid'].map(counter, **kwargs)
            labels = sorted(label_set)
        else:
            raise ValueError(f'Unsupported labels: {labels}')
        self.labels = labels

    @property
    def train(self):
        return self.datasets.get('train', None)

    @property
    def dev(self):
        return self.datasets.get('valid', None)

    @property
    def valid(self):
        return self.datasets.get('valid', None)

    @property
    def test(self):
        return self.datasets.get('test', None)

    @classmethod
    def from_config(
        cls,
        task: Optional[str] = None,
        data_file: Optional[Union[str, Dict[str, str]]] = None,
        path: Optional[str] = None,
        name: Optional[str] = None,
        access_token: Optional[str] = None,
        labels: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        transform: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> 'DatasetManager':
        if data_file is not None:
            if isinstance(data_file, str):
                if not is_remote_url(data_file) and not osp.exists(data_file):
                    raise RuntimeError('`data_file` not exists: %s', data_file)
                kwargs.update(data_dir=data_file)
            elif isinstance(data_file, dict):
                for k, v in data_file.items():
                    if is_remote_url(v):
                        continue
                    if not osp.exists(v):
                        raise RuntimeError('`data_file[%s]` not exists: %s', k, v)
                    if not osp.isabs(v):
                        raise RuntimeError('`data_file[%s]` must be a absolute path: %s', k, v)

                if 'dev' in data_file:
                    data_file['valid'] = data_file.pop('dev')
                kwargs.update(data_files=data_file)
            else:
                raise ValueError(f'Unsupported data_file: {data_file}')

            assert task is not None and task in BUILTIN_TASKS, 'Need a specific task!'
            task_builder_name = task
            if task in {Tasks.word_segmentation, Tasks.part_of_speech}:
                task_builder_name = Tasks.named_entity_recognition

            if task == Tasks.entity_typing and 'cand' in data_file:
                task_builder_name = 'mcce-entity-typing'

            path = osp.join(
                osp.dirname(osp.abspath(__file__)),
                'dataset_builders',
                task_builder_name.replace('-', '_') + '_dataset_builder.py',
            )

        if isinstance(path, str):
            if path.endswith('.py') or osp.isdir(path):
                logger.info('Will use a custom loading script: %s', path)

            if name is not None:
                logger.info("Passing `name='%s'` to `datasets.load_dataset`", name)

            if path.endswith('.py') and osp.isfile(path):
                datasets = _load_dataset_from_script(path, name=name, **kwargs)
            else:
                from datasets import load_dataset as hf_load_dataset
                hfdataset = hf_load_dataset(path, name=name, **kwargs)
                datasets = {k: v for k, v in hfdataset.items()}

        elif isinstance(name, str):
            if access_token is not None:
                from modelscope.hub.api import HubApi

                HubApi().login(access_token)

            from modelscope.msdatasets import MsDataset
            msdataset = MsDataset.load(name, **kwargs)
            datasets = {k: v._hf_ds for k, v in msdataset.items()}

        else:
            raise RuntimeError('Unsupported dataset!')

        if 'dev' in datasets and 'valid' not in datasets:
            datasets['valid'] = datasets.pop('dev')

        if 'test' in datasets and 'valid' not in datasets:
            datasets['valid'] = datasets['test']
            logger.warning('Validation set not found. Reuse test set for validation!')

        if 'valid' in datasets and 'test' not in datasets:
            datasets['test'] = datasets['valid']
            logger.warning('Test set not found. Reuse validation set for testing!')

        if 'train' not in datasets:
            logger.warning('Training set not found!')

        if 'valid' not in datasets:
            logger.warning('Validation set not found!')

        if 'test' not in datasets:
            logger.warning('Test set not found!')

        if transform:
            datasets = apply_transform(datasets, **transform)

        if labels is None:
            if task in {
                Tasks.named_entity_recognition,
                Tasks.word_segmentation,
                Tasks.part_of_speech,
                Tasks.entity_typing,
            }:
                labels = dict(type='count_span_labels')
            elif task == Tasks.relation_extraction:
                labels = dict(type='count_labels')

        if 'train' in datasets:
            logger.info('First sample in train set: ' + str(datasets['train'][0]))

        return cls(datasets, labels)


def apply_transform(
    datasets: Dict[str, Dataset],
    name: str,
    **kwargs,
) -> Dict[str, Dataset]:
    if name not in DATASET_TRANSFORMS:
        raise RuntimeError(f'{name} not in {DATASET_TRANSFORMS.keys()}')
    kwargs = kwargs or dict()

    for k in datasets.keys():
        datasets[k] = DATASET_TRANSFORMS[name](datasets[k], **kwargs)

    return datasets
