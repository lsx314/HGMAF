from collections.abc import Mapping
import torch

class DataBatch(Mapping):
    def __init__(self, batch, keep_fields=None):
        self.keep_fields = keep_fields or {}
        if len(self.keep_fields) == 0:
            if isinstance(batch, tuple):
                batch = dict(batch)
            self.batch = batch
        else:
            self.batch = self.tensorize(batch)

    def __repr__(self):
        return str(self.batch)

    def __getitem__(self, key):
        return self.batch.get(key, None)

    def __contains__(self, key):
        return key in self.batch

    def __iter__(self):
        return iter(self.batch)

    def __len__(self):
        return len(self.batch)

    def tensorize(self, batch):
        if isinstance(batch, tuple):
            return dict(batch)
        for k in batch.keys():
            if k.endswith('tokens'):
                for sub in batch[k].keys():
                    dtype = torch.bool if sub.endswith('mask') else torch.long
                    batch[k][sub] = torch.tensor(batch[k][sub], dtype=dtype)
            elif k in self.keep_fields:
                continue
            elif torch.is_tensor(batch[k]):
                continue
            elif (
                isinstance(batch[k], list)
                and len(batch[k]) > 0
                and all(torch.is_tensor(item) for item in batch[k])
            ):
                batch[k] = torch.stack(batch[k], dim=0)
            else:
                batch[k] = torch.tensor(batch[k])
        return batch

    def to(self, device):
        for k in self.batch.keys():
            if k.endswith('tokens'):
                for sub in self.batch[k].keys():
                    self.batch[k][sub] = self.batch[k][sub].to(device)
            elif k in self.keep_fields:
                continue
            else:
                self.batch[k] = self.batch[k].to(device)
