from __future__ import absolute_import, division, print_function

import h5py as h5py
import numpy as np
import torch


def read_h5_file(h5_file, is_train):
    split = 'train' if is_train else 'test'
    with h5py.File(h5_file, 'r') as f:
        X = f[split]['x'][()]
        e = f[split]['e'][()].reshape(-1, 1)
        y = f[split]['t'][()].reshape(-1, 1)
    return X, e, y


class SurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, is_train):
        super().__init__()
        self.X, self.e, self.y = read_h5_file(h5_file, is_train)
        self._normalize()


    def _normalize(self):
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)

    def __getitem__(self, ix):
        return {
            'X': torch.from_numpy(self.X[ix]),
            'e': torch.from_numpy(self.e[ix]),
            'y': torch.from_numpy(self.y[ix])
        }

    def __len__(self):
        return len(self.X)

    @property
    def ndim(self):
        return self.X.shape[1]


class SurvivalDataset2(SurvivalDataset):
    def __init__(self, h5_file, is_train, interval=30):
        super().__init__(h5_file, is_train)
        self.s = self._get_hit_series(self.e, self.y, interval=interval)

    def _get_hit_series(self, e, y, interval):
        max_length = int(np.ceil(np.max(y) / interval))
        s = np.zeros([super().__len__(), max_length + 1], dtype=np.float32)

        time_steps = np.ceil(y / interval).astype(int)
        for i in range(len(s)):
            s[i, time_steps[i]] = e[i]
        return s

    def __getitem__(self, ix):
        data_dict = super().__getitem__(ix)
        return {
            'X': data_dict['X'],
            'e': data_dict['e'],
            'y': data_dict['y'],
            's': torch.from_numpy(self.s[ix])
        }

    @property
    def max_length(self):
        return self.s.shape[1]


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_ds = SurvivalDataset2("./data/whas/whas_train_test.h5", is_train=True)
    _ = train_ds[0]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    import time

    time_meter = []
    for i, batch in enumerate(train_loader, start=1):
        t0 = time.time()
        time_meter.append(time.time() - t0)

    print()
    print(f"Epoch : {np.sum(time_meter)} s, "
          f"Step Time: {np.mean(time_meter)} s/batch")
