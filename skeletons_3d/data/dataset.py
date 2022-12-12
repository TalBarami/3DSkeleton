from os import path as osp

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mesh_utils.utility.utils import read_pkl, farthest_point_sample


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class PartDataset(Dataset):
    def __init__(self, pkl_path, num_pts, train=False):
        self._data = read_pkl(pkl_path)
        self.X, self.Y = self._data['x'], self._data['y']
        self.num_pts = num_pts
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, ids = farthest_point_sample(self.X[idx], self.num_pts)
        y = np.array([v for v in self.Y[idx].values()]).reshape(-1)
        # ids = np.random.choice(len(self.X[idx]), self.num_pts, replace=True)
        # x, y = self.X[idx][ids], np.array([v for v in self.Y[idx].values()]).reshape(-1)
        if self.train:
            # x = translate_pointcloud(x)
            np.random.shuffle(x)
        return x, y

class PointCloudDataset(Dataset):
    def __init__(self, h5_path, num_pts, train=False):
        with h5py.File(h5_path, "r") as f:
            self.X = np.array(f['point_cloud'])
            self.Y = np.array(f['skeleton'])
            self.s = np.array(f['segmentations'])
        self.num_pts = num_pts
        self.train = train

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, ids = farthest_point_sample(self.X[idx], self.num_pts)
        s = self.s[idx][ids].reshape(-1, 1)
        h = np.zeros((s.size, s.max() + 1))
        h[np.arange(s.size), s] = 1
        y = self.Y[idx].reshape(-1)
        x = np.concatenate((x, h), axis=1)
        # ids = np.random.choice(len(self.X[idx]), self.num_pts, replace=True)
        # x, y, s = self.X[idx][ids], self.Y[idx].reshape(-1), self.s[idx][ids]
        if self.train:
            # x = translate_pointcloud(x)
            np.random.shuffle(x)
        return torch.tensor(x, dtype=torch.float), y

if __name__ == '__main__':
    data = ['train', 'val', 'test']
    part = 'R_Foot.pkl'

    pkl_path = r'/mnt/DS_SHARED/users/talb/data/separate_body_parts'
    gen = lambda s, t: PointCloudDataset(osp.join(pkl_path, s, part), 200, train=t)
    train, val, test = gen('train', True), gen('val', True), gen('test', False)
    x, y = data[0]
    print(1)
