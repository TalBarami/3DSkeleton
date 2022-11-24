import pickle
from os import path as osp

import h5py
import numpy as np
import torch
from tqdm import tqdm
from utility.constants import DataPaths
from utility.utils import init_directories, write_pkl

from skeletons_3d.layout import SkeletonLayout

if __name__ == '__main__':
    data = ['train', 'val', 'test']
    checkpoints_dir = r'/mnt/DS_SHARED/users/talb/projects/3d_avatar_generation/checkpoints'
    out_root = r'/mnt/DS_SHARED/users/talb/data/separate_body_parts'
    layout = SkeletonLayout()
    parts_names = {v: k for k,v in layout._parts.items()}
    body_parts = {k: set([j for tup in v for j in tup]) for k,v in layout._parts_segmentation.items()}
    init_directories(*[osp.join(out_root, d) for d in data])
    for d in data:
        out_dir = osp.join(out_root, d)
        init_directories(out_dir)
        with h5py.File(osp.join(DataPaths.POINT_CLOUDS_DB, 'segmentation', f'{d}.h5'), "r") as f:
            pcs = f['point_cloud']
            skeletons = f['skeleton']
            segmentations = f['segmentations']
            for body_part, joints in body_parts.items():
                part_name = parts_names[body_part]
                out = {k: [] for k in ['x', 'y']}
                for pc, skeleton, segments in list(zip(pcs, skeletons, segmentations)):
                    x = pc[segments == body_part]
                    y = {j: skeleton[j] for j in joints}
                    out['x'].append(x)
                    out['y'].append(y)
                write_pkl(out, osp.join(out_dir, f'{part_name}.pkl'))
