#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""

import sys

sys.path.append(r'/mnt/DS_SHARED/users/talb/projects/3DSkeleton')
sys.path.append(r'/mnt/DS_SHARED/users/talb/projects/MeshUtils')
from matplotlib import pyplot as plt
from skeletons_3d.logger import NeptuneLogger, PrintLogger, CometLogger
import os
from os import path as osp
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn.metrics as metrics
from mesh_utils.utility.plot3d import plot_3d_point_cloud
from mesh_utils.utility.constants import DataPaths

from skeletons_3d.data.dataset import PointCloudDataset
from skeletons_3d.dgcnn import DGCNN
from skeletons_3d.layout import SkeletonLayout


class Trainer:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.train_loader = self.get_data_loader('train', self.args.batch_size, True)
        self.val_loader = self.get_data_loader('val', self.args.batch_size, True)
        self.test_loader = self.get_data_loader('test', self.args.batch_size, False)
        self.samples = next(iter(self.get_data_loader('test', 4, False)))
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.model = DGCNN(args, in_channels=args.n_in, output_channels=args.n_out).to(self.device)
        self.model = nn.DataParallel(self.model)
        self.layout = SkeletonLayout()
        if args.resume_from:
            tqdm.write(f'Continue training: {args.resume_from}')
            self.model.load_state_dict(torch.load(args.resume_from))
        else:
            tqdm.write('Model initialized.')
        if args.optimizer == 'sgd':
            tqdm.write("Use SGD")
            self.opt = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        else:
            tqdm.write("Use Adam")
            self.opt = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.opt, args.epochs, eta_min=args.lr)
        self.criterion = MSELoss()

    def get_data_loader(self, dtype, batch_size, train=False):
        file = osp.join(DataPaths.POINT_CLOUDS_DB, 'segmentation', f'{dtype}.h5')
        return DataLoader(PointCloudDataset(file, self.args.num_points, train=train), num_workers=8,
                          batch_size=batch_size, shuffle=train, drop_last=True)

    def calc_mse(self, y_true, y_pred):
        y_true = y_true.reshape((-1, 24, 3))
        y_pred = y_pred.reshape((-1, 24, 3))
        diff = y_true - y_pred
        pw_mse = np.square(diff).sum(axis=2).mean(axis=0)
        joints = list(self.layout.joints.keys())
        all_mse = np.mean(pw_mse)
        rmse_norm = np.sqrt(all_mse) / 2
        return rmse_norm, all_mse, list(zip(joints, pw_mse))

    def iterate(self, data_loader, train):
        self.model.train(train)
        cum_loss = 0.0
        n = len(data_loader)
        y_true, y_pred = [], []
        for data, label in tqdm(data_loader):
            data, label = data.to(self.device), label.to(self.device).squeeze()
            data = data.permute(0, 2, 1)
            if train:
                self.opt.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, label)
            if train:
                loss.backward()
                self.opt.step()
            cum_loss += loss.item()
            y_true.append(label.detach().cpu().numpy())
            y_pred.append(logits.detach().cpu().numpy())
        if train:
            self.scheduler.step()
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        return cum_loss / n, y_true, y_pred

    def evaluate(self, data_loader=None, train=False):
        if data_loader is None:
            data_loader = self.test_loader
        loss, y_true, y_pred = self.iterate(data_loader, train)
        rmse_norm, mse_all, mse_pw = self.calc_mse(y_true, y_pred)
        return rmse_norm, mse_all, mse_pw

    def train(self):
        tqdm.write('--- Training ---')
        tqdm.write(str(self.model))
        tqdm.write(f'Using {torch.cuda.device_count()} GPUs.')
        best_loss = np.inf
        train_losses, val_losses = [], []
        for epoch in range(self.args.epochs):
            rmse_norm, mse_all, mse_pw = self.evaluate(self.train_loader, True)
            # rmse_norm, mse_all, mse_pw = self.evaluate(self.val_loader, True)
            train_losses.append(mse_all)
            self.logger.log('Train RMSE', rmse_norm)
            self.logger.log('Train MSE', mse_all)
            tqdm.write(f'Train :: RMSE: {np.round(rmse_norm, 6)}, MSE: {np.round(mse_all, 6)}' +
                       '\n\t' + ','.join([f'{j}: {np.round(v, 6)}' for j, v in mse_pw]))
            rmse_norm, mse_all, mse_pw = self.evaluate(self.val_loader, False)
            val_losses.append(mse_all)
            self.logger.log('Val RMSE', rmse_norm)
            self.logger.log('Val MSE', mse_all)
            tqdm.write(f'Validation :: RMSE: {np.round(rmse_norm, 6)}, MSE: {np.round(mse_all, 6)}' +
                       '\n\t' + ','.join([f'{j}: {np.round(v, 6)}' for j, v in mse_pw]))

            batch_pred = self.inference(self.samples[0])
            for i, (x, y_true, y_pred) in enumerate(zip(*self.samples, batch_pred)):
                x = x.detach().cpu().numpy()
                pc, segments = x[:, :3], x[:, 3:]
                y_true = y_true.detach().cpu().numpy().reshape(24, 3)
                fig = plt.figure(figsize=(16, 8))
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                plot_3d_point_cloud(pc=pc, skeleton=y_true, segments=None, show=False, title='Ground Truth', axis=ax1, skeleton_layout=self.layout)
                plot_3d_point_cloud(pc=pc, skeleton=y_pred, segments=None, show=False, title='Reconstruct', axis=ax2, skeleton_layout=self.layout)
                self.logger.log_fig(f'Sample {i}', fig)

            if mse_all < best_loss:
                tqdm.write(f'New best validation: {epoch} ({mse_all} < {best_loss})')
                self.logger.log('New Best', epoch)
                best_loss = mse_all
                torch.save(self.model.state_dict(), f'checkpoints/{self.args.exp_name}/models/model.t7')

        rmse_norm, mse_all, mse_pw = self.evaluate(self.test_loader, False)
        joints, mse_pw = zip(*mse_pw)

        fig, ax = plt.subplots()
        ax.plot(range(args.epochs), train_losses)
        ax.plot(range(args.epochs), val_losses)
        logger.log_fig('training_loss', fig)

        fig, ax = plt.subplots()
        sns.barplot(x=['all'] + mse_pw[0::2], y=np.concatenate(([mse_all], list(mse_pw[1::2]))), ax=ax)
        ax.set_xticks(rotation=70)
        fig.suptitle(f'MSE={np.round(mse_all, 5)}, RMSE (Scaled)={np.round(rmse_norm, 5)}%')
        fig.tight_layout()
        logger.log_fig('test mse', fig)

    def inference(self, data_batch):
        data_batch = data_batch.to(self.device)
        data_batch = data_batch.permute(0, 2, 1)
        logits = self.model(data_batch)
        y_pred = logits.detach().cpu().numpy().reshape((-1, 24, 3))
        return y_pred


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    # os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    # os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    # os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    # os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

logger_init = {
    'neptune': lambda: NeptuneLogger(project_name='talbarami/Skeleton3D',
                                     api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYzg2M2UyOS1jNDIxLTQwZjctYTc3Yi1iMmY0ZGU0MzljZjUifQ=='),
    'comet': lambda: CometLogger(project_name='3d-skeleton',
                                 api_key='9Rrf3dI3oglv3rFEwtTCYuiej',
                                 workspace='talbarami'),
    'print': lambda: PrintLogger()
}

if __name__ == "__main__":
    logger_type = 'comet'
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Optimizer (sgd/adam)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume_from', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    args.n_out = 24 * 3
    args.n_in = 3 + 14

    _init_()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    with logger_init[logger_type]() as logger:
        t = Trainer(args, logger)
        if args.eval:
            t.inference(t.test_loader)
        else:
            t.train()
    # with PrintLogger() as logger:
    #     t = Trainer(args, logger)
    #     if args.eval:
    #         t.inference(t.test_loader)
    #     else:
    #         t.train()
