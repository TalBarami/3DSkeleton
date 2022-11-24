import h5py
import numpy as np
from os import path as osp

import trimesh
from matplotlib import pyplot as plt
import seaborn as sns

from mesh_segmentation.training.inference import Segmentor
from tqdm import tqdm

from skeletons.skeleton_layout import SkeletonLayout
from utility.constants import DataPaths
from utility.plot3d import plot_3d_point_cloud


class Reconstructor:
    def __init__(self, layout):
        self.layout = layout

    def _reconstruct_limbs(self, mesh, segments, out):
        for j, jid in self.layout.limb_joints().items():
            p1, p2 = self.layout.parts(jid)
            i1, i2 = np.where(np.isin(segments, p1))[0], np.where(np.isin(segments, p2))[0]
            n1, n2 = set(item for sublist in [mesh.vertex_neighbors[p] for p in i1] for item in sublist), set(item for sublist in [mesh.vertex_neighbors[p] for p in i2] for item in sublist)
            intersection = list(n1.intersection(n2))
            pts = np.array([mesh.vertices[x] for x in intersection])
            chosen = np.mean(pts, axis=0)
            out[jid] = chosen

    def _reconstruct_edges(self, mesh, segments, out):
        scalers = {
            'Head': 0.8,
            'Foot': 1.2,
            'Hand': 0.75
        }
        for j, jid in self.layout.edge_joints().items():
            p = self.layout.parts(jid)
            i = np.where(np.isin(segments, p))[0]
            pts = mesh.vertices[i]
            chosen = np.mean(pts, axis=0)
            s = out[self.layout.neighbors(jid)[0]]
            v = chosen - s
            p = scalers[j.split('_')[1] if '_' in j else j]
            out[jid] = s + v * p

    def _reconstruct_body(self, mesh, segments, out, epsilon = 0.025):
        pe, ne, lh, rh = self.layout.joints['Pelvis'], self.layout.joints['Neck'], self.layout.joints['L_Hip'], self.layout.joints['R_Hip']
        sp1, sp2, sp3 = [self.layout.joints[f'Spine{i}'] for i in [1, 2, 3]]
        out[pe] = np.average(out[[ne, lh, rh]], weights=[0.2, 1, 1], axis=0)
        def plane_intersect(p, v):
            d = -spine.dot(p)
            intersect = pts[np.abs(pts.dot(v) + d) < epsilon]
            return np.mean(intersect, axis=0)

        spine = out[ne] - out[pe]
        s = out[pe]
        pts = mesh.vertices[segments == self.layout._parts['Body']]

        out[sp2] = plane_intersect(s + spine * 0.5, spine)
        # p = s + spine * 0.5
        # d = -spine.dot(p)
        # intersect = pts[np.abs(pts.dot(spine) + d) < epsilon]
        # out[sp2] = np.mean(intersect, axis=0)

        # known = out[[pe, sp2, ne]].T
        # curve = bezier.Curve(known, degree=2)
        # points_fine = curve.evaluate_multi(np.linspace(0, 1, 16)).T
        # out[sp3] = points_fine[10]

        p = 0.76
        out[sp3] = p * out[sp2] + (1 - p) * out[ne]

        p = 0.41
        # out[sp1] = p * out[pe] + (1-p) * out[sp2]
        out[sp1] = plane_intersect(p * out[pe] + (1-p) * out[sp2], spine)

        # m = np.average([out[ne], out[sp3]], weights=[0.4, 0.6], axis=0)
        m = np.mean([out[ne], out[sp3]], axis=0)
        for s in ['L', 'R']:
            out[self.layout.joints[f'{s}_Collar']] = np.average([m, out[self.layout.joints[f'{s}_Shoulder']]], weights=[0.6, 0.4], axis=0)


    def reconstruct(self, mesh, segments):
        out = np.zeros([24, 3])
        self._reconstruct_limbs(mesh, segments, out)
        self._reconstruct_edges(mesh, segments, out)
        self._reconstruct_body(mesh, segments, out)

        return out
        # for i in range(1, 4):
        #     j = self.layout.joints[f'Spine{i}']
        #     frac = i / 4
        #     p = s + spine * frac
        #     d = -spine.dot(p)
        #     intersect = pts[np.abs(pts.dot(spine) + d) < epsilon]
        #
        #     out[j] = np.mean(intersect, axis=0)

        # for i in range(1, 4):
        #     j = self.layout.joints[f'Spine{i}']
        #     p = i / 4
        #     out[j] = p * out[ne] + (1-p) * out[pe]


        # for limb in set(self.layout.limbs().values()):
        #     i = np.where(segments == limb)[0]
        #     print(limb, i.shape[0])
        #     pts = np.array([mesh.vertices[x] for x in i])
        #     m = pts.mean(axis=0)
        #     uu, dd, vv = np.linalg.svd(pts - m)
        #     linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
        #     linepts += m
        #
        #     candidate_pts, distances = list(zip(*[self.closest_point_and_distance(p, linepts[0], linepts[1]) for p in pts]))
        #     import matplotlib.pyplot as plt
        #     import mpl_toolkits.mplot3d as m3d
        #
        #     ax = m3d.Axes3D(plt.figure())
        #     ax.scatter3D(*pts.T)
        #     ax.plot3D(*linepts.T, c=(1, 0, 0))
        #     plt.show()

def plot_spine(pts, a, b, frac):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts.T[0], pts.T[1], pts.T[2], marker='o')
    spine = b - a
    ax.plot(a, b)
    xx, yy = np.meshgrid(range(-2, 2), range(-2, 2))
    d = -(frac * spine).dot(spine)
    z = (-spine[0] * xx - spine[1] * yy - d) * 1. / spine[2]
    ax.plot_surface(xx, yy, z)


def plot_results(pc, org_segments, predicted_segments, skeleton, reconstructed_skeleton):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(221, projection='3d')
    plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], org_segments, axis=ax, show=False, title='Manual Segments')
    ax = fig.add_subplot(222, projection='3d')
    plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], predicted_segments, axis=ax, show=False, title='Predicted Segments')
    ax = fig.add_subplot(223, projection='3d')
    plot_3d_point_cloud(skeleton.T[0], skeleton.T[1], skeleton.T[2], axis=ax, show=False, title='Ground Truth Skeleton')
    ax = fig.add_subplot(224, projection='3d')
    plot_3d_point_cloud(reconstructed_skeleton.T[0], reconstructed_skeleton.T[1], reconstructed_skeleton.T[2], axis=ax, show=True, title='Skeleton Reconstruction')


if __name__ == '__main__':
    data = 'test'
    checkpoints_dir = r'/mnt/DS_SHARED/users/talb/projects/3d_avatar_generation/checkpoints'
    s = Segmentor(checkpoints_dir)
    r = Reconstructor(SkeletonLayout())
    with h5py.File(osp.join(DataPaths.POINT_CLOUDS_DB, 'segmentation', f'{data}.h5'), "r") as f:
        pcs = np.array(f['point_cloud'])
        faces = np.array(f['faces'])
        skeletons = np.array(f['skeleton'])
        segments = np.array(f['segmentations'])
    n = len(pcs)
    y_pred = []
    y_true = []
    nans = []
    for i, (pc, face, skeleton, org_segments) in tqdm(enumerate(zip(pcs, faces, skeletons, segments))):
        mesh = trimesh.Trimesh(vertices=pc, faces=faces)
        predicted_segments = s.predict(mesh)
        reconstructed_skeleton = r.reconstruct(mesh, predicted_segments)
        if np.any(np.isnan(reconstructed_skeleton)):
            nans.append(i)
        else:
            y_pred.append(reconstructed_skeleton)
            y_true.append(skeleton)
        if (i+1) % 100 == 0:
            plot_results(pc, org_segments, predicted_segments, skeleton, reconstructed_skeleton)
            # break
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
    # diff = np.nan_to_num(y_true - y_pred, nan=2)
    pw_mse = np.square(diff).sum(axis=2).mean(axis=0)
    all_mse = np.mean(pw_mse)
    rmse_norm = np.sqrt(all_mse) * 100 / 2
    joints = ['all'] + list(SkeletonLayout().joints.keys())
    sns.barplot(x=joints, y=np.concatenate(([all_mse], pw_mse)))
    plt.xticks(rotation=70)
    plt.title(f'MSE={np.round(all_mse, 5)}, RMSE (Scaled)={np.round(rmse_norm, 5)}%')
    plt.tight_layout()
    plt.show()

        # sns.barplot(x=joints, y=np.concatenate(([all_mse], pw_mse)) * 100 / 2)
        # plt.title(pt_node)
        # plt.xticks(rotation=70)
        # plt.tight_layout()
        # plt.show()
