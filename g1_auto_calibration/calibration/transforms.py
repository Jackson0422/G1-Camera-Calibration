import numpy as np
from scipy.spatial.transform import Rotation


def transform(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t).reshape(3)
    return T


def checked(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Expected finite 4x4 transform')
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7) or not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or abs(np.linalg.det(R)-1) > 1e-5:
        raise ValueError('Invalid SE(3) transform')
    return T


def inverse(T):
    T = checked(T)
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def mean_pose(poses):
    return transform(Rotation.from_matrix(np.array(poses)[:, :3, :3]).mean().as_matrix(),
                     np.mean(np.array(poses)[:, :3, 3], axis=0))


def pose_errors(reference, poses):
    delta = np.array([inverse(reference) @ p for p in poses])
    return (np.linalg.norm(delta[:, :3, 3], axis=1)*1000,
            np.rad2deg(Rotation.from_matrix(delta[:, :3, :3]).magnitude()))


def describe(T):
    T = checked(T)
    r = Rotation.from_matrix(T[:3, :3])
    return dict(matrix=T.tolist(), translation_m=T[:3, 3].tolist(),
                quaternion_xyzw=r.as_quat().tolist(), euler_xyz_rad=r.as_euler('xyz').tolist(),
                convention='T_parent_child maps child coordinates into parent; column vectors; metres; optical x right y down z forward')
