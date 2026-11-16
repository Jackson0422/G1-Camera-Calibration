"""Sampling motion budgets expressed in mount coordinates, using actual URDF FK.

These budgets come from the operator's calibration accuracy requirements, not
from motor velocity limits. They bound mount motion relative to the configured
fixed reference; they do not measure camera blur, foot slip or absolute accuracy.
"""

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..common import ROOT
from .forward_kinematics import ForwardKinematics


def policy(robot):
    value = robot.get('sampling_stability')
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get('method') != 'urdf_mount_pose':
        raise ValueError('Sampling stability requires the urdf_mount_pose method')
    if not isinstance(value.get('criteria_source'), str) or not value['criteria_source'].strip():
        raise ValueError('Sampling stability requires an explicit accuracy criteria source')
    for key in ('translation_limit_m', 'rotation_limit_deg'):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) or number <= 0:
            raise ValueError(f'Invalid sampling stability {key}')
    return value


@lru_cache(maxsize=8)
def _fk(path, stamp, base, mount, indices):
    return ForwardKinematics(path, base, mount, dict(indices))


def model(robot):
    path = Path(robot['urdf'])
    if not path.is_absolute():
        path = ROOT / 'config' / path
    path = path.resolve(strict=True)
    return _fk(str(path), path.stat().st_mtime_ns, robot['base_frame'], robot['mount_frame'],
               tuple(sorted(robot['joint_indices'].items())))


def measure(q, robot):
    """Conservative pairwise pose-diameter bounds over the complete state window.

    The translation bounding-box diagonal bounds all pairwise distances. Twice
    the largest rotation from the first pose bounds pairwise geodesic angles by
    the triangle inequality. No velocity reading is reinterpreted as zero.
    """
    criteria = policy(robot)
    if criteria is None:
        raise ValueError('No mount-pose sampling policy is configured')
    q = np.asarray(q, float)
    if q.ndim != 2 or q.shape[1:] != (29,) or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('Mount stability requires at least two finite 29-joint states')
    fk = model(robot)
    transforms = fk.compute_many(q)
    translation = float(np.linalg.norm(np.ptp(transforms[:, :3, 3], axis=0)))
    relative = transforms[0, :3, :3].T @ transforms[:, :3, :3]
    rotation = float(min(180., 2 * np.rad2deg(np.max(Rotation.from_matrix(relative).magnitude()))))
    return dict(method='urdf_mount_pose', passed=bool(
        translation <= criteria['translation_limit_m'] and rotation <= criteria['rotation_limit_deg']),
        translation_diameter_bound_m=translation, rotation_diameter_bound_deg=rotation,
        translation_limit_m=float(criteria['translation_limit_m']),
        rotation_limit_deg=float(criteria['rotation_limit_deg']),
        criteria_source=criteria['criteria_source'], checked_state_count=len(q),
        urdf_sha256=fk.sha256, base_frame=fk.base, mount_frame=fk.mount,
        bounds_method='translation bounding-box diagonal; rotation triangle-inequality diameter',
        raw_joint_speed_used_as_stability_gate=False,
        actual_camera_motion_measured=False, actual_fixed_base_verified=False)


def verify_record(value, robot):
    """Validate recorded bounded-window evidence without relabeling old samples."""
    criteria = policy(robot)
    if criteria is None or not isinstance(value, Mapping):
        raise ValueError('Missing mount-pose stability evidence')
    fk = model(robot)
    if (value.get('method') != 'urdf_mount_pose' or value.get('passed') is not True
            or value.get('urdf_sha256') != fk.sha256 or value.get('base_frame') != fk.base
            or value.get('mount_frame') != fk.mount
            or value.get('criteria_source') != criteria['criteria_source']
            or type(value.get('checked_state_count')) is not int or value['checked_state_count'] < 2):
        raise ValueError('Mount-pose stability evidence differs from its model or policy')
    for measured, limit in (('translation_diameter_bound_m', 'translation_limit_m'),
                            ('rotation_diameter_bound_deg', 'rotation_limit_deg')):
        number = value.get(measured)
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not np.isfinite(number) or number < 0 or number > criteria[limit]
                or value.get(limit) != criteria[limit]):
            raise ValueError('Mount motion exceeds its recorded calibration accuracy budget')
