"""Monitor a confirmed fixed stance without claiming to detect common foot slip."""

import numpy as np
from scipy.spatial.transform import Rotation

from .forward_kinematics import ForwardKinematics
from .transforms import inverse


class FixedStanceMonitor:
    """Reject relative foot displacement; external fixed-foot evidence is required."""

    def __init__(self, urdf, joint_indices, reference_q, *, translation_limit_m=.005,
                 rotation_limit_deg=1.0):
        self.fk = ForwardKinematics(urdf, 'left_ankle_roll_link',
                                    'right_ankle_roll_link', joint_indices)
        self.reference = self.fk.compute(reference_q)
        self.translation_limit_m = float(translation_limit_m)
        self.rotation_limit_deg = float(rotation_limit_deg)
        if not (0 < self.translation_limit_m <= .02 and 0 < self.rotation_limit_deg <= 5):
            raise ValueError('Fixed stance monitor requires finite, conservative positive limits')

    def check(self, q):
        return self.check_many(np.asarray(q)[None])[0]

    def check_many(self, q):
        relative = self.fk.compute_many(q)
        change = inverse(self.reference) @ relative
        translation = np.linalg.norm(change[:, :3, 3], axis=1)
        rotation = (np.rad2deg(Rotation.from_matrix(change[:, :3, :3]).magnitude())
                    if len(change) else np.empty(0))
        return [{
            'passed': bool(t <= self.translation_limit_m and r <= self.rotation_limit_deg),
            'relative_foot_translation_change_m': float(t),
            'relative_foot_rotation_change_deg': float(r),
            'translation_limit_m': self.translation_limit_m,
            'rotation_limit_deg': self.rotation_limit_deg,
            'common_foot_slip_observable': False,
        } for t, r in zip(translation, rotation)]
