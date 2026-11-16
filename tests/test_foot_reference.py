"""Fixed-foot FK must include leg-induced pelvis motion and match MuJoCo."""

from pathlib import Path
import unittest

import numpy as np

from g1_auto_calibration.common import ROOT, load_yaml
from g1_auto_calibration.calibration.forward_kinematics import ForwardKinematics
from g1_auto_calibration.calibration.foot_reference import FixedStanceMonitor
from g1_auto_calibration.calibration.transforms import inverse, transform

MUJOCO_MODEL = Path('/home/kaixin/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml')


class FootReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_yaml(ROOT / 'config/robot.yaml')
        cls.urdf = ROOT / 'config/g1_29dof.urdf'
        cls.indices = cls.config['joint_indices']
        cls.foot = ForwardKinematics(cls.urdf, 'left_ankle_roll_link', 'torso_link', cls.indices)

    def test_relative_transform_matches_mujoco_with_nonzero_legs_and_waist(self):
        if not MUJOCO_MODEL.exists():
            self.skipTest('Optional local MuJoCo model is unavailable')
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(MUJOCO_MODEL))
        data = mujoco.MjData(model)
        names = [name for name, _ in sorted(self.indices.items(), key=lambda item: item[1])]
        addresses = model.jnt_qposadr[[model.joint(name).id for name in names]]
        rng = np.random.default_rng(112)
        for _ in range(12):
            q = np.zeros(29)
            q[:15] = rng.uniform(-.05, .05, 15)
            q[[3, 9]] = rng.uniform(.1, .25, 2)
            data.qpos[:] = model.qpos0
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            def world(link):
                body = model.body(link).id
                return transform(data.xmat[body].reshape(3, 3), data.xpos[body])
            expected = inverse(world('left_ankle_roll_link')) @ world('torso_link')
            np.testing.assert_allclose(self.foot.compute(q), expected, atol=1e-6)

    def test_fixed_foot_composition_compensates_leg_induced_pelvis_motion(self):
        root_to_foot = ForwardKinematics(self.urdf, 'pelvis', 'left_ankle_roll_link', self.indices)
        root_to_mount = ForwardKinematics(self.urdf, 'pelvis', 'torso_link', self.indices)
        q = np.zeros(29)
        q[[0, 3, 4, 12, 13, 14]] = [-.2, .4, -.2, .1, -.05, .08]
        world_pelvis = inverse(root_to_foot.compute(q))
        np.testing.assert_allclose(world_pelvis @ root_to_mount.compute(q), self.foot.compute(q), atol=1e-12)
        self.assertGreater(np.linalg.norm(self.foot.compute(q)[:3, 3] - self.foot.compute(np.zeros(29))[:3, 3]), .01)
        self.assertTrue(any(name.startswith('inverse:left_') for name in self.foot.chain_names))

    def test_relative_stance_monitor_accepts_waist_and_rejects_foot_rotation(self):
        monitor = FixedStanceMonitor(self.urdf, self.indices, np.zeros(29))
        q = np.zeros(29)
        q[12:15] = [.2, -.1, .1]
        result = monitor.check(q)
        self.assertTrue(result['passed'])
        self.assertFalse(result['common_foot_slip_observable'])
        q[5] = .04
        self.assertFalse(monitor.check(q)['passed'])

    def test_full_state_window_batch_matches_individual_relative_transforms(self):
        rng = np.random.default_rng(113)
        q = np.zeros((1200, 29))
        q[:, :15] = rng.uniform(-.03, .03, (1200, 15))
        q[:, [3, 9]] = .1
        actual = self.foot.compute_many(q)
        for index in (0, 129, 589, 1199):
            np.testing.assert_allclose(actual[index], self.foot.compute(q[index]), atol=1e-12)
        monitor = FixedStanceMonitor(self.urdf, self.indices, q[0])
        checks = monitor.check_many(q)
        self.assertEqual(len(checks), 1200)
        self.assertTrue(checks[0]['passed'])
        with self.assertRaises(ValueError):
            self.foot.compute_many(np.full((2, 29), np.nan))


if __name__ == '__main__':
    unittest.main()
