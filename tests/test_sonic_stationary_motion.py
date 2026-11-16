import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_auto_calibration.common import ROOT, load_yaml
from g1_auto_calibration.verification import sonic_stationary_motion as motion


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('Motion design opened a socket'))
    names = [n for n, i in sorted(load_yaml(ROOT/'config/robot_sonic.yaml')['joint_indices'].items(), key=lambda t: t[1])]
    profile = dict(default_angles_hw=[-.3, .03, 0., .6, -.3, -.03]*2+[0.]*17,
        lower_hw=[-2.]*29, upper_hw=[2.]*29, velocity_hw=[2.]*29,
        joint_names_hw=names, mujoco_to_isaaclab=np.roll(np.arange(29), 5).tolist())
    sonic = tmp_path/'sonic'
    path = sonic/'robot_decoder/profiles/g1_29_wave_control.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(motion, 'ROOT', tmp_path/'calibration')
    return tmp_path/'result', sonic, profile


@pytest.mark.parametrize('angles', [(6., 4., 4.), (3., 6., 6.), (0., 8., 8.), (15., 15., 15.)])
def test_exact_two_minutes_has_stationary_legs_and_diverse_bounded_references(inputs, angles):
    output, sonic, profile = inputs
    reference, _, manifest = motion.prepare(output, sonic_root=sonic,
        yaw_deg=angles[0], roll_deg=angles[1], pitch_deg=angles[2])
    with np.load(output/'reference_sdk.npz', allow_pickle=False) as arrays:
        q = arrays['q_sdk_rad']
    assert q.shape == (6001, 29) and (len(q)-1)/50 == 120.
    frozen = np.r_[0:12, 15:29]
    expected = np.asarray(profile['default_angles_hw'])
    np.testing.assert_array_equal(q[:, frozen], np.tile(expected[frozen], (len(q), 1)))
    np.testing.assert_array_equal(q[[0, -1]], np.tile(expected, (2, 1)))
    assert len(np.unique(q[np.array([h['reference_end_frame']-1 for h in manifest['holds']]), 12:15], axis=0)) >= 18
    assert np.max(np.abs(np.gradient(q, .02, axis=0, edge_order=2))) <= .14+1e-10
    for hold in manifest['holds']:
        start, end = hold['reference_start_frame'], hold['reference_end_frame']
        assert (end-start)/50 >= 2.
        np.testing.assert_allclose(q[start:end, 12:15], np.tile(hold['target_waist_rad'], (end-start, 1)), atol=1e-15)
    exported = np.loadtxt(reference/'joint_pos.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(exported[:, np.argsort(profile['mujoco_to_isaaclab'])], q, atol=1e-11)
    assert manifest['actual_lower_body_constraint_enforced'] is False
    assert manifest['hardware_eligible'] is False


@pytest.mark.parametrize('angles', [(-1., 4., 4.), (float('nan'), 4., 4.), (0., 0., 16.)])
def test_bad_or_too_fast_reference_is_rejected_without_a_motor_path(inputs, angles):
    output, sonic, _ = inputs
    with pytest.raises(ValueError):
        motion.prepare(output, sonic_root=sonic, yaw_deg=angles[0], roll_deg=angles[1], pitch_deg=angles[2])
    assert not (output/'references/calibration_120s/joint_pos.csv').exists()


def test_pose_diameter_uses_short_so3_angle_and_is_world_frame_invariant():
    p = np.tile(np.eye(4), (2, 1, 1))
    p[:, :3, :3] = Rotation.from_euler('z', [[179.], [-179.]], degrees=True).as_matrix()
    p[1, 0, 3] = .004
    mm, deg = motion._diameter(p)
    assert mm == pytest.approx(.004) and deg == pytest.approx(2.)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler('xy', [.3, -.5]).as_matrix()
    world[:3, 3] = [3., 2., 1.]
    np.testing.assert_allclose(motion._diameter(world@p), (mm, deg), atol=1e-12)


def test_operator_40_pose_list_is_visited_in_order_and_held(inputs):
    output, sonic, profile = inputs
    poses = load_yaml(ROOT/'config/calibration_poses_40.yaml')['poses']
    assert len(poses) == 40 and len({tuple(p) for p in poses}) == 40 and poses[0] == [0, 0, 0]
    reference, _, manifest = motion.prepare_pose_list(output, poses, sonic_root=sonic)
    with np.load(output/'reference_sdk.npz', allow_pickle=False) as arrays:
        q = arrays['q_sdk_rad']
    default = np.asarray(profile['default_angles_hw'])
    frozen = np.r_[0:12, 15:29]
    np.testing.assert_array_equal(q[:, frozen], np.tile(default[frozen], (len(q), 1)))
    np.testing.assert_array_equal(q[[0, -1]], np.tile(default, (2, 1)))
    assert manifest['duration_s'] == (len(q)-1)/50 and 150 < manifest['duration_s'] < 240
    assert np.max(np.abs(np.gradient(q, .02, axis=0, edge_order=2))) <= .14+1e-10
    assert [h['pose_id'] for h in manifest['holds']] == list(range(1, 41))
    previous_end = 0
    for hold, pose in zip(manifest['holds'], poses):
        start, end = hold['reference_start_frame'], hold['reference_end_frame']
        assert start >= previous_end and (end-start)/50 == 2.
        np.testing.assert_allclose(q[start:end, 12:15], np.tile(default[12:15]+np.deg2rad(pose), (end-start, 1)), atol=1e-15)
        previous_end = end
    assert np.loadtxt(reference/'joint_pos.csv', delimiter=',', skiprows=1).shape == (len(q), 29)


@pytest.mark.parametrize('poses, message', [([[25, 0, 0]], 'within 20'), ([], 'nonempty'), ([[0, 0]], 'nonempty')])
def test_pose_list_rejects_out_of_range_or_malformed_poses(inputs, poses, message):
    output, sonic, _ = inputs
    with pytest.raises(ValueError, match=message):
        motion.prepare_pose_list(output, poses, sonic_root=sonic)
