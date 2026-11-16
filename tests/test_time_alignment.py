"""Time mapping is estimated from synthetic clocks and actual URDF rotations."""

from copy import deepcopy
import socket

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_auto_calibration.calibration.stability import model
from g1_auto_calibration.calibration.time_alignment import associate_time, fit_motion_clock
from g1_auto_calibration.calibration.transforms import inverse, transform
from g1_auto_calibration.common import ROOT, load_yaml


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('Timing tests must not connect to hardware'))


@pytest.fixture(scope='module')
def trace():
    robot = load_yaml(ROOT / 'config/robot_sonic.yaml')
    def joints(t):
        q = np.zeros((len(t), 29))
        q[:, 12] = .23 * np.sin(.37*t) + .07*np.sin(.113*t*t / 20)
        q[:, 13] = .12 * np.sin(.23*t+.4)
        q[:, 14] = .15 * np.sin(.41*t-.2)
        return q
    times = np.arange(980., 1065., .01)
    q = joints(times-1000.)
    source = np.arange(500., 560., .05)
    expected = 1000.37 + 1.00017 * (source - 500.)
    G = model(robot).compute_many(joints(expected-1000.))
    X = transform(Rotation.from_euler('xyz', [.2, -.3, .1]).as_matrix(), [.1, -.03, .2])
    Y = transform(Rotation.from_euler('xyz', [.1, .2, -.3]).as_matrix(), [.5, .2, 2.])
    frames = [dict(source_timestamp_s=float(c), received_monotonic_s=float(t+.15),
        source_clock='SYNTHETIC_HARDWARE_CLOCK', serial='SYNTHETIC_CAMERA', optical_frame='synthetic_optical',
        sequence=i+1, T_camera_board=inverse(g @ X) @ Y)
        for i, (c, t, g) in enumerate(zip(source, expected, G))]
    return robot, times, q, frames


def test_recovers_offset_rate_and_excludes_saved_calibration_images(trace):
    robot, times, q, frames = trace
    excluded = list(range(17, len(frames), 17))
    result = fit_motion_clock(times, q, frames, robot, excluded_sequences=excluded)
    assert result['validation_passed']
    assert result['deterministic_transport_bound'] is False
    assert result['camera_to_state_rate'] == pytest.approx(1.00017, abs=2e-5)
    mapped, radius = associate_time(result, 530.)
    assert mapped == pytest.approx(1000.37 + 1.00017*30, abs=.003)
    assert radius > 0
    train = {s for pair in result['training_frame_sequences'] for s in pair}
    held = {s for pair in result['held_out_frame_sequences'] for s in pair}
    assert not train.intersection(held)
    assert not (train | held).intersection(excluded)


def test_static_data_is_not_claimed_to_identify_time_offset(trace):
    robot, times, q, frames = trace
    stationary = deepcopy(frames)
    for frame in stationary:
        frame['T_camera_board'] = np.eye(4)
    with pytest.raises(ValueError, match='not observable'):
        fit_motion_clock(times, np.zeros_like(q), stationary, robot)


def test_clock_reset_is_not_silently_mixed_into_one_mapping(trace):
    robot, times, q, frames = trace
    changed = deepcopy(frames)
    changed[-1]['source_clock'] = 'RESTARTED_CLOCK'
    with pytest.raises(ValueError, match='source clock changed'):
        fit_motion_clock(times, q, changed, robot)


def test_unknown_alignment_cannot_supply_a_capture_interval():
    with pytest.raises(ValueError, match='did not pass'):
        associate_time({'status': 'unobservable'}, 0.)


def test_noisy_camera_rotation_and_receipt_jitter_keep_known_mapping(trace):
    robot, times, q, frames = trace
    noisy = deepcopy(frames)
    rng = np.random.default_rng(145)
    for frame in noisy:
        frame['T_camera_board'][:3, :3] = (frame['T_camera_board'][:3, :3]
            @ Rotation.from_rotvec(rng.normal(0., np.deg2rad(.015), 3)).as_matrix())
        frame['received_monotonic_s'] += float(rng.uniform(-.005, .005))
    result = fit_motion_clock(times, q, noisy, robot)
    mapped, radius = associate_time(result, 530.)
    assert abs(mapped - (1000.37 + 1.00017*30)) < radius
    assert result['held_out_rotation_rms_deg'] < .5


def test_incomplete_state_history_does_not_produce_mapping(trace):
    robot, times, q, frames = trace
    keep = np.ones(len(times), bool)
    keep[500:550] = False
    with pytest.raises(ValueError, match='complete, strictly advancing'):
        fit_motion_clock(times[keep], q[keep], frames, robot)


def test_constant_angular_speed_cannot_identify_offset(trace):
    robot, times, q, frames = trace
    constant = deepcopy(frames)
    q = np.zeros_like(q)
    q[:, 12] = .01 * (times - 1000.)
    fk = model(robot)
    for frame in constant:
        value = np.zeros(29)
        value[12] = .01 * (1000.37 + 1.00017*(frame['source_timestamp_s'] - 500.) - 1000.)
        frame['T_camera_board'] = inverse(fk.compute(value))
    with pytest.raises(ValueError, match='not identifiable|Ambiguous timing|insufficient offset sensitivity'):
        fit_motion_clock(times, q, constant, robot)
