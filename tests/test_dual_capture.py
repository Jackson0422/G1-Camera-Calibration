import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from g1_auto_calibration.common import ROOT, load_yaml
from g1_auto_calibration.hardware.base import Frame
from g1_auto_calibration.hardware.unitree_state import State, StateBuffer
from g1_auto_calibration.observation.dual_capture import (
    CameraPacket, CameraWorker, ObservationRecorder, StationarityError, receive_time_stationarity,
    calibration_evidence_failures,
)
from g1_auto_calibration.scripts.observe_sonic import verify_wired_routes
from g1_auto_calibration.scripts.check_hardware import model_from_config


def states(end=5., moving_at=None):
    buffer = StateBuffer()
    for index, stamp in enumerate(np.arange(0., end + .005, .01)):
        q, dq = np.zeros(29), np.zeros(29)
        if moving_at is not None and moving_at[0] <= stamp < moving_at[1]:
            dq[12] = .1
        buffer.append(State(float(stamp), index, q, dq, 0, np.zeros(3), 2))
    return buffer


def packet(stamp, sequence=1, passed=True):
    frame = Frame(np.zeros((48, 64, 3), np.uint8), stamp + 1000., 'remote_test',
                  stamp, 'test_optical', 'test_serial',
                  np.array([[80., 0., 32.], [0., 80., 24.], [0., 0., 1.]]),
                  np.zeros(5), sequence)
    return CameraPacket(frame, dict(T_camera_board=np.eye(4), image_points_px=[],
                                    reprojection_rms_px=.2) if passed else None,
                        None if passed else 'Too few reliable known markers')


class DualCaptureTests(unittest.TestCase):
    def setUp(self):
        self.robot = load_yaml(ROOT / 'config/robot.yaml')
        self.board = load_yaml(ROOT / 'config/board.yaml')
        # Explicitly unverified fixture; the real board was subsequently
        # confirmed by the operator's supplied physical-panel specification.
        self.board['physical_pattern_verified'] = False
        self.cameras = load_yaml(ROOT / 'config/cameras.yaml')

    def test_supplied_panel_and_camera_specific_confirmation_do_not_authorize_oak_mount(self):
        from g1_auto_calibration.calibration.board_detector import BoardDetector

        robot = load_yaml(ROOT / 'config/robot_sonic.yaml')
        board = load_yaml(ROOT / 'config/board.yaml')
        detector = BoardDetector(board)
        self.assertTrue(board['physical_pattern_verified'])
        self.assertEqual(len(detector.charuco.getChessboardCorners()), 24)
        self.assertEqual(len(detector.markers), 18)
        self.assertNotIn('robot.mount_attachment_evidence', calibration_evidence_failures(robot, self.cameras['d435'], board))
        self.assertIn('robot.mount_attachment_evidence', calibration_evidence_failures(robot, self.cameras['oak'], board))

    def test_candidate_images_preserve_missing_timing_and_physical_board(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ObservationRecorder(Path(directory) / 'session', self.robot,
                                           self.cameras, self.board)
            record = recorder.observe('d435', packet(2.), states(end=2.), now=2.)
            self.assertTrue(record['observation_quality_passed'])
            self.assertFalse(record['calibration_ready'])
            self.assertIsNone(record['planned_pose_id'])
            self.assertIsNone(record['bounded_capture_alignment'])
            self.assertIn('latency', record['capture_alignment_failure'])
            self.assertFalse(recorder.session['board_config']['physical_pattern_verified'])
            self.assertIn('robot.base_fixed_evidence', recorder.session['cameras']['d435'][
                'missing_calibration_evidence'])
            self.assertTrue((Path(directory) / 'session/d435/0000.png').exists())
            saved = load_yaml(Path(directory) / 'session/observations.yaml')
            self.assertEqual(saved['control_commands_sent'], 0)
            self.assertNotEqual(saved['kind'], 'real')

    def test_camera_rejections_do_not_block_other_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ObservationRecorder(Path(directory) / 'session', self.robot,
                                           self.cameras, self.board)
            buffer = states(end=3.)
            rejected = recorder.observe('d435', packet(2., passed=False), buffer, now=3.)
            accepted = recorder.observe('oak', packet(2.01), buffer, now=3.)
            retry = recorder.observe('d435', packet(2.8, sequence=2), buffer, now=3.)
            self.assertFalse(rejected['observation_quality_passed'])
            self.assertTrue(accepted['observation_quality_passed'])
            self.assertTrue(retry['observation_quality_passed'])
            self.assertEqual(rejected['stationary_interval_id'], accepted['stationary_interval_id'])
            self.assertEqual(recorder.session['cameras']['d435']['quality_passed_count'], 1)
            self.assertEqual(recorder.session['cameras']['oak']['quality_passed_count'], 1)
            self.assertIsNone(recorder.observe('oak', packet(3., sequence=2), buffer, now=3.))

    def test_capture_aligned_state_is_preserved_separately_from_receive_time_state(self):
        with tempfile.TemporaryDirectory() as directory:
            self.robot.update(state_max_latency_s=.02, state_timing_evidence='OFFLINE TEST bound')
            self.cameras['d435'].update(max_transport_latency_s=.1, timing_evidence='OFFLINE TEST bound')
            _, fk = model_from_config(ROOT / 'config/robot.yaml')
            buffer = StateBuffer()
            for index, stamp in enumerate(np.arange(0., 2.11, .01)):
                q, dq = np.zeros(29), np.zeros(29)
                q[12], dq[12] = stamp * .0005, .0005
                buffer.append(State(float(stamp), index, q, dq, 0, np.zeros(3), 5))
            recorder = ObservationRecorder(Path(directory) / 'session', self.robot,
                self.cameras, self.board, fk=fk)
            record = recorder.observe('d435', packet(2.), buffer, now=2.1)
            aligned = record['capture_aligned_state']
            self.assertIsNotNone(aligned)
            self.assertEqual(aligned['tick'], 195)
            self.assertEqual(record['tick'], 200)
            self.assertLess(aligned['q'][12], record['q'][12])
            np.testing.assert_allclose(record['T_reference_mount_from_capture_aligned_state'],
                                       fk.compute(aligned['q']))
            self.assertEqual(record['urdf_sha256'], fk.sha256)
            self.assertEqual(record['urdf_path'], str(fk.path.resolve()))
            self.assertIsNone(record['capture_alignment_failure'])
            self.assertTrue(record['candidate_only'])
            self.assertFalse(record['calibration_ready'])

    def test_stationary_interval_does_not_invent_commanded_pose_arrival(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ObservationRecorder(Path(directory) / 'session', self.robot,
                                           self.cameras, self.board)
            with self.assertRaisesRegex(ValueError, 'arrival evidence'):
                recorder.set_pose_context(1)
            recorder.set_pose_context(17, arrival_evidence='External coordinator measured target error')
            record = recorder.observe('d435', packet(2.), states(end=2.), now=2.)
            self.assertEqual(record['planned_pose_id'], 17)
            self.assertFalse(record['receive_time_stationarity']['commanded_target_arrival_verified'])
            self.assertIn('External coordinator', record['external_arrival_evidence'])

    def test_movement_rearms_observation_without_old_camera_resetting_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ObservationRecorder(Path(directory) / 'session', self.robot,
                                           self.cameras, self.board)
            buffer = states(end=4., moving_at=(2.1, 2.3))
            first = recorder.observe('d435', packet(2.), buffer, now=4.)
            self.assertIsNone(recorder.observe('d435', packet(2.5, sequence=2), buffer, now=4.))
            second = recorder.observe('d435', packet(3.5, sequence=3), buffer, now=4.)
            self.assertIsNone(recorder.observe('oak', packet(2.5), buffer, now=4.))
            other = recorder.observe('oak', packet(3.6, sequence=2), buffer, now=4.)
            self.assertEqual(second['stationary_interval_id'], first['stationary_interval_id'] + 1)
            self.assertEqual(other['stationary_interval_id'], second['stationary_interval_id'])

    def test_stale_state_does_not_trigger_capture(self):
        with self.assertRaisesRegex(ValueError, 'stale'):
            receive_time_stationarity(states(end=2.), 2., self.robot, now=3.)

    def test_rejected_stationary_window_retains_measurements_without_relaxing_limits(self):
        buffer = states(end=2., moving_at=(1., 2.1))
        with self.assertRaises(StationarityError) as raised:
            receive_time_stationarity(buffer, 2., self.robot, now=2.)
        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics['maximum_joint_speed_rad_s'], .1)
        self.assertEqual(diagnostics['maximum_speed_joint_index'], 12)
        self.assertEqual(diagnostics['joint_speed_limit_rad_s'], self.robot['max_joint_speed_rad_s'])
        self.assertEqual(diagnostics['maximum_joint_range_rad'], 0.)
        self.assertEqual(diagnostics['maximum_imu_gyro_rad_s'], 0.)
        self.assertIn('0.100000', str(raised.exception))

    def test_wired_route_gate_rejects_wireless_or_wrong_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            interface = Path(directory) / 'eno1'
            interface.mkdir()
            routes = verify_wired_routes(self.cameras, 'eno1', sys_root=directory,
                                        route_reader=lambda host: [dict(dev='eno1', dst=host)])
            self.assertEqual(set(routes), {'d435', 'oak'})
            with self.assertRaisesRegex(ValueError, 'not routed'):
                verify_wired_routes(self.cameras, 'eno1', sys_root=directory,
                                    route_reader=lambda host: [dict(dev='wlan0')])
            (interface / 'wireless').mkdir()
            with self.assertRaisesRegex(ValueError, 'wireless'):
                verify_wired_routes(self.cameras, 'eno1', sys_root=directory,
                                    route_reader=lambda host: [dict(dev='eno1')])

    def test_camera_threads_are_independent_and_close_on_stop(self):
        closed = {}
        slow_started = threading.Event()

        class Reader:
            def __init__(self, config):
                self.name = config['name']
                self.sequence = 0

            def read(self, timeout_s):
                if self.name == 'slow':
                    slow_started.set()
                    time.sleep(min(timeout_s, .2))
                    raise TimeoutError('Unavailable camera')
                time.sleep(.01)
                self.sequence += 1
                return packet(time.monotonic(), self.sequence).frame

            def close(self):
                closed[self.name] = True

        class Detector:
            def estimate(self, frame):
                return dict(reprojection_rms_px=.1)

        callbacks = []
        slow = CameraWorker('slow', dict(name='slow'), Reader, Detector(), read_timeout_s=.2)
        fast = CameraWorker('fast', dict(name='fast'), Reader, Detector(), read_timeout_s=.2,
                            packet_callback=lambda value: callbacks.append((value, threading.current_thread().name)))
        slow.start()
        fast.start()
        try:
            self.assertTrue(slow_started.wait(timeout=1.))
            deadline = time.monotonic() + 1.
            while fast.snapshot()[0] is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIsNotNone(fast.snapshot()[0])
            self.assertIsNone(slow.snapshot()[0])
        finally:
            self.assertTrue(slow.stop())
            self.assertTrue(fast.stop())
        self.assertEqual(closed, dict(slow=True, fast=True))
        self.assertTrue(callbacks)
        self.assertTrue(all(thread == 'observe_fast' for _, thread in callbacks))


if __name__ == '__main__':
    unittest.main()


def test_parallel_detection_keeps_frame_order_and_throughput():
    import threading
    import time as _time
    from types import SimpleNamespace
    from g1_auto_calibration.observation.dual_capture import CameraWorker

    class Reader:
        def __init__(self):
            self.n = 0
        def read(self, timeout_s):
            if self.n >= 30:
                _time.sleep(.01)
                raise TimeoutError('done')
            self.n += 1
            return SimpleNamespace(sequence=self.n, received_monotonic_s=_time.monotonic(), validate=lambda: None)
        def close(self):
            pass

    class SlowDetector:  # 60 ms, alternating fast/slow so completion order differs from read order
        def estimate(self, frame):
            _time.sleep(.09 if frame.sequence % 2 else .03)
            if frame.sequence % 5 == 0:
                raise ValueError('no board')
            return dict(quality_rejection_reason=None, sequence=frame.sequence)

    published, finished = [], threading.Event()
    def callback(packet):
        published.append((packet.frame.sequence, packet.observation and packet.observation['sequence']))
        if len(published) == 30:
            finished.set()
    worker = CameraWorker('oak', {}, lambda config: Reader(), SlowDetector(), packet_callback=callback)
    started = _time.monotonic()
    worker.start()
    assert finished.wait(5)
    elapsed = _time.monotonic()-started
    worker.stop()
    assert [s for s, _ in published] == list(range(1, 31))
    assert all(o == s or (o is None and s % 5 == 0) for s, o in published)
    assert elapsed < 30*.06*.6  # clearly faster than serial detection
