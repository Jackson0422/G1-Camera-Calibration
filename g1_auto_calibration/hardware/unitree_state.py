"""Only ChannelSubscriber on rt/lowstate. No controller/mode switching APIs."""
from collections import deque
from dataclasses import dataclass
import sys
import threading
import time
import numpy as np
from .reader_diagnostics import ReaderDiagnostics


def remote_keys(payload):
    """Read the two button bytes used by the official SDK joystick decoder.

    This is software input monitoring, not proof of a paired controller or an
    independent emergency-stop path. Unknown input remains unavailable.
    """
    try:
        if len(payload) != 40 or any(isinstance(value, (bool, np.bool_))
                                     or not isinstance(value, (int, np.integer))
                                     or not 0 <= int(value) <= 255 for value in payload):
            return None
        return int(payload[2]) | (int(payload[3]) << 8)
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass
class State:
    received_s: float
    tick: int
    q: np.ndarray
    dq: np.ndarray
    mode_pr: int
    imu_gyro: np.ndarray
    mode_machine: int | None = None
    remote_keys: int | None = None
    # Original sensor values for diagnostics; never substitute a model torque
    # or infer support forces from these fields. Older saved/test states may
    # omit them. The primary HG IMU quaternion is in SDK WXYZ order.
    tau_est: np.ndarray | None = None
    imu_quaternion_wxyz: np.ndarray | None = None


class StateBuffer:
    def __init__(self):
        self.states = deque(maxlen=20000)
        self.lock = threading.Lock()
        self.error = None
        self.duplicate_ticks = 0

    def append(self, s):
        if s.mode_pr != 0:
            self.error = 'LowState is AB actuator coordinates; PR FK is invalid. Do not switch robot modes here.'
            return
        if not np.isfinite(np.r_[s.q, s.dq, s.imu_gyro]).all():
            self.error = 'Nonfinite robot state'
            return
        with self.lock:
            if self.states and s.tick == self.states[-1].tick:
                self.duplicate_ticks += 1
                return
            if self.states and s.tick < self.states[-1].tick:
                self.error = 'LowState tick reset/wrapped/reordered; restart collection'
                return
            self.states.append(s)

    def settled_at(self, frame, robot_config, camera_config):
        if self.error:
            raise ValueError(self.error)
        sl = robot_config.get('state_max_latency_s')
        cl = camera_config.get('max_transport_latency_s')
        if sl is None or not robot_config.get('state_timing_evidence'):
            raise ValueError('Missing independently established DDS transport latency bound/evidence')
        sl = float(sl)
        if not np.isfinite(sl) or not 0 <= sl <= 5:
            raise ValueError('Invalid DDS latency bound')
        if frame.capture_monotonic_s is not None and frame.timing_uncertainty_s is not None:
            lo = frame.capture_monotonic_s-frame.timing_uncertainty_s
            hi = frame.capture_monotonic_s+frame.timing_uncertainty_s
        else:
            if cl is None or not camera_config.get('timing_evidence'):
                raise ValueError('Remote capture clock is unaligned; establish transport bound before collection')
            cl = float(cl)
            if not np.isfinite(cl) or not 0 < cl <= 5:
                raise ValueError('Invalid camera latency bound')
            lo, hi = frame.received_monotonic_s-cl, frame.received_monotonic_s
        start = lo-float(robot_config['settle_window_s'])
        end = hi+sl
        with self.lock:
            states = list(self.states)
        if not states or states[0].received_s > start or states[-1].received_s < end:
            raise ValueError('Need complete state history covering capture uncertainty and settling interval')
        subset = [s for s in states if start-.03 <= s.received_s <= end+.03]
        if len(subset) < 5 or np.max(np.diff([s.received_s for s in subset])) > .05:
            raise ValueError('State history has gaps')
        qs = np.array([s.q for s in subset])
        from ..calibration.stability import measure, policy
        mount_stability = None
        if policy(robot_config) is not None:
            mount_stability = measure(qs, robot_config)
            if not mount_stability['passed']:
                raise ValueError('Mount pose exceeds the accepted accuracy budget over the capture interval')
        else:
            if np.max(np.abs([s.dq for s in subset])) > robot_config['max_joint_speed_rad_s'] or np.max(np.ptp(qs, axis=0)) > robot_config['max_joint_range_rad']:
                raise ValueError('Robot has not settled over complete capture interval')
            if np.max(np.linalg.norm([s.imu_gyro for s in subset], axis=1)) > .05:
                raise ValueError('IMU indicates motion (quiet IMU cannot prove fixed base)')
        nearest = min(subset, key=lambda s: abs(s.received_s-(lo+hi)/2))
        return nearest, dict(capture_interval_local_s=[lo, hi], state_latency_bound_s=sl,
                             state_interval_s=[subset[0].received_s, subset[-1].received_s],
                             max_joint_range_rad=float(np.max(np.ptp(qs, axis=0))),
                             mount_stability=mount_stability,
                             pairing=('bounded capture interval with URDF mount-pose stability; no device-clock equality assumed'
                                      if mount_stability is not None else
                                      'bounded capture interval with stationary joints; no device-clock equality assumed'))


class UnitreeState(StateBuffer):
    def __init__(self, config):
        super().__init__()
        self._reader_diagnostics = ReaderDiagnostics()
        sys.path.insert(0, config['sdk_source'])
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        if config.get('topic') != 'rt/lowstate':
            raise ValueError('Only rt/lowstate allowed')
        ChannelFactoryInitialize(int(config['domain']), config['interface'])
        self.subscriber = ChannelSubscriber('rt/lowstate', LowState_)
        self.subscriber.Init(self._receive, 10)
        self._reader_diagnostics.install(self.subscriber)

    def _receive(self, msg):
        now = time.monotonic()
        self.append(State(now, int(msg.tick), np.array([s.q for s in msg.motor_state[:29]]),
                          np.array([s.dq for s in msg.motor_state[:29]]), int(msg.mode_pr),
                          np.array(msg.imu_state.gyroscope), int(msg.mode_machine),
                          remote_keys(getattr(msg, 'wireless_remote', None)),
                          tau_est=np.array([s.tau_est for s in msg.motor_state[:29]]),
                          imu_quaternion_wxyz=np.array(msg.imu_state.quaternion)))

    def close(self):
        self._reader_diagnostics.mark_closing()
        try:
            self.subscriber.Close()
        finally:
            self._reader_diagnostics.restore()

    def reader_diagnostics_snapshot(self):
        return self._reader_diagnostics.snapshot()
