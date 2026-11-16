"""Record existing state and camera observations without command-thread I/O."""

import json
from pathlib import Path
import threading

import numpy as np

from ..common import digest, load_yaml, plain, write_yaml


class TimingTrace:
    """Own only files and a reader thread; create no robot/camera transport.

    State packets are retained at their original timestamps and values. Camera
    callbacks run on their existing camera workers, never on the motor loop.
    """

    def __init__(self, output, state_buffer):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.buffer = state_buffer
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.last_tick = None
        self.last_received_s = None
        self.chunks = []
        self.camera_files = {}
        self.camera_counts = {}
        self.camera_lock = threading.Lock()
        self.pending = []
        self.state_count = 0

    def start(self):
        self.thread = threading.Thread(target=self._run, name='calibration_timing_state_log', daemon=True)
        self.thread.start()
        return self

    def _save(self):
        if not self.pending:
            return
        states, self.pending = self.pending, []
        path = self.output / f'state_{len(self.chunks):05d}.npz'
        np.savez_compressed(path,
            received_s=np.array([s.received_s for s in states]),
            tick=np.array([s.tick for s in states], dtype=np.int64),
            q=np.array([s.q for s in states]), dq=np.array([s.dq for s in states]),
            imu_gyro=np.array([s.imu_gyro for s in states]),
            mode_pr=np.array([s.mode_pr for s in states]),
            mode_machine=np.array([s.mode_machine for s in states]))
        self.state_count += len(states)
        self.chunks.append(dict(file=path.name, sha256=digest(path), count=len(states)))

    def _poll(self):
        with self.buffer.lock:
            states = list(self.buffer.states)
        fresh = [s for s in states if self.last_tick is None or s.tick > self.last_tick]
        if fresh:
            if self.last_received_s is not None and fresh[0].received_s - self.last_received_s > .05:
                raise ValueError('Timing trace lost complete state history')
            self.pending.extend(fresh)
            self.last_tick = fresh[-1].tick
            self.last_received_s = fresh[-1].received_s
        if len(self.pending) >= 2000:
            self._save()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self._poll()
                self.stop_event.wait(.05)
            self._poll()
            self._save()
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def record_camera(self, name, packet):
        if self.error:
            return
        try:
            frame = packet.frame
            observation = packet.observation
            record = dict(source_timestamp_s=frame.source_timestamp_s, source_clock=frame.source_clock,
                received_monotonic_s=frame.received_monotonic_s, sequence=frame.sequence,
                serial=frame.serial, optical_frame=frame.optical_frame,
                T_camera_board=None if observation is None else observation['T_camera_board'],
                reprojection_rms_px=None if observation is None else observation['reprojection_rms_px'],
                rejection_reason=packet.rejection_reason)
            with self.camera_lock:
                if name not in self.camera_files:
                    self.camera_files[name] = (self.output / f'{name}_frames.jsonl').open('x', buffering=65536)
                    self.camera_counts[name] = 0
                self.camera_files[name].write(json.dumps(plain(record), allow_nan=False) + '\n')
                self.camera_counts[name] += 1
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.)
        pending = self.thread is not None and self.thread.is_alive()
        cameras = {}
        with self.camera_lock:
            for name, stream in self.camera_files.items():
                stream.close()
                path = self.output / f'{name}_frames.jsonl'
                cameras[name] = dict(file=path.name, sha256=digest(path), count=self.camera_counts[name])
        report = dict(kind='continuous_calibration_timing_trace', complete=not pending and self.error is None,
            error=self.error, state_count=self.state_count, states=self.chunks, cameras=cameras,
            source_clock='local_monotonic_state_receipt', values_averaged=False,
            robot_control_commands_sent=0, pending_state_writer=pending)
        if not pending:
            write_yaml(self.output / 'manifest.yaml', report)
        return report


def load_trace(directory, camera):
    """Read closed, hash-checked data; never initialize a subscriber or camera."""
    directory = Path(directory).resolve()
    manifest = load_yaml(directory / 'manifest.yaml')
    if manifest.get('complete') is not True or manifest.get('kind') != 'continuous_calibration_timing_trace':
        raise ValueError('Continuous timing trace is incomplete')

    def checked_path(entry):
        value = entry['file']
        if not isinstance(value, str) or Path(value).name != value:
            raise ValueError('Timing trace file must remain inside its trace directory')
        path = (directory / value).resolve(strict=True)
        path.relative_to(directory)
        if digest(path) != entry['sha256']:
            raise ValueError('Timing trace file hash changed after recording')
        return path

    parts = []
    for entry in manifest['states']:
        with np.load(checked_path(entry), allow_pickle=False) as data:
            part = {key: data[key].copy() for key in
                ('received_s', 'tick', 'q', 'dq', 'imu_gyro', 'mode_pr', 'mode_machine')}
        count = entry['count']
        if (count < 1 or any(len(value) != count or not np.isfinite(value).all() for value in part.values())
                or part['q'].shape != (count, 29) or part['dq'].shape != (count, 29)
                or part['imu_gyro'].shape != (count, 3)
                or any(part[key].shape != (count,) for key in ('received_s', 'tick', 'mode_pr', 'mode_machine'))):
            raise ValueError('Timing state chunk has invalid dimensions or values')
        parts.append(part)
    if not parts:
        raise ValueError('Timing trace contains no joint state')
    states = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if (np.any(np.diff(states['tick']) <= 0) or np.any(states['mode_pr'] != 0)
            or len(np.unique(states['mode_machine'])) != 1):
        raise ValueError('Timing trace contains a state reset, mode change or non-PR coordinates')
    frames = [json.loads(line) for line in checked_path(manifest['cameras'][camera]).read_text().splitlines() if line]
    if len(frames) != manifest['cameras'][camera]['count']:
        raise ValueError('Timing camera frame count differs from its manifest')
    return manifest, states, frames
