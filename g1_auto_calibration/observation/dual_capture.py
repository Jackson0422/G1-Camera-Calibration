"""Independent camera diagnostics and automatic, explicitly unpaired observations.

Receive-time stationarity is useful for collecting candidate images, but cannot
establish camera/DDS capture alignment, target arrival, or a fixed pelvis. This
module contains no motion transport and never promotes its files to calibration
sessions. A later, evidence-backed conversion is a separate operation.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from ..common import canonical_digest, write_yaml
from ..calibration.foot_reference import FixedStanceMonitor
from ..calibration.stability import measure as measure_mount_stability, policy as stability_policy
from ..calibration.time_alignment import has_motion_alignment


def _atomic_yaml(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    write_yaml(temporary, value)
    temporary.replace(path)


class StationarityError(ValueError):
    """A rejected stationary window with its measured values still available."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def receive_time_stationarity(state_buffer, received_s, robot_config, *, now=None,
                              maximum_state_age_s=.25):
    """Inspect a local receive-time window without claiming exposure alignment."""
    now = time.monotonic() if now is None else float(now)
    if state_buffer.error:
        raise ValueError(state_buffer.error)
    with state_buffer.lock:
        states = list(state_buffer.states)
    if not states or now - states[-1].received_s > maximum_state_age_s:
        raise ValueError('Robot state is missing or stale')
    if states[-1].received_s > now + .05:
        raise ValueError('Robot state receive time is in the future')
    if received_s > now + .05:
        raise ValueError('Frame receive time is in the future')
    window_s = float(robot_config['settle_window_s'])
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError('Settling window must be positive and finite')
    start = float(received_s) - window_s
    before = [s for s in states if s.received_s <= start]
    within = [s for s in states if start < s.received_s <= received_s]
    if not before or not within:
        raise ValueError('Insufficient receive-time settling history')
    subset = [before[-1], *within]
    times = np.array([s.received_s for s in subset])
    if (len(subset) < 5 or np.max(np.diff(times)) > .05
            or received_s - times[-1] > .05):
        raise ValueError('Robot state history has gaps near the frame')
    q = np.array([s.q for s in subset])
    dq = np.array([s.dq for s in subset])
    gyroscope = np.array([s.imu_gyro for s in subset])
    if q.shape != (len(subset), 29) or dq.shape != q.shape:
        raise ValueError('Expected 29 PR joint positions and velocities')
    if not np.isfinite(np.r_[q.ravel(), dq.ravel(), gyroscope.ravel()]).all():
        raise ValueError('Nonfinite state in settling history')
    if any(s.mode_pr != 0 for s in subset):
        raise ValueError('PR joint coordinates are required')
    maximum_speed = float(np.max(np.abs(dq)))
    maximum_range = float(np.max(np.ptp(q, axis=0)))
    maximum_gyro = float(np.max(np.linalg.norm(gyroscope, axis=1)))
    diagnostics = dict(
        state_receive_interval_s=[float(times[0]), float(times[-1])],
        maximum_joint_speed_rad_s=maximum_speed,
        maximum_speed_joint_index=int(np.unravel_index(np.argmax(np.abs(dq)), dq.shape)[1]),
        maximum_joint_range_rad=maximum_range,
        maximum_range_joint_index=int(np.argmax(np.ptp(q, axis=0))),
        maximum_imu_gyro_rad_s=maximum_gyro,
        joint_speed_limit_rad_s=float(robot_config['max_joint_speed_rad_s']),
        joint_range_limit_rad=float(robot_config['max_joint_range_rad']),
        imu_gyro_limit_rad_s=.05)
    if stability_policy(robot_config) is not None:
        stability = measure_mount_stability(q, robot_config)
        diagnostics.update(mount_stability=stability, raw_joint_speed_used_as_stability_gate=False)
        diagnostics.update(joint_speed_limit_rad_s=None, joint_range_limit_rad=None, imu_gyro_limit_rad_s=None)
        if not stability['passed']:
            raise StationarityError('URDF mount pose changes beyond the accepted sampling accuracy budget: '
                f'{stability["translation_diameter_bound_m"]*1000:.6f} mm, '
                f'{stability["rotation_diameter_bound_deg"]:.6f} deg', diagnostics)
    elif (maximum_speed > float(robot_config['max_joint_speed_rad_s'])
            or maximum_range > float(robot_config['max_joint_range_rad'])
            or maximum_gyro > .05):
        raise StationarityError('Robot is moving in the local receive-time window: '
            f'joint speed {maximum_speed:.6f}/{diagnostics["joint_speed_limit_rad_s"]:.6f} rad/s, '
            f'joint range {maximum_range:.6f}/{diagnostics["joint_range_limit_rad"]:.6f} rad, '
            f'IMU gyro norm {maximum_gyro:.6f}/0.050000 rad/s', diagnostics)
    return subset[-1], dict(
        status='receive_time_stationarity_only',
        camera_exposure_alignment_verified=False,
        commanded_target_arrival_verified=False,
        fixed_pelvis_verified=False,
        **diagnostics,
    )


def calibration_evidence_failures(robot_config, camera_config, board_config):
    """Return concrete missing evidence without changing configuration values."""
    missing = []
    if not (camera_config.get('mount_attachment_evidence') or robot_config.get('mount_attachment_evidence')):
        missing.append('robot.mount_attachment_evidence')
    for name in ('base_fixed_evidence', 'board_stationary_evidence'):
        if not robot_config.get(name):
            missing.append('robot.' + name)
    if not has_motion_alignment(camera_config):
        if not robot_config.get('state_timing_evidence'):
            missing.append('robot.state_timing_evidence')
        if robot_config.get('state_max_latency_s') is None:
            missing.append('robot.state_max_latency_s')
        for name in ('timing_evidence', 'max_transport_latency_s'):
            if camera_config.get(name) is None or not camera_config.get(name):
                missing.append('camera.' + name)
    if board_config.get('physical_pattern_verified') is not True:
        missing.append('board.physical_pattern_verified')
    return missing


@dataclass(frozen=True)
class CameraPacket:
    frame: object
    observation: dict | None
    rejection_reason: str | None
    png: bytes | None = None  # lossless encoding prepared off the recording thread


class CameraWorker:
    """Own one reader in its own thread so another camera cannot block it."""

    def __init__(self, name, config, opener, detector, *, read_timeout_s=.5, packet_callback=None,
                 detection_workers=3, max_pending=8, encode_png=False):
        self.name, self.config = name, config
        self.opener, self.detector = opener, detector
        # Board detection (~25-60 ms at 1080p) runs on a small pool so that, with the
        # ordered recording callback (~45 ms), a 15 fps stream is not halved.
        self.detection_workers = int(detection_workers)
        self.max_pending = int(max_pending)
        if self.detection_workers < 1 or self.max_pending < 1:
            raise ValueError('Need at least one detection worker and one pending frame')
        self.encode_png = bool(encode_png)
        self.read_timeout_s = float(read_timeout_s)
        self.packet_callback = packet_callback
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.packet = None
        self.error = None
        self.last_frame_received_s = None
        self.thread = threading.Thread(target=self._run,
                                       name='observe_' + name, daemon=True)

    def start(self):
        self.thread.start()

    def _detect(self, frame):
        png = None
        if self.encode_png:
            ok, encoded = cv2.imencode('.png', frame.bgr)
            if not ok:
                raise OSError('Lossless frame encoding failed')
            png = encoded.tobytes()
        try:
            observation = self.detector.estimate(frame)
            return observation, observation.get('quality_rejection_reason'), png
        except ValueError as exc:
            return None, str(exc), png

    def _publish(self, frame, observation, reason, png=None):
        packet = CameraPacket(frame, observation, reason, png)
        with self.lock:
            self.packet = packet
            self.last_frame_received_s = frame.received_monotonic_s
            self.error = None
        if self.packet_callback is not None:
            self.packet_callback(packet)

    def _run(self):
        reader = None
        pending = deque()
        executor = ThreadPoolExecutor(max_workers=self.detection_workers,
                                      thread_name_prefix='detect_' + self.name)
        try:
            reader = self.opener(self.config)
            while not self.stop_event.is_set():
                try:
                    frame = reader.read(timeout_s=self.read_timeout_s)
                except TimeoutError as exc:
                    with self.lock:
                        self.error = str(exc)
                    frame = None
                if frame is not None:
                    frame.validate()
                    pending.append((frame, executor.submit(self._detect, frame)))
                # Publish in arrival order; wait on the oldest only when the queue is full.
                while pending and (pending[0][1].done() or len(pending) > self.max_pending):
                    done, future = pending.popleft()
                    self._publish(done, *future.result())
            while pending:  # frames already read are still recorded, in order
                done, future = pending.popleft()
                self._publish(done, *future.result())
        except Exception as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:
                    with self.lock:
                        self.error = f'Camera close failed: {type(exc).__name__}: {exc}'

    def snapshot(self):
        with self.lock:
            return self.packet, self.error, self.last_frame_received_s

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=max(2., self.read_timeout_s + 1.))
        return not self.thread.is_alive()


class ObservationRecorder:
    """Save independent raw candidates once per observed stationary interval.

    A stationary interval is an observed event, not an identifier for a planned
    pose. A motion coordinator may attach an explicit pose ID only together with
    arrival evidence. Missing board detections are retained as independent raw
    attempts and never prevent the other camera from accepting an observation.
    """

    def __init__(self, output, robot_config, camera_configs, board_config, *,
                 max_attempts=3, retry_interval_s=.75, fk=None):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('max_attempts must be a positive integer')
        if not np.isfinite(retry_interval_s) or retry_interval_s <= 0:
            raise ValueError('retry_interval_s must be positive and finite')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.robot_config, self.camera_configs = robot_config, camera_configs
        self.board_config, self.fk = board_config, fk
        self.max_attempts, self.retry_interval_s = max_attempts, retry_interval_s
        self.interval_id = 0
        self.interval_q = None
        self.was_stationary = False
        self.last_evaluated_receive_s = -np.inf
        self.context = (None, None)
        self.attempts = {}
        self.camera_identities = {}
        self.last_status = {}
        self.stance_monitor = None
        self.session = dict(
            schema_version=1, kind='read_only_dual_camera_observations',
            status='recording', candidate_only=True, calibration_ready=False,
            control_commands_sent=0, read_only=True,
            created_utc=datetime.now(timezone.utc).isoformat(),
            robot_config=robot_config, camera_configs=camera_configs,
            board_config=board_config, board_sha256=canonical_digest(board_config),
            pose_association='Observed stationary intervals; no planned pose is inferred',
            reference_frame=robot_config.get('base_frame'),
            geometry_note='SONIC balancing can move pelvis relative to the board; '
                          'a configured foot reference requires an independently fixed foot. '
                          'Receive-time stationarity and quiet IMU alone do not prove it.',
            cameras={name: dict(samples=[], quality_passed_count=0,
                               rejected_attempt_count=0,
                               missing_calibration_evidence=calibration_evidence_failures(
                                   robot_config, config, board_config))
                     for name, config in camera_configs.items()},
            errors=[],
        )
        self.flush()

    def set_pose_context(self, pose_id=None, *, arrival_evidence=None):
        if pose_id is not None:
            if type(pose_id) is not int or pose_id <= 0:
                raise ValueError('Explicit pose ID must be a positive integer')
            if not isinstance(arrival_evidence, str) or not arrival_evidence.strip():
                raise ValueError('Explicit pose IDs require external arrival evidence')
        elif arrival_evidence is not None:
            raise ValueError('Arrival evidence requires an explicit pose ID')
        context = (pose_id, arrival_evidence)
        if context != self.context:
            self.was_stationary = False
            self.context = context

    def _identity(self, name, frame):
        signature = (frame.serial, frame.optical_frame, frame.bgr.shape,
                     frame.K.tolist(), frame.distortion.tolist())
        if name in self.camera_identities and self.camera_identities[name] != signature:
            raise ValueError('Camera identity, image profile, or calibration changed')
        self.camera_identities[name] = signature

    def observe(self, name, packet, state_buffer, *, now=None):
        if name not in self.camera_configs:
            raise ValueError('Unknown camera name')
        now = time.monotonic() if now is None else float(now)
        frame = packet.frame
        self._identity(name, frame)
        try:
            state, receive_timing = receive_time_stationarity(
                state_buffer, frame.received_monotonic_s, self.robot_config, now=now)
            stance = None
            policy = self.robot_config.get('stance_monitor', {})
            if policy.get('enabled'):
                if self.fk is None or self.fk.base != 'left_ankle_roll_link':
                    raise ValueError('Fixed stance monitoring requires left-foot FK')
                if self.stance_monitor is None:
                    self.stance_monitor = FixedStanceMonitor(self.fk.path,
                        self.robot_config['joint_indices'], state.q,
                        translation_limit_m=policy['translation_limit_m'],
                        rotation_limit_deg=policy['rotation_limit_deg'])
                with state_buffer.lock:
                    monitored = [value for value in state_buffer.states if
                        receive_timing['state_receive_interval_s'][0] <= value.received_s
                        <= receive_timing['state_receive_interval_s'][1]]
                checks = self.stance_monitor.check_many([value.q for value in monitored]) if monitored else []
                if not checks or not all(check['passed'] for check in checks):
                    raise ValueError('Relative foot stance changed in the observation interval')
                stance = dict(passed=True, checked_states=len(checks),
                    maximum_relative_translation_change_m=max(check['relative_foot_translation_change_m'] for check in checks),
                    maximum_relative_rotation_change_deg=max(check['relative_foot_rotation_change_deg'] for check in checks),
                    common_foot_slip_observable=False)
        except ValueError as exc:
            if frame.received_monotonic_s >= self.last_evaluated_receive_s:
                self.was_stationary = False
                self.last_evaluated_receive_s = frame.received_monotonic_s
            self.last_status[name] = str(exc)
            return None
        threshold = float(self.robot_config['max_joint_range_rad'])
        same_interval = (self.interval_q is not None and
            (measure_mount_stability(np.stack([state.q, self.interval_q]), self.robot_config)['passed']
             if stability_policy(self.robot_config) is not None
             else np.max(np.abs(state.q - self.interval_q)) <= threshold))
        if frame.received_monotonic_s < self.last_evaluated_receive_s:
            if not self.was_stationary or not same_interval:
                self.last_status[name] = 'Older frame does not belong to the current stationary interval'
                return None
        if not self.was_stationary or not same_interval:
            self.interval_id += 1
            self.interval_q = state.q.copy()
            self.was_stationary = True
        self.last_evaluated_receive_s = max(self.last_evaluated_receive_s,
                                            frame.received_monotonic_s)
        key = (self.interval_id, name)
        attempt = self.attempts.get(key, dict(count=0, passed=False, last_received_s=-np.inf))
        if (attempt['passed'] or attempt['count'] >= self.max_attempts
                or frame.received_monotonic_s - attempt['last_received_s'] < self.retry_interval_s):
            return None
        aligned_timing = None
        aligned_state = None
        alignment_error = None
        try:
            aligned_state, aligned_timing = state_buffer.settled_at(
                frame, self.robot_config, self.camera_configs[name])
        except ValueError as exc:
            alignment_error = str(exc)
        index = len(self.session['cameras'][name]['samples'])
        directory = self.output / name
        directory.mkdir(exist_ok=True)
        image_name = f'{index:04d}.png'
        temporary_image = directory / f'{index:04d}.tmp.png'
        if not cv2.imwrite(str(temporary_image), frame.bgr):
            raise OSError('Could not save raw observation image')
        temporary_image.replace(directory / image_name)
        quality_passed = packet.observation is not None
        pose_id, arrival_evidence = self.context
        record = dict(
            index=index, image=str(Path(name) / image_name),
            stationary_interval_id=self.interval_id,
            planned_pose_id=pose_id, external_arrival_evidence=arrival_evidence,
            candidate_only=True, calibration_ready=False,
            observation_quality_passed=quality_passed,
            observation_rejection_reason=packet.rejection_reason,
            observation=packet.observation,
            source_timestamp_s=frame.source_timestamp_s,
            source_clock=frame.source_clock, received_monotonic_s=frame.received_monotonic_s,
            capture_monotonic_s=frame.capture_monotonic_s,
            timing_uncertainty_s=frame.timing_uncertainty_s,
            sequence=frame.sequence, serial=frame.serial,
            optical_frame=frame.optical_frame, K=frame.K, distortion=frame.distortion,
            q=state.q, dq=state.dq, tick=state.tick,
            mode_pr=state.mode_pr, mode_machine=state.mode_machine,
            state_received_s=state.received_s,
            receive_time_stationarity=receive_timing,
            stance_monitoring=stance,
            bounded_capture_alignment=aligned_timing,
            capture_alignment_failure=alignment_error,
            capture_aligned_state=None if aligned_state is None else dict(
                q=aligned_state.q.copy(), dq=aligned_state.dq.copy(),
                tick=aligned_state.tick, mode_pr=aligned_state.mode_pr,
                mode_machine=aligned_state.mode_machine,
                imu_gyro=aligned_state.imu_gyro.copy(),
                state_received_s=aligned_state.received_s),
        )
        if self.fk is not None:
            record['kinematic_reference_frame'] = self.fk.base
            record['T_reference_mount_from_receive_time_state'] = self.fk.compute(state.q)
            record['urdf_path'] = str(self.fk.path.resolve())
            record['urdf_sha256'] = self.fk.sha256
            record['mount_frame'] = self.fk.mount
            if aligned_state is not None:
                record['T_reference_mount_from_capture_aligned_state'] = self.fk.compute(aligned_state.q)
        _atomic_yaml(directory / f'{index:04d}.yaml', record)
        summary = self.session['cameras'][name]
        summary['samples'].append(record)
        summary['quality_passed_count'] += int(quality_passed)
        summary['rejected_attempt_count'] += int(not quality_passed)
        self.attempts[key] = dict(count=attempt['count'] + 1, passed=quality_passed,
                                  last_received_s=frame.received_monotonic_s)
        self.last_status[name] = ('Board geometry passed; candidate only' if quality_passed
                                  else packet.rejection_reason or 'Board pose rejected')
        self.flush()
        return record

    def flush(self):
        self.session['last_status'] = dict(self.last_status)
        _atomic_yaml(self.output / 'observations.yaml', self.session)

    def finish(self, status='complete', error=None):
        self.session['status'] = status
        if error:
            self.session['errors'].append(str(error))
        self.session['finished_utc'] = datetime.now(timezone.utc).isoformat()
        self.flush()


def overlay_packet(name, packet, status, *, width=960, read_only=True):
    """Draw measured target features and English diagnostic text for one camera."""
    view = packet.frame.bgr.copy()
    observation = packet.observation
    points = np.empty((0, 2))
    if observation is not None:
        points = np.asarray(observation.get('image_points_px', []), float).reshape(-1, 2)
        for point in points:
            cv2.circle(view, tuple(np.rint(point).astype(int)), 4, (0, 220, 0), -1)
        matrix = observation.get('T_camera_board')
        if matrix is not None:
            matrix = np.asarray(matrix, float)
            rvec = cv2.Rodrigues(matrix[:3, :3])[0]
            cv2.drawFrameAxes(view, packet.frame.K, packet.frame.distortion,
                             rvec, matrix[:3, 3], .10)
    scale = min(1., float(width) / view.shape[1])
    if scale < 1.:
        view = cv2.resize(view, None, fx=scale, fy=scale)
    quality = (f'Board OK | corners={len(points)} | RMS={observation["reprojection_rms_px"]:.3f} px'
               if observation is not None else packet.rejection_reason or 'No accepted board pose')
    text = [f'{name}  serial={packet.frame.serial}  frame={packet.frame.sequence}',
            quality,
            ('READ ONLY | ' if read_only else 'AUTO CAPTURE | ') +
            (status or ('No control commands' if read_only else 'Waiting for pose'))]
    # Keep the status panel away from an accepted board in the upper image.
    top = view.shape[0] - 84 if len(points) and np.mean(points[:, 1]) < packet.frame.bgr.shape[0] / 2 else 0
    cv2.rectangle(view, (0, top), (view.shape[1], top + 84), (24, 24, 24), -1)
    for index, line in enumerate(text):
        cv2.putText(view, line[:130], (12, top + 22 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
    return view
