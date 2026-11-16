"""A source-pinned, two-minute SONIC trial with independent raw acquisition.

Preparation and simulation never import the SDK or open robot transports.
Only the upstream robot-local decoder owns the motor command publisher.
"""

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

from ..common import ROOT, digest, load_yaml, write_yaml
from ..verification.sonic_policy import DEFAULT_SONIC_ROOT, _no_transport
from ..verification.sonic_stationary_motion import prepare, prepare_pose_list


def runtime(sonic_root=DEFAULT_SONIC_ROOT):
    sys.dont_write_bytecode = True
    root = Path(sonic_root).resolve()
    sys.path[:0] = [str(root/'robot_decoder/tools'), str(root/'robot_decoder/src')]
    modules = [importlib.import_module(name) for name in ('run_wave', 'preview_wave', 'deploy')]
    if any(Path(m.__file__).resolve().parent != root/'robot_decoder/tools' for m in modules):
        raise ValueError('A different SONIC checkout was already loaded')
    return modules


DEFAULT_POSES = ROOT/'config/calibration_poses_40.yaml'


def prepare_trial(output, *, sonic_root=DEFAULT_SONIC_ROOT, amplitude_deg=15., standby=False,
                  poses=DEFAULT_POSES):
    """Build a genuine source CSV, exact reference and dedicated deployment bundle.

    ``poses`` is a YAML pose list (default: the operator's 40 poses). ``poses=None``
    selects the older 120 s yaw/roll/pitch grid with ``amplitude_deg``.
    """
    output, sonic_root = Path(output).resolve(), Path(sonic_root).resolve()
    if not output.is_relative_to(ROOT.parent.resolve()):
        raise ValueError('Trial output must remain in the calibration workspace')
    output.mkdir(parents=True, exist_ok=False)
    if poses is None:
        reference, original, motion = prepare(output/'motion', sonic_root=sonic_root,
            yaw_deg=amplitude_deg, roll_deg=amplitude_deg, pitch_deg=amplitude_deg)
    else:
        plan = load_yaml(poses)
        if plan.get('units') != 'deg' or plan.get('order') != ['yaw', 'roll', 'pitch']:
            raise ValueError('Pose list must give [yaw, roll, pitch] in degrees')
        reference, original, motion = prepare_pose_list(output/'motion', plan['poses'],
            hold_s=float(plan.get('hold_s', 2.)), sonic_root=sonic_root)
    duration = float(motion['duration_s'])
    name = f"waist_{motion.get('pose_count', 20)}_poses_{duration:.0f}s"
    profile = json.loads(original.read_text())
    with np.load(output/'motion/reference_sdk.npz', allow_pickle=False) as arrays:
        q = arrays['q_sdk_rad']
    # Source has SDK joint order, explicit root XYZ/WXYZ, and a documented rate.
    # Root position is reference metadata, never a measured world trajectory.
    sources = output/'motion/sources'
    sources.mkdir()
    source = sources/f'{name}.csv'
    root = np.tile([0., 0., .79, 1., 0., 0., 0.], (len(q), 1))
    np.savetxt(source, np.c_[root, q], delimiter=',', fmt='%.12f', comments='',
        header=','.join(['root_x', 'root_y', 'root_z', 'root_qw', 'root_qx', 'root_qy',
                         'root_qz', *profile['joint_names_hw']]))
    write_yaml(sources/'source.yaml', dict(fps=50, frames=len(q), duration_s=duration,
        joint_order='SDK29', source_sha256=digest(source),
        root_xyz_is_measured=False, root_world_motion_compensation_available=False))
    profile['profile_name'] = f'g1_29_{name}'
    profile['motion_source_sha256'] = digest(source)
    profile['trial_purpose'] = 'operator observation and raw candidate data; no automatic calibration'
    if standby:
        profile['profile_name'] += '_balanced_standby'
        profile['standby_session'] = True
        profile['startup_completion'] = 'reference_ramp_complete'
        profile['observation_start_mode'] = 'automatic_after_qualified_standby'
    profile_path = output/'profile.json'
    profile_path.write_text(json.dumps(profile, indent=2)+'\n')
    helper, _, deploy = runtime(sonic_root)
    helper.validate_encoder_source(profile_path, helper.ENCODER, hardware=True, source=source)
    bundle = deploy.build_bundle(helper.DECODER, profile_path, helper.FIXTURE)
    (output/'decoder_bundle.tar').write_bytes(bundle.archive)
    (output/'bundle_manifest.json').write_text(json.dumps(bundle.manifest, indent=2)+'\n')
    files = [*reference.iterdir(), original, source, sources/'source.yaml', profile_path,
        output/'decoder_bundle.tar', output/'bundle_manifest.json',
        output/'motion/reference_sdk.npz', output/'motion/prepared_reference.yaml',
        Path(__file__), Path(__file__).parents[1]/'observation/continuous_video.py',
        Path(__file__).parents[1]/'execution/timing_trace.py',
        Path(__file__).parents[1]/'calibration/board_detector.py',
        Path(__file__).parents[1]/'observation/dual_capture.py',
        Path(__file__).parents[1]/'hardware/sonic_pc_session.py',
        Path(__file__).parents[1]/'hardware/robot_oak_server.py',
        Path(__file__).parents[1]/'observation/camera_view.py',
        Path(__file__).parents[1]/'verification/sonic_stationary_motion.py']
    report = dict(kind='sonic_two_minute_observation_trial', sonic_root=str(sonic_root),
        reference=str(reference), source=str(source), profile=str(profile_path),
        duration_s=duration, amplitude_deg=motion['amplitude_deg'], pose_count=len(motion['holds']),
        poses_deg=motion.get('poses_deg'), bundle_hash=bundle.bundle_hash, standby_mode=standby,
        files_sha256={str(p.resolve()): digest(p) for p in files if p.is_file()},
        reference_nonwaist_fixed=True, actual_nonwaist_fixed=False,
        controller='unmodified SONIC robot-local guarded decoder',
        automatic_calibration=False, capture_time_joint_alignment_verified=False,
        world_foot_motion_compensation_available=False, hardware_started=False,
        robot_connections_opened=0, robot_control_commands_sent=0)
    write_yaml(output/'trial.yaml', report)
    return report


def validate_trial(output):
    output = Path(output).resolve()
    report = load_yaml(output/'trial.yaml')
    if report.get('kind') != 'sonic_two_minute_observation_trial' or report.get('automatic_calibration') is not False:
        raise ValueError('Expected an observation-only trial')
    for name, expected in report['files_sha256'].items():
        if digest(name) != expected:
            raise ValueError(f'Prepared trial file changed: {name}')
    # Bind the source's actual joint contents to the exact encoder reference,
    # rather than merely assigning a new filename to an unrelated motion.
    profile = json.loads(Path(report['profile']).read_text())
    source = np.loadtxt(report['source'], delimiter=',', skiprows=1, ndmin=2)
    positions = np.loadtxt(Path(report['reference'])/'joint_pos.csv', delimiter=',', skiprows=1, ndmin=2)
    frames = int(round(float(report['duration_s'])*50))+1
    if source.shape != (frames, 36) or positions.shape != (frames, 29):
        raise ValueError(f'Trial must contain all {frames} frames at 50 Hz')
    if not np.allclose(source[:, 7:][:, profile['mujoco_to_isaaclab']], positions, atol=2e-11, rtol=0):
        raise ValueError('Motion source differs from the encoder reference')
    return report


def clear_deployed_bytecode(template, bundle_hash, *, run=subprocess.run):
    """Remove Python bytecode the robot wrote into an already-deployed bundle.

    Running the decoder creates src/sonic_robot_decoder/__pycache__/*.pyc inside
    its version directory, and upstream deploy then refuses to reuse that
    directory ("Unexpected existing bundle file"). Only *.pyc files under
    __pycache__ are removed; deploy still hash-checks every remaining file.
    """
    user = template['robot_user']
    directory = f'/home/{user}/sonic_decoder_{bundle_hash[:12]}'
    script = (f'd={shlex.quote(directory)}; [ -d "$d" ] || exit 0; '
              "find \"$d\" -path '*/__pycache__/*.pyc' -type f -delete && "
              'find "$d" -type d -name __pycache__ -empty -delete')
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5']
    if template.get('ssh_host_key_alias'):
        command += ['-o', 'HostKeyAlias='+template['ssh_host_key_alias']]
    command += [f"{user}@{template['robot_ip']}", script]
    result = run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'Could not clear bytecode in {directory} on the robot: {result.stderr.strip()}')


class CaptureSession:
    """Read-only camera/state ownership, independent of the robot motor owner."""

    def __init__(self, output, *, camera='d435', duration_s=120.):
        self.output, self.camera, self.duration_s = Path(output), camera, float(duration_s)
        self.state = self.worker = self.recording = self.server = None

    def open(self):
        from ..calibration.board_detector import RawBoardDetector
        from ..hardware.base import open_camera
        from ..hardware.robot_oak_server import start_robot_server
        from ..hardware.unitree_state import UnitreeState
        from ..observation.dual_capture import CameraWorker
        from ..observation.continuous_video import ContinuousVideo
        from ..scripts.observe_sonic import verify_wired_routes
        robot = load_yaml(ROOT/'config/robot_sonic.yaml')
        cameras = load_yaml(ROOT/'config/cameras.yaml')
        board = load_yaml(ROOT/'config/board.yaml')
        route = verify_wired_routes({self.camera: cameras[self.camera]}, robot['interface'])
        self.server = start_robot_server(cameras[self.camera])
        self.state = UnitreeState(robot)
        self.recording = ContinuousVideo(self.output, self.state, camera=self.camera,
                                         duration_s=self.duration_s).start()
        self.worker = CameraWorker(self.camera, cameras[self.camera], open_camera, RawBoardDetector(board),
                                   packet_callback=self.recording.record, encode_png=True)
        self.worker.start()
        deadline = time.monotonic()+15.
        while True:
            packet, error, _ = self.worker.snapshot()
            try:
                self.check()
                if packet is not None and packet.observation is not None:
                    break
            except RuntimeError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(error or self.state.error or 'Need fresh measured joints and a visible calibration board')
            time.sleep(.05)
        write_yaml(self.output/'configuration.yaml', dict(robot=robot,
            camera=cameras[self.camera], board=board, route=route,
            robot_config_used_for_read_only_subscription=True,
            motor_controller_profile_is_separately_pinned=True))
        return self

    def check(self):
        if self.state is None or self.worker is None or self.recording is None:
            raise RuntimeError('Raw recording has not been opened')
        _, error, received = self.worker.snapshot()
        now = time.monotonic()
        with self.state.lock:
            stale = not self.state.states or now-self.state.states[-1].received_s > .25
        failure = self.recording.error or self.recording.trace.error or self.state.error or error
        if failure or stale or received is None or now-received > 2.:
            raise RuntimeError(failure or 'Camera or measured joint stream is stale')

    def close(self):
        try:
            if self.worker is not None:
                self.worker.stop()
            if self.recording is not None:
                # Exit promptly after U: review.mp4 is rendered offline by review_sonic_observation.
                print('Saving raw recording ...', flush=True)
                return self.recording.close(render_video=False)
        finally:
            try:
                if self.state is not None:
                    self.state.close()
            finally:
                if self.server is not None:
                    self.server.stop()


class RecordingEncoder:
    """Stamp selected references in the camera/state local monotonic clock.

    This is reference timing telemetry, not proof of camera exposure timing or
    measured target arrival. Returning uses the original encoder implementation.
    """

    def __init__(self, encoder, capture, events, *, original=True, clock=time.monotonic):
        self.encoder, self.capture, self.events = encoder, capture, events
        self.original, self.clock = original, clock

    def encode(self, frame, quaternion, *, playing=True):
        self.capture.check()
        if playing and self.original and self.capture.recording.started is None:
            self.capture.recording.begin()
        self.events.append(dict(local_monotonic_s=self.clock(), source_frame=int(frame),
            playing=bool(playing), original_reference=self.original,
            measured_joint_pose=False, camera_capture_time=False))
        return self.encoder.encode(frame, quaternion, playing=playing)

    def retarget_reference(self, reference):
        return type(self)(self.encoder.retarget_reference(reference), self.capture, self.events,
                          original=False, clock=self.clock)


def ask_preview(input_fn=input):
    """Return True to preview the motion first, False to skip to execution."""
    while True:
        answer = input_fn('Preview the motion first? Enter y/Y to preview, n/N to skip the preview and execute: ').strip()
        if answer in ('y', 'Y'):
            return True
        if answer in ('n', 'N'):
            return False
        print('Please enter y/Y or n/N', flush=True)


def _normalized_digests(digests, session, trial_dir):
    session, trial_dir = str(Path(session).resolve()), str(Path(trial_dir).resolve())
    return {name.replace(session, '<SESSION>').replace(trial_dir, '<TRIAL>'): value
            for name, value in digests.items()}


def reuse_preview(session, trial_dir, trial, helper, preview, *, standby, search_root=None):
    """Reuse the newest earlier eligible preview of exactly this motion, or return None.

    Trials are generated deterministically. An earlier session previewed the same
    reference, sources, profile, models, decoder, tools and physics files exactly
    when its digests equal the current ones after each run's own trial/session
    directories are replaced by placeholders. The copied report is rebound to the
    current files, so upstream approval still re-hashes everything this run uses.
    """
    current = preview.pipeline_digests(trial['reference'], trial['profile'], helper.ENCODER, helper.DECODER)
    wanted = _normalized_digests(current, session, trial_dir)
    root = Path(search_root or ROOT/'results')
    candidates = sorted(root.glob('sonic_observation_*/sessions/*/preview/report.json'),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        old_session = path.parent.parent
        if old_session.resolve() == Path(session).resolve():
            continue
        try:
            report = json.loads(path.read_text())
            if not preview.preview_is_eligible(report) or bool(report.get('standby_mode')) != bool(standby):
                continue
            if standby:
                helper.validate_standby_preview(report)
        except (ValueError, OSError, TypeError, AttributeError):
            continue
        if _normalized_digests(report['pipeline_digests'], old_session, old_session.parent.parent) != wanted:
            continue
        reused = dict(report, pipeline_digests=current, reused_preview_report=str(path.resolve()),
                      reused_preview_report_sha256=digest(path))
        (Path(session)/'preview').mkdir(parents=True)
        (Path(session)/'preview/report.json').write_text(json.dumps(reused, indent=2)+'\n')
        return reused
    return None


def run_trial(output, *, camera='d435', hardware=False, local_ip=None,
              deployment_template=None, camera_view=True, input_fn=input):
    """Camera check -> preview (or reuse an identical one) -> onsite yes -> deployment -> trial."""
    trial = validate_trial(output)
    standby = trial.get('standby_mode', False)
    helper, preview, deploy = runtime(trial['sonic_root'])
    if hardware and (not local_ip or not deployment_template):
        raise ValueError('Hardware requires explicit local IP and a deployment template')
    session = Path(output)/'sessions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    session.mkdir(parents=True)
    # Upstream binds approval to reference.parent.parent/preview/report.json.
    # Give every attempt its own immutable reference AND source snapshots there.
    reference_path = session/'references'/Path(trial['reference']).name
    shutil.copytree(trial['reference'], reference_path)
    shutil.copytree(Path(trial['source']).parent, session/'sources')
    trial = dict(trial, reference=str(reference_path),
                 source=str(session/'sources'/Path(trial['source']).name))
    server = None
    try:
        if hardware and camera_view:
            from ..hardware.robot_oak_server import start_robot_server
            from ..observation.camera_view import show_camera_view
            cameras = load_yaml(ROOT/'config/cameras.yaml')
            # Started once here; CaptureSession reuses this endpoint and leaves it to us.
            server = start_robot_server(cameras[camera])
            if not show_camera_view(camera, cameras[camera], load_yaml(ROOT/'config/board.yaml')):
                raise RuntimeError('Camera check aborted by the operator; no robot connection was opened')
        return _run_session(output, session, trial, helper, preview, deploy, camera=camera,
                            hardware=hardware, local_ip=local_ip, deployment_template=deployment_template,
                            standby=standby, input_fn=input_fn)
    finally:
        if server is not None:
            server.stop()


def _run_session(output, session, trial, helper, preview, deploy, *, camera, hardware, local_ip,
                 deployment_template, standby, input_fn):
    report = None
    if hardware and not ask_preview(input_fn):
        report = reuse_preview(session, output, trial, helper, preview, standby=standby)
        if report is None:
            print('No successful preview of exactly this motion and controller was found; previewing first.', flush=True)
        else:
            print(f"Skipping the preview; reusing the identical successful preview {report['reused_preview_report']}", flush=True)
    if report is None:
        with _no_transport():
            report = preview.run_preview(trial['reference'], session/'preview', profile=trial['profile'],
                encoder_model=helper.ENCODER, decoder_model=helper.DECODER,
                headless=False, realtime=True, record_video=True, standby_mode=standby)
    if not hardware:
        return dict(preview=report, session=str(session), hardware_started=False)
    amplitude = trial['amplitude_deg']
    label = (f"{trial.get('pose_count', 20)} waist calibration poses (about "
             f"{float(trial['duration_s'])/60:.1f} min, maximum angles {amplitude})")
    approval = helper.approve_preview(report, session, motion_label=label, standby_mode=standby,
                                       input_fn=input_fn)
    validate_trial(output)
    helper.validate_approval(approval, trial['reference'], trial['profile'], helper.ENCODER, helper.DECODER, standby_mode=standby)
    from sonic_robot_decoder.encoder import Mode0Reference, WaveEncoder
    reference = Mode0Reference.from_directory(trial['reference'])
    encoder_hash = helper.validate_encoder_source(trial['profile'], helper.ENCODER,
                                                hardware=True, source=trial['source'])
    preloaded = WaveEncoder(helper.ENCODER, reference, [1., 0., 0., 0.], expected_sha256=encoder_hash)
    for playing in (False, True, False, True):
        preloaded.encode(0, [1., 0., 0., 0.], playing=playing)
    capture = CaptureSession(session/'recording', camera=camera, duration_s=float(trial['duration_s']))
    events, motion, archive = [], None, None
    auto_start_events = []
    failure = None
    try:
        capture.open()
        template = json.loads(Path(deployment_template).read_text())
        bundle = deploy.build_bundle(helper.DECODER, trial['profile'], helper.FIXTURE)
        if bundle.bundle_hash != trial['bundle_hash']:
            raise ValueError('Robot-local source changed since trial preparation')
        clear_deployed_bytecode(template, bundle.bundle_hash)
        deployed, _ = deploy.deploy(bundle, template['robot_ip'], template['robot_user'],
            output_config=session/'deployment.json', robot_python=template['robot_python'],
            sdk_path=template['sdk_path'], ssh_host_key_alias=template.get('ssh_host_key_alias'))

        def encoder_factory(model, ref, quaternion, *, expected_sha256):
            if preloaded.model_sha256 != expected_sha256:
                raise ValueError('Preloaded encoder hash differs')
            return RecordingEncoder(preloaded.with_initial_heading(quaternion), capture, events)

        from sonic_robot_decoder.motion_service import MotionTokenClient
        from ..hardware.sonic_pc_session import AdvancingFeedbackClient, QualifiedAutoStartKeys
        helper.validate_local_ipv4(local_ip)

        def client_factory(*args, **kwargs):
            return AdvancingFeedbackClient(MotionTokenClient(*args, **kwargs))

        def keys_factory(controls):
            return (QualifiedAutoStartKeys(controls, helper.OperatorKeys, auto_start_events)
                    if standby else helper.OperatorKeys(controls))

        motion = helper.run_remote_wave(deployed, reference=trial['reference'], output=session/'hardware',
            local_ip=local_ip, hardware=True, approval=approval, profile=trial['profile'],
            encoder_model=helper.ENCODER, decoder_model=helper.DECODER, duration=None if standby else float(trial['duration_s'])+60.,
            encoder_factory=encoder_factory, motion_source=trial['source'], standby_mode=standby,
            client_factory=client_factory, keys_factory=keys_factory)
        failure = motion.get('failure')
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            archive = capture.close()
        finally:
            (session/'reference_timing.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            write_yaml(session/'observation_trial.yaml', dict(kind='sonic_observation_trial_result',
                camera=camera, motion=motion, recording=archive, failure=failure,
                automatic_calibration=False, calibration_ready=False,
                capture_time_joint_alignment_verified=False,
                world_foot_motion_compensation_available=False,
                automatic_start_requests=auto_start_events,
                reference_timing_file='reference_timing.jsonl',
                reference_timing_sha256=digest(session/'reference_timing.jsonl')))
    return dict(session=str(session), motion=motion, recording=archive, calibration_ready=False)
