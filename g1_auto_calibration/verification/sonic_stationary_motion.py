"""Design and measure a 120 s SONIC calibration candidate without robot I/O."""

import importlib.util
import json
from pathlib import Path
import sys
from collections import deque
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ..common import ROOT, digest, load_yaml, write_yaml
from .sonic_policy import DEFAULT_SONIC_ROOT, _no_transport, _write_array, analyze_feet


def prepare(output, *, yaw_deg=6., roll_deg=4., pitch_deg=4., sonic_root=DEFAULT_SONIC_ROOT,
            motion_kind='holds'):
    """Keep default lower12/arms as references, and visit18 multi-axis targets.

    Twenty blocks include initial/final neutral holds. Each block has a
    zero-endpoint quintic transition and a hold; total frame span is120 s.
    Leg references being constant is not a constraint on the policy's output.
    """
    amplitudes = np.asarray([yaw_deg, roll_deg, pitch_deg], float)
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0) or np.max(amplitudes) > 15:
        raise ValueError('Require finite nonnegative waist amplitudes at most15 degrees')
    if motion_kind not in ('holds', 'sine'):
        raise ValueError('Unknown reference motion kind')
    output, sonic_root = Path(output).resolve(), Path(sonic_root).resolve()
    if not output.is_relative_to(ROOT.parent.resolve()):
        raise ValueError('Output must remain in the calibration workspace')
    output.mkdir(parents=True, exist_ok=False)
    profile_path = sonic_root/'robot_decoder/profiles/g1_29_wave_control.json'
    profile = json.loads(profile_path.read_text())
    default = np.asarray(profile['default_angles_hw'], float)
    grid = [(y, r) for j, r in enumerate((-1., 0., 1.))
            for y in ((-1., 0., 1.) if j % 2 == 0 else (1., 0., -1.))]
    if yaw_deg == 0:
        grid = [(0., r) for r in np.linspace(-1., 1., 9)]
    targets = [np.zeros(3)]
    targets += [np.deg2rad(amplitudes)*[y, r, -1.] for y, r in grid]
    targets += [np.deg2rad(amplitudes)*[y, r, 1.] for y, r in reversed(grid)]
    targets += [np.zeros(3)]
    dt, speed = .02, .14
    previous = np.zeros(3)
    transitions = []
    for target in targets:
        transitions.append(int(np.ceil(max(2., 1.875*float(np.max(np.abs(target-previous)))/speed)/dt)))
        previous = target
    # Keep the first six seconds neutral. Allocate the remaining time across
    # slow transitions and equal holds, without increasing the requested speed.
    remaining = 6000-300-sum(transitions[1:])
    hold_frames, extra = divmod(remaining, len(targets)-1)
    if hold_frames < 50:
        raise ValueError('Requested amplitudes leave less than1 s stationary hold per block')
    q = np.tile(default, (6001, 1))
    holds, previous, start = [], np.zeros(3), 0
    for i, target in enumerate(targets):
        transition_frames = transitions[i]
        length = 300 if i == 0 else transition_frames+hold_frames+(i <= extra)
        end = start+length
        u = np.minimum(np.arange(length)*dt/(transition_frames*dt), 1.)
        blend = 10*u**3-15*u**4+6*u**5
        q[start:end, 12:15] = default[12:15]+previous+blend[:, None]*(target-previous)
        holds.append(dict(pose_id=i+1, reference_start_frame=start+transition_frames,
                          reference_end_frame=end, target_waist_rad=(default[12:15]+target).tolist()))
        previous = target
        start = end
    if start != 6000:
        raise AssertionError('Motion allocation must span exactly120 s')
    q[-1] = default
    if motion_kind == 'sine':
        t = np.arange(6001)*dt
        u = np.clip((t-6.)/108., 0., 1.)
        fade = np.minimum(np.clip((t-6.)/6., 0., 1.), np.clip((114.-t)/6., 0., 1.))
        envelope = 10*fade**3-15*fade**4+6*fade**5
        waves = np.c_[np.sin(2*np.pi*u), np.sin(4*np.pi*u), np.sin(6*np.pi*u)]
        q[:, 12:15] = default[12:15]+envelope[:, None]*waves*np.deg2rad(amplitudes)
        q[-1] = default
        holds = [dict(pose_id=i+1, reference_start_frame=(i+1)*300-50,
            reference_end_frame=(i+1)*300, target_waist_rad=q[(i+1)*300-1, 12:15].tolist(),
            phase='continuous_motion_candidate_window') for i in range(20)]
    return _write_reference(output, q, holds, profile, profile_path, speed=speed, dt=dt,
        reference_name='calibration_120s', manifest_fields=dict(
            kind='two_minute_stationary_calibration_reference', duration_s=120.,
            amplitude_deg=dict(zip(('yaw', 'roll', 'pitch'), amplitudes.tolist())), motion_kind=motion_kind))


def _write_reference(output, q, holds, profile, profile_path, *, speed, dt, reference_name, manifest_fields):
    """Check limits, write the encoder reference and its manifest (shared by all motions)."""
    limits = np.c_[profile['lower_hw'], profile['upper_hw']]
    velocity = np.gradient(q, dt, axis=0, edge_order=2)
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError('Reference exceeds the unchanged joint limits')
    if np.any(np.abs(velocity) > np.minimum(profile['velocity_hw'], speed)+1e-10):
        raise ValueError('Reference exceeds unchanged mechanical/requested velocity limits')
    gather = np.asarray(profile['mujoco_to_isaaclab'], int)
    if sorted(gather.tolist()) != list(range(29)):
        raise ValueError('Profile joint mapping is not a29-joint permutation')
    reference = output/'references'/reference_name
    reference.mkdir(parents=True)
    _write_array(reference/'joint_pos.csv', q[:, gather])
    _write_array(reference/'joint_vel.csv', velocity[:, gather])
    _write_array(reference/'body_quat.csv', np.tile([1., 0., 0., 0.], (len(q), 1)))
    (reference/'metadata.txt').write_text(f'Total timesteps: {len(q)}\nFPS: 50\nEncode mode: 0 (G1)\n')
    (reference/'conversion.json').write_text(json.dumps(dict(frames=len(q), target_fps=50,
        joint_order='isaaclab', quaternion_order='wxyz', encode_mode=0, body_indexes=[0],
        simulation_only=True, hardware_eligible=False), indent=2)+'\n')
    np.savez_compressed(output/'reference_sdk.npz', q_sdk_rad=q, route_start_frame=0)
    frozen = np.r_[0:12, 15:29]
    default = np.asarray(profile['default_angles_hw'], float)
    manifest = dict(schema_version=1, fps=50, reference_frames=len(q), holds=holds,
        joint_order='SDK29 before exact profile mapping', profile=str(profile_path),
        profile_sha256=digest(profile_path), reference_q_rad=default.tolist(),
        frozen_reference_indices=frozen.tolist(), lower_body_reference_fixed=True,
        arms_reference_fixed=True, actual_lower_body_fixed=False,
        actual_lower_body_constraint_enforced=False, root_orientation_reference='identity, intended upright',
        suspension_after_startup=False, simulation_only=True, hardware_eligible=False,
        robot_connections_opened=0, robot_control_commands_sent=0,
        source_sha256=digest(Path(__file__)),
        note='The original SONIC policy is unchanged; no joints, floor contacts or base are artificially locked.',
        **manifest_fields)
    write_yaml(output/'prepared_reference.yaml', manifest)
    return reference, profile_path, manifest


def prepare_pose_list(output, poses_deg, *, hold_s=2., initial_s=6., max_abs_deg=20.,
                      sonic_root=DEFAULT_SONIC_ROOT, speed=.14):
    """Visit an ordered list of waist [yaw, roll, pitch] targets (degrees from SONIC default).

    Each pose gets a zero-endpoint quintic transition no faster than ``speed`` and a
    stationary hold; the reference starts and ends at the default posture. Pose IDs
    are the 1-based positions in the operator's list.
    """
    poses = np.asarray(poses_deg, float)
    if poses.ndim != 2 or poses.shape[1] != 3 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Need a nonempty list of finite [yaw, roll, pitch] poses in degrees')
    if np.max(np.abs(poses)) > max_abs_deg:
        raise ValueError(f'Waist poses must stay within {max_abs_deg:g} degrees of the default posture')
    if hold_s < 1. or initial_s < 3.:
        raise ValueError('Need at least 1 s holds and 3 s of initial neutral standing')
    output, sonic_root = Path(output).resolve(), Path(sonic_root).resolve()
    if not output.is_relative_to(ROOT.parent.resolve()):
        raise ValueError('Output must remain in the calibration workspace')
    output.mkdir(parents=True, exist_ok=False)
    profile_path = sonic_root/'robot_decoder/profiles/g1_29_wave_control.json'
    profile = json.loads(profile_path.read_text())
    default = np.asarray(profile['default_angles_hw'], float)
    dt = .02
    hold_frames, first = int(round(hold_s/dt)), int(round(initial_s/dt))
    targets = [*np.deg2rad(poses), np.zeros(3)]  # final return to the default posture
    pieces, holds, previous, start = [np.tile(default, (first, 1))], [], np.zeros(3), first
    for i, target in enumerate(targets):
        transition = int(np.ceil(max(2., 1.875*float(np.max(np.abs(target-previous)))/speed)/dt))
        length = transition+hold_frames
        u = np.minimum(np.arange(length)/transition, 1.)
        blend = 10*u**3-15*u**4+6*u**5
        block = np.tile(default, (length, 1))
        block[:, 12:15] = default[12:15]+previous+blend[:, None]*(target-previous)
        pieces.append(block)
        if i < len(poses):
            holds.append(dict(pose_id=i+1, target_deg=poses[i].tolist(),
                              reference_start_frame=start+transition, reference_end_frame=start+length,
                              target_waist_rad=(default[12:15]+target).tolist()))
        previous, start = target, start+length
    q = np.vstack([*pieces, default[None]])
    duration = (len(q)-1)*dt
    return _write_reference(output, q, holds, profile, profile_path, speed=speed, dt=dt,
        reference_name=f'calibration_{len(poses)}_poses', manifest_fields=dict(
            kind='pose_list_calibration_reference', duration_s=duration, pose_count=len(poses),
            poses_deg=poses.tolist(), hold_s=float(hold_s), max_speed_rad_s=speed, motion_kind='pose_list',
            amplitude_deg=dict(zip(('yaw', 'roll', 'pitch'), np.max(np.abs(poses), axis=0).tolist()))))


def _diameter(poses):
    p = np.asarray(poses)
    distance = np.linalg.norm(p[:, None, :3, 3]-p[None, :, :3, 3], axis=-1)
    relative = np.swapaxes(p[:, :3, :3], -1, -2)[:, None] @ p[:, :3, :3][None]
    angles = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    return float(distance.max()), float(np.rad2deg(angles.max()))


def analyze(output, sonic_root, manifest, policy):
    """Assess real simulated torso/feet, including an explicitly synthetic camera.

    Virtual camera data use MuJoCo world transforms, independently of the FK
    fixed-foot assumption being tested. No real images/intrinsics are invented.
    """
    import mujoco as mj
    from ..calibration.forward_kinematics import ForwardKinematics
    from ..calibration.transforms import inverse, transform, pose_errors
    from ..calibration.handeye_solver import solve, observability
    from ..calibration.validator import validate
    import cv2

    output, sonic_root = Path(output), Path(sonic_root)
    rows = np.loadtxt(output/'dynamics/simulation.csv', delimiter=',', skiprows=1, ndmin=2)
    profile = json.loads(Path(manifest['profile']).read_text())
    robot = load_yaml(ROOT/'config/robot_sonic.yaml')
    fk = ForwardKinematics(ROOT/'config/g1_29dof.urdf', 'left_ankle_roll_link',
                           'torso_link', robot['joint_indices'])
    model = mj.MjModel.from_xml_path(str(sonic_root/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'))
    data = mj.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in profile['joint_names_hw']]
    torso_id = model.body('torso_link').id
    world_torso = np.tile(np.eye(4), (len(rows), 1, 1))
    for i, row in enumerate(rows):
        data.qpos[:7], data.qpos[addresses] = row[2:9], row[9:38]
        mj.mj_forward(model, data)
        world_torso[i, :3, :3] = data.xmat[torso_id].reshape(3, 3)
        world_torso[i, :3, 3] = data.xpos[torso_id]
    # Explicitly synthetic forward/downward optical mount and ideal board.
    optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    X = transform(Rotation.from_euler('y', 30., degrees=True).as_matrix()@optical, [.06, 0., .3])
    K = np.array([[580., 0., 320.], [0., 580., 240.], [0., 0., 1.]])
    points = np.array([[i*.115, j*.115, 0.] for j in range(1, 4) for i in range(1, 9)])
    G, C, observations, samples = [], [], [], []
    Yworld = None
    for hold in manifest['holds']:
        end = hold['reference_end_frame']
        selected = np.flatnonzero((rows[:, 1] >= end-50) & (rows[:, 1] < end))
        if len(selected) < 45:
            continue
        index = int(selected[len(selected)//2])
        world_cam = world_torso[selected]@X
        relative_cam = fk.compute_many(rows[selected, 9:38])@X
        real_m, real_deg = _diameter(world_cam)
        fk_m, fk_deg = _diameter(relative_cam)
        stable = max(real_m, fk_m) <= .005 and max(real_deg, fk_deg) <= .5
        samples.append(dict(pose_id=hold['pose_id'], row_index=index, stable=stable,
            world_camera_window_diameter_m=real_m, world_camera_window_diameter_deg=real_deg,
            fk_camera_window_diameter_m=fk_m, fk_camera_window_diameter_deg=fk_deg))
        if not stable:
            continue
        g = fk.compute(rows[index, 9:38])
        if Yworld is None:
            Yworld = world_torso[index]@X@transform(t=[-.5175, -.23, 1.5])
        c = inverse(world_torso[index]@X)@Yworld
        pixels = cv2.projectPoints(points, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], K, np.zeros(5))[0].reshape(-1, 2)
        G.append(g); C.append(c)
        observations.append(dict(object_points_m=points.tolist(), image_points_px=pixels.tolist(),
                                 K=K.tolist(), distortion=[0.]*5))
    calibration = dict(passed=False, synthetic=True, ideal_noiseless_images=True,
                       actual_camera_or_board_visibility_validated=False, input_sample_count=len(G))
    try:
        if len(G) < 15:
            raise ValueError('Fewer than15 stable candidate poses for the existing solver')
        calibration['observability'] = observability(G, C)
        x, y, quality = solve(G, C)
        v = validate(G, C, x, y, quality['held_out_indices'], observations)
        mm, deg = pose_errors(X, [x])
        calibration.update(passed=bool(v['passed'] and mm[0] <= 5. and deg[0] <= .5),
            heldout_consistency_passed=bool(v['passed']), validation=v,
            truth_error_check='Optimistic synthetic design check against existing5mm/0.5deg scale; not real-device validation',
            known_injected_extrinsic_error_mm=float(mm[0]), known_injected_extrinsic_error_deg=float(deg[0]),
            T_mount_camera_truth=X.tolist(), T_mount_camera_estimated=x.tolist(), solver=quality)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        calibration['reason'] = str(exc)
    write_yaml(output/'synthetic_camera_check.yaml', calibration)
    feet = analyze_feet(output, sonic_root)
    left = feet.get('world_foot_results', {}).get('left_ankle_roll_link', {})
    policy_rows = rows[rows[:, 1] > 150]
    if not len(policy_rows):
        policy_rows = rows
    legs = policy_rows[:, 9:21]
    result = dict(kind='two_minute_sonic_dynamics_diagnostic', simulation_only=True,
        reference_duration_s=120., target_hold_count=20, complete_hold_count=len(samples),
        stable_candidate_count=sum(s['stable'] for s in samples), sampling_windows=samples,
        sampling_budgets=dict(translation_m=.005, rotation_deg=.5, window_s=1.,
            source='Previously accepted project calibration budgets; not URDF limits'),
        actual_leg_joint_range_deg=np.rad2deg(np.ptp(legs, axis=0)).tolist(),
        maximum_actual_leg_joint_range_deg=float(np.rad2deg(np.ptp(legs, axis=0)).max()),
        reference_left_foot=left, synthetic_camera_check_file=str(output/'synthetic_camera_check.yaml'),
        synthetic_camera_check_passed=calibration['passed'],
        synthetic_translation_rms_mm=calibration.get('validation', {}).get('translation_rms_mm'),
        synthetic_rotation_rms_deg=calibration.get('validation', {}).get('rotation_rms_deg'),
        synthetic_known_extrinsic_error_mm=calibration.get('known_injected_extrinsic_error_mm'),
        synthetic_known_extrinsic_error_deg=calibration.get('known_injected_extrinsic_error_deg'),
        policy_reference_complete=policy.get('reference_completed', False), falls=policy.get('falls'),
        post_run_return_complete=policy.get('balanced_return_confirmed', False),
        lower_body_fixed_in_actual_execution=False, hardware_execution_authorized=False,
        robot_connections_opened=0, robot_control_commands_sent=0)
    write_yaml(output/'motion_assessment.yaml', result)
    return result


@contextmanager
def constrained_output(mode):
    """Offline-only motor-target constraint, before the ORIGINAL torque guard.

    The unchanged guard reconstructs the applied action from these targets;
    decoder history therefore sees what the simulated motors actually receive.
    This alters the controller, not the physics, and has no hardware entry.
    """
    if mode == 'none':
        yield
        return
    if mode not in ('legs', 'nonwaist', 'settled_legs', 'settled_nonwaist'):
        raise ValueError('Unknown offline motor-target constraint')
    from sonic_robot_decoder import control
    if mode.startswith('settled_'):
        parent = control.GuardedController
        indices = np.arange(12) if mode == 'settled_legs' else np.r_[0:12, 15:29]

        class SettledTargetController(parent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._neutral_commands = deque(maxlen=50)
                self._fixed_neutral_target = None

            def update(self, state, now, **kwargs):
                command = super().update(state, now, **kwargs)
                if command is None or command.damping or not self.requires_decoder:
                    return command
                if self._fixed_neutral_target is None:
                    self._neutral_commands.append(command.q_target_hw.copy())
                    # First six reference seconds are neutral. Capture the
                    # preceding one-second actuator-target mean at policy6 s,
                    # after the one-second handoff and five neutral seconds.
                    if self.phase == 'POLICY' and now-self._policy_at >= 6. and len(self._neutral_commands) == 50:
                        self._fixed_neutral_target = np.mean(self._neutral_commands, axis=0)
                    else:
                        return command
                target = command.q_target_hw.copy().astype(float)
                target[indices] = self._fixed_neutral_target[indices]
                previous = self.last_command.q_target_hw if self.last_command is not None else state['q_hw']
                dt = .020 if self._last_command_at is None else min(.020, now-self._last_command_at)
                constrained = control.project_targets(self.profile, target, state['q_hw'], state['dq_hw'], previous,
                    kp_hw=command.kp_hw, kd_hw=command.kd_hw, dt=dt)
                self._proposal, self._proposal_at = constrained, now
                return constrained

        with patch.object(control, 'GuardedController', SettledTargetController):
            yield
        return
    original = control.targets_from_action
    indices = np.arange(12) if mode == 'legs' else np.r_[0:12, 15:29]

    def targets(profile, residual, weight=1.):
        target = original(profile, residual, weight)
        target[indices] = np.asarray(profile['default_angles_hw'])[indices]
        return target

    with patch.object(control, 'targets_from_action', targets):
        yield


def run(output, *, yaw_deg=6., roll_deg=4., pitch_deg=4., sonic_root=DEFAULT_SONIC_ROOT,
        fixed_targets='none', motion_kind='holds'):
    sys.dont_write_bytecode = True
    reference, profile, manifest = prepare(output, yaw_deg=yaw_deg, roll_deg=roll_deg,
                                           pitch_deg=pitch_deg, sonic_root=sonic_root, motion_kind=motion_kind)
    sonic_root, output = Path(sonic_root).resolve(), Path(output).resolve()
    sys.path[:0] = [str(sonic_root/'robot_decoder/tools'), str(sonic_root/'robot_decoder/src')]
    helper_path = sonic_root/'robot_decoder/tools/preview_wave.py'
    spec = importlib.util.spec_from_file_location('_stationary_sonic_preview', helper_path)
    helper = importlib.util.module_from_spec(spec)
    with _no_transport():
        spec.loader.exec_module(helper)
        with constrained_output(fixed_targets):
            policy = helper.run_preview(reference, output/'dynamics', profile=profile,
                encoder_model=sonic_root/'gear_sonic_deploy/policy/release/model_encoder.onnx',
                decoder_model=sonic_root/'gear_sonic_deploy/policy/release/model_decoder.onnx',
                headless=True, realtime=False, record_video=False)
        assessment = analyze(output, sonic_root, manifest, policy)
    assessment.update(offline_motor_target_constraint=fixed_targets,
        stock_sonic_controller=fixed_targets == 'none',
        modified_controller_hardware_implemented=False)
    write_yaml(output/'motion_assessment.yaml', assessment)
    report = dict(simulation_only=True, robot_connections_opened=0, robot_control_commands_sent=0,
        hardware_execution_authorized=False, controller_validated=False,
        physical_safety_validated=False, status='offline_motion_candidate',
        prepared_reference=manifest, policy_report=policy, assessment=assessment,
        offline_motor_target_constraint=fixed_targets,
        relative_stance_checks_passed=False, helper_sha256=digest(helper_path))
    write_yaml(output/'verification.yaml', report)
    return report
