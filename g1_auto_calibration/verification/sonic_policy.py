"""Shared SONIC simulation helpers: pinned checkout, reference arrays, foot FK, no-transport guard."""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ..common import digest, load_yaml, write_yaml


DEFAULT_SONIC_ROOT = Path('/home/kaixin/GR00T-WholeBodyControl')


def _write_array(path, array):
    np.savetxt(path, array, delimiter=',', fmt='%.12g',
               header=','.join(f'value_{index}' for index in range(array.shape[1])),
               comments='')


def analyze_feet(output, sonic_root=DEFAULT_SONIC_ROOT):
    """Recover world foot poses from saved floating-base dynamics and real FK."""
    import mujoco
    from scipy.spatial.transform import Rotation
    output, sonic_root = Path(output), Path(sonic_root)
    manifest = load_yaml(output / 'prepared_reference.yaml')
    profile = json.loads(Path(manifest['profile']).read_text())
    trace = np.loadtxt(output / 'dynamics/simulation.csv', delimiter=',', skiprows=1, ndmin=2)
    scene = sonic_root / 'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    joint_ids = [model.joint(name).id for name in profile['joint_names_hw']]
    addresses = model.jnt_qposadr[joint_ids]
    names = ['left_ankle_roll_link', 'right_ankle_roll_link']
    body_ids = [model.body(name).id for name in names]
    poses = np.tile(np.eye(4), (len(trace), 2, 1, 1))
    for index, row in enumerate(trace):
        data.qpos[:7] = row[2:9]
        data.qpos[addresses] = row[9:38]
        mujoco.mj_forward(model, data)
        poses[index, :, :3, 3] = data.xpos[body_ids]
        poses[index, :, :3, :3] = data.xmat[body_ids].reshape(2, 3, 3)
    np.savez_compressed(output / 'feet_world_trace.npz', simulation_time_s=trace[:, 0],
                        reference_frame=trace[:, 1], world_T_foot=poses)
    medians, per_pose = [], []
    for hold in manifest['holds']:
        end = hold['reference_end_frame']
        selected = poses[(trace[:, 1] >= end-50) & (trace[:, 1] < end)]
        if len(selected) < 45:
            continue
        median = np.tile(np.eye(4), (2, 1, 1))
        median[:, :3, 3] = np.median(selected[:, :, :3, 3], axis=0)
        for side in range(2):
            median[side, :3, :3] = Rotation.from_matrix(selected[:, side, :3, :3]).mean().as_matrix()
        medians.append(median)
        relative = np.linalg.inv(median[0]) @ median[1]
        per_pose.append(dict(pose_id=hold['pose_id'],
            world_T_left_foot=median[0].tolist(), world_T_right_foot=median[1].tolist(),
            left_T_right_foot=relative.tolist(),
            within_hold_foot_position_range_m=np.ptp(selected[:, :, :3, 3], axis=0).tolist()))
    if not medians:
        result = dict(scope='world foot FK reconstructed from saved floating-base dynamics',
                      complete_hold_count=0, reason='No complete hold windows', per_pose=[])
        write_yaml(output / 'feet_verification.yaml', result)
        return result
    medians = np.asarray(medians)
    positions = medians[:, :, :3, 3]
    rotations = medians[:, :, :3, :3]
    relative = np.array([np.linalg.inv(pair[0]) @ pair[1] for pair in medians])
    report = {}
    for side, name in enumerate(names):
        distance = np.linalg.norm(positions[:, side] - positions[0, side], axis=1)
        angle = Rotation.from_matrix(np.swapaxes(rotations[0, side], -1, -2)
                                     @ rotations[:, side]).magnitude()
        report[name] = dict(position_range_xyz_m=np.ptp(positions[:, side], axis=0).tolist(),
            max_position_change_from_first_hold_m=float(distance.max()),
            max_rotation_change_from_first_hold_deg=float(np.rad2deg(angle).max()))
    relative_delta = np.array([np.linalg.inv(relative[0]) @ value for value in relative])
    relative_position = np.linalg.norm(relative_delta[:, :3, 3], axis=1)
    relative_angle = np.rad2deg(Rotation.from_matrix(relative_delta[:, :3, :3]).magnitude())
    common_displacement = np.mean(positions-positions[0], axis=1)
    checks = [dict(pose_id=entry['pose_id'], translation_change_mm=float(distance*1000),
                   rotation_change_deg=float(angle), passed=bool(distance <= .005 and angle <= 1.))
              for entry, distance, angle in zip(per_pose, relative_position, relative_angle)]
    result = dict(schema_version=1, simulation_only=True, robot_control_commands_sent=0,
        scope='world foot poses reconstructed from floating-base trace and the exact simulation model',
        hold_window_seconds=1., complete_hold_count=len(per_pose),
        world_foot_results=report,
        max_common_foot_translation_from_first_hold_m=float(np.linalg.norm(common_displacement, axis=1).max()),
        relative_feet_max_translation_change_m=float(relative_position.max()),
        relative_feet_max_rotation_change_deg=float(relative_angle.max()),
        relative_feet_exceed_5mm_or_1deg=bool(relative_position.max() > .005 or relative_angle.max() > 1.),
        stance_checks_against_first_hold=checks,
        relative_stance_pass_count=sum(check['passed'] for check in checks),
        first_relative_stance_failure=next((check for check in checks if not check['passed']), None),
        common_foot_drift_can_be_hidden_from_relative_monitor=True,
        foot_contact_or_slip_not_inferred_from_pelvis_motion=True,
        scene_sha256=digest(scene), per_pose=per_pose)
    write_yaml(output / 'feet_verification.yaml', result)
    return result


@contextmanager
def _no_transport():
    """Fail immediately if this diagnostic attempts Python network transport."""
    def blocked(*args, **kwargs):
        raise RuntimeError('Network transport is prohibited in the in-memory SONIC diagnostic')
    with patch('socket.socket', side_effect=blocked), patch('socket.create_connection', side_effect=blocked):
        yield
