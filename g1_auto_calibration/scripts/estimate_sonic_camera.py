"""Estimate torso<-camera from closed SONIC raw data, retaining all caveats."""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2

from ..common import ROOT, digest, load_yaml, write_yaml
from ..execution.timing_trace import load_trace
from ..calibration.time_alignment import fit_motion_clock, associate_time
from ..calibration.stability import model, measure
from ..calibration.forward_kinematics import ForwardKinematics
from ..calibration.handeye_solver import solve, linear_fit, refine, observability
from ..calibration.transforms import describe, mean_pose, pose_errors, inverse
from ..calibration.validator import validate, plot_quality


def estimate(session, output, *, intrinsics=None, redetect_corners=False):
    session, output = Path(session).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    directory = session/'recording'
    archive = load_yaml(directory/'recording.yaml')
    if archive.get('complete') is not True:
        raise ValueError('Need a complete closed raw recording')
    if digest(directory/'frames.jsonl') != archive['frame_index_sha256']:
        raise ValueError('Raw frame index changed')
    config = load_yaml(directory/'configuration.yaml')
    robot = config['robot']
    _, states, timing_frames = load_trace(directory/'timing_trace', archive['camera'])
    frames = [json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
    fitted_intrinsics = None
    if redetect_corners and intrinsics is None:
        raise ValueError('Corner redetection requires fitted intrinsics')
    if intrinsics is not None:
        fitted_intrinsics = json.loads(Path(intrinsics).read_text())
        if fitted_intrinsics['source_frames_sha256'] != digest(directory/'frames.jsonl'):
            raise ValueError('Refitted intrinsics belong to different raw observations')
        K = np.asarray(fitted_intrinsics['camera_matrix'], float)
        D = np.asarray(fitted_intrinsics['distortion_coefficients'], float)
        limit = config['board']['max_reprojection_rms_px']
        corrected = {}
        redetection_failures = {}
        if redetect_corners:
            from ..calibration.board_detector import BoardDetector
            from ..hardware.base import Frame
            detector = BoardDetector(config['board'])
        for packet in frames:
            obs = packet['board_observation']
            packet['K'], packet['distortion'] = K.tolist(), D.tolist()
            if redetect_corners:
                image_path = directory/packet['image_file']
                if digest(image_path) != packet['image_sha256']:
                    raise ValueError('Raw image changed before corner redetection')
                frame = Frame(cv2.imread(str(image_path)), packet['source_timestamp_s'],
                    packet['source_clock'], packet['received_monotonic_s'], packet['optical_frame'],
                    packet['serial'], K, D, packet['source_sequence']).validate()
                try:
                    obs = detector.estimate(frame)
                    obs['intrinsics_refitted'] = True
                except ValueError as exc:
                    obs = None
                    reason = str(exc)
                    redetection_failures[reason] = redetection_failures.get(reason, 0)+1
                packet['board_observation'] = obs
                if obs is not None:
                    corrected[packet['source_sequence']] = obs
                continue
            if obs is None:
                continue
            obj, pixels = np.asarray(obs['object_points_m'], float), np.asarray(obs['image_points_px'], float)
            ok, r, t = cv2.solvePnP(obj, pixels, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or np.any((cv2.Rodrigues(r)[0] @ obj.T+t)[2] <= 0):
                continue
            R = cv2.Rodrigues(r)[0]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t.ravel()
            rms = float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)-pixels)**2, axis=1))))
            obs.update(T_camera_board=T.tolist(), reprojection_rms_px=rms,
                       geometric_gates_passed=rms <= limit, intrinsics_refitted=True)
            corrected[packet['source_sequence']] = obs
        if redetect_corners:
            write_yaml(output/'redetection_summary.yaml', dict(
                source_frame_index_sha256=digest(directory/'frames.jsonl'),
                fitted_intrinsics_sha256=digest(intrinsics), raw_observations_modified=False,
                frame_count=len(frames), passed_unchanged_geometric_gates=len(corrected),
                failures=redetection_failures, corner_redetection_performed=True))
            print(f'Redetected board corners with fitted intrinsics: {len(corrected)}/{len(frames)} passed unchanged gates.', flush=True)
        for record in timing_frames:
            obs = corrected.get(record['sequence'])
            record['T_camera_board'] = obs['T_camera_board'] if obs and obs['geometric_gates_passed'] else None
            record['reprojection_rms_px'] = obs['reprojection_rms_px'] if obs else None
    events = [json.loads(line) for line in (session/'reference_timing.jsonl').read_text().splitlines()]
    events = [e for e in events if e['original_reference'] and e['playing']]
    if not events:
        raise ValueError('No original reference timing events')
    manifest = load_yaml(session.parent.parent/'motion/prepared_reference.yaml')
    et, ef = np.array([e['local_monotonic_s'] for e in events]), np.array([e['source_frame'] for e in events])
    received = np.array([f['received_monotonic_s'] for f in frames])
    samples = []
    for hold in manifest['holds']:
        frame = max(hold['reference_start_frame'], hold['reference_end_frame']-25)
        desired = float(np.interp(frame, ef, et))
        index = int(np.argmin(np.abs(received-desired)))
        packet = frames[index]
        if packet['board_observation'] is not None:
            samples.append(dict(pose_id=hold['pose_id'], index=index, packet=packet))
    print(f'Selected {len(samples)} end-of-hold frames from measured observations.', flush=True)
    excluded = [s['packet']['source_sequence'] for s in samples]
    alignment = None
    try:
        alignment = fit_motion_clock(states['received_s'], states['q'], timing_frames, robot,
                                     excluded_sequences=excluded)
        alignment['fixed_world_reference_assumption_verified'] = False
        alignment['conditional_on_fixed_reference_foot'] = True
        write_yaml(output/'time_alignment.yaml', alignment)
        print(f'Conditional motion clock fit: validation RMS {alignment["held_out_rotation_rms_deg"]:.4f} deg.', flush=True)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        write_yaml(output/'time_alignment.yaml', dict(validation_passed=False,
            failure=str(exc), fallback='PC receipt time; capture-time association unverified'))
        print(f'Clock fit unavailable: {exc}; estimating provisionally with receipt time.', flush=True)
    times, q = states['received_s'], states['q']
    qs, C, observations, sample_reports = [], [], [], []
    for sample in samples:
        packet = sample['packet']
        center, radius = packet['received_monotonic_s'], None
        if alignment is not None:
            center, radius = associate_time(alignment, packet['source_timestamp_s'])
        if not times[0] <= center <= times[-1]:
            raise ValueError('Selected camera time is outside measured state support')
        actual_q = np.array([np.interp(center, times, q[:, j]) for j in range(29)])
        start = np.searchsorted(times, center-robot['settle_window_s']-(radius or 0.))
        end = np.searchsorted(times, center+(radius or 0.), side='right')
        stability = measure(q[max(0, start-1):min(len(q), end+1)], robot)
        qs.append(actual_q)
        C.append(np.array(packet['board_observation']['T_camera_board']))
        observations.append(dict(packet['board_observation'], K=packet['K'], distortion=packet['distortion']))
        sample_reports.append(dict(pose_id=sample['pose_id'], video_frame_index=packet['video_frame_index'],
            source_sequence=packet['source_sequence'], image_file=packet['image_file'],
            state_time_s=center, timing_uncertainty_s=radius, q_measured_interpolated_rad=actual_q,
            stability=stability))
    G = model(robot).compute_many(qs)
    np.savez_compressed(output/'selected_data.npz', G=G, C=C, q=qs)
    write_yaml(output/'selected_frames.yaml', dict(samples=sample_reports,
        capture_time_alignment_conditional=alignment is not None,
        world_reference_foot_independently_verified=False))
    print('Solving measured-data hand-eye equation G X C = Y ...', flush=True)
    formal_failure = None
    try:
        X, Y, fit = solve(G, C)
        formal_fit = True
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        formal_failure = str(exc)
        formal_fit = False
        hold = list(range(0, len(G), 5))
        train = [i for i in range(len(G)) if i not in hold]
        candidates = []
        for method in ('svd_log', 'kronecker'):
            try:
                tg, tc = G[train], np.asarray(C)[train]
                x = refine(tg, tc, linear_fit(tg, tc, method))
                y = mean_pose([g @ x @ c for g, c in zip(tg, tc)])
                mm, deg = pose_errors(y, [g @ x @ c for g, c in zip(tg, tc)])
                candidates.append((float(np.mean(mm/5+deg/.5)), method, x, y))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass
        if not candidates:
            write_yaml(output/'result.yaml', dict(status='no_solution', reason=formal_failure,
                accepted_for_calibration=False, robot_connections_opened=0))
            raise
        _, method, X, Y = min(candidates, key=lambda c: c[0])
        fit = dict(method=method, training_indices=train, held_out_indices=hold,
            accepted_train_indices=train, observability=observability(G, C),
            diagnostic_fallback=True, adequate_training_consensus_established=False)
    quality = validate(G, C, X, Y, fit['held_out_indices'], observations)
    plot_quality(quality, output/'held_out_quality.png')
    all_mm, all_deg = pose_errors(Y, [g @ X @ c for g, c in zip(G, C)])
    sensitivity = {}
    for side, indices in [('first_half', list(range(len(G)//2))),
                           ('second_half', list(range(len(G)//2, len(G))))]:
        try:
            g, c = G[indices], np.asarray(C)[indices]
            x = refine(g, c, linear_fit(g, c, 'svd_log'))
            mm, deg = pose_errors(X, [x])
            sensitivity[side] = dict(T_torso_camera=describe(x), difference_from_primary_mm=float(mm[0]),
                                     difference_from_primary_deg=float(deg[0]))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            sensitivity[side] = dict(failure=str(exc))
    try:
        right = ForwardKinematics(model(robot).path, 'right_ankle_roll_link', robot['mount_frame'], robot['joint_indices'])
        rg = right.compute_many(qs)
        train = fit['accepted_train_indices']
        rx = refine(rg[train], np.asarray(C)[train], linear_fit(rg[train], np.asarray(C)[train], 'svd_log'))
        mm, deg = pose_errors(X, [rx])
        sensitivity['right_foot_reference'] = dict(T_torso_camera=describe(rx),
            difference_from_primary_mm=float(mm[0]), difference_from_primary_deg=float(deg[0]))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        sensitivity['right_foot_reference'] = dict(failure=str(exc))
    result = dict(status='provisional_measured_data_estimate', camera=archive['camera'],
        parent_frame=robot['mount_frame'], child_frame=config['camera']['optical_frame'],
        T_torso_camera_optical=describe(X), T_camera_optical_torso=describe(inverse(X)),
        T_left_foot_board=describe(Y), fit=fit, robust_solver_succeeded=formal_fit,
        robust_solver_failure=formal_failure, held_out_validation=quality,
        all_pose_translation_rms_mm=float(np.sqrt(np.mean(all_mm**2))),
        all_pose_rotation_rms_deg=float(np.sqrt(np.mean(all_deg**2))), sensitivity=sensitivity,
        sampling_stable_count=sum(s['stability']['passed'] for s in sample_reports), sample_count=len(samples),
        conditional_motion_clock_fit_passed=alignment is not None,
        physical_reference_foot_fixation_verified=False, accepted_for_calibration=False,
        previous_calibration_overwritten=False, robot_connections_opened=0, robot_control_commands_sent=0,
        corners_redetected_with_refitted_intrinsics=redetect_corners,
        assumptions=['Reference left foot and calibration panel fixed in world',
            'Camera rigidly attached to torso; supplied metric ChArUco geometry correct',
            'Motion-derived affine timing is conditional on the fixed-foot kinematic model'],
        source=dict(session=str(session), frame_index_sha256=digest(directory/'frames.jsonl'),
            timing_trace_manifest_sha256=digest(directory/'timing_trace/manifest.yaml'), urdf_sha256=model(robot).sha256))
    if fitted_intrinsics is not None:
        result['refitted_intrinsics'] = fitted_intrinsics
        result['refitted_intrinsics_file_sha256'] = digest(intrinsics)
    write_yaml(output/'result.yaml', result)
    print(summary(result, output), flush=True)
    return result


def summary(result, output=None):
    """Human-readable console report of one estimate."""
    T = np.asarray(result['T_torso_camera_optical']['matrix'], float)
    x, y, z = T[:3, 3]*100
    down = float(np.degrees(np.arcsin(np.clip(-T[2, 2], -1, 1))))
    q = result['held_out_validation']
    limits = q.get('thresholds', {})
    lines = [
        f"===== {result['camera']} camera extrinsics (relative to {result['parent_frame']}) =====",
        f'Position: forward {x:.1f} cm, left {y:.1f} cm, up {z:.1f} cm',
        f'Looking down: {down:.1f} deg',
        f"Quaternion xyzw: {[round(v, 5) for v in result['T_torso_camera_optical']['quaternion_xyzw']]}",
        f"Held-out validation: translation RMS {q['translation_rms_mm']:.1f} mm "
        f"(limit {limits.get('translation_rms_mm', '?')}), rotation RMS {q['rotation_rms_deg']:.2f} deg "
        f"(limit {limits.get('rotation_rms_deg', '?')}): {'PASSED' if q['passed'] else 'NOT PASSED'}",
        f"Samples: {result['sampling_stable_count']}/{result['sample_count']} stable; robust solver: "
        f"{'succeeded' if result['robust_solver_succeeded'] else 'no consistent solution (diagnostic only)'}"]
    if output is not None:
        lines.append(f'Result directory: {output}')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--intrinsics', type=Path, help='Optional hash-bound multi-view intrinsic refit JSON')
    p.add_argument('--redetect-corners', action='store_true', help='Re-detect raw images using fitted intrinsics, preserving all original geometric gates')
    a = p.parse_args()
    estimate(a.session, a.output, intrinsics=a.intrinsics, redetect_corners=a.redetect_corners)


if __name__ == '__main__':
    main()
