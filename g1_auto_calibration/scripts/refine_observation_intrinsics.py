"""Offline multi-view intrinsic fitting with whole-pose held-out checks."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..common import digest, load_yaml


def reprojection(record, K, D):
    obs = record['board_observation']
    obj = np.asarray(obs['object_points_m'], np.float64)
    pixels = np.asarray(obs['image_points_px'], np.float64)
    ok, r, t = cv2.solvePnP(obj, pixels, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or np.any((cv2.Rodrigues(r)[0] @ obj.T+t)[2] <= 0):
        raise ValueError('Intrinsic validation placed the board behind the camera')
    predicted = cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((predicted-pixels)**2, axis=1))))


def fit(session, output):
    session, output = Path(session).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    acquisition = load_yaml(session/'recording/recording.yaml')
    if acquisition.get('complete') is not True:
        raise ValueError('Intrinsic refit requires a complete recording; partial trials are not promoted')
    frames_file = session/'recording/frames.jsonl'
    frames = [json.loads(line) for line in frames_file.read_text().splitlines()]
    manifest = load_yaml(session.parent.parent/'motion/prepared_reference.yaml')
    events = [json.loads(line) for line in (session/'reference_timing.jsonl').read_text().splitlines()]
    events = [e for e in events if e['original_reference'] and e['playing']]
    et, ef = [e['local_monotonic_s'] for e in events], [e['source_frame'] for e in events]
    available = [f for f in frames if f['board_observation'] is not None]
    if not available:
        raise ValueError('No recorded board corners')
    times = np.array([f['received_monotonic_s'] for f in available])
    selected, groups = [], []
    seen = set()
    for group, hold in enumerate(manifest['holds']):
        for offset in (25, 60):
            frame = max(hold['reference_start_frame'], hold['reference_end_frame']-offset)
            if frame < ef[0] or frame > ef[-1]:
                continue
            when = np.interp(frame, ef, et)
            record = available[int(np.argmin(abs(times-when)))]
            if record['source_sequence'] not in seen:
                selected.append(record)
                groups.append(group)
                seen.add(record['source_sequence'])
    holdout_groups = set(np.random.default_rng(42).permutation(20)[:4].tolist())
    training = [i for i, g in enumerate(groups) if g not in holdout_groups]
    testing = [i for i, g in enumerate(groups) if g in holdout_groups]
    if len(training) < 15 or len(testing) < 4:
        raise ValueError('Insufficient training or whole-pose validation views')
    first = selected[0]
    K0, D0 = np.array(first['K']), np.array(first['distortion'])
    image = cv2.imread(str(session/'recording'/first['image_file']))
    size = (image.shape[1], image.shape[0])
    obj = [np.asarray(selected[i]['board_observation']['object_points_m'], np.float32) for i in training]
    pixels = [np.asarray(selected[i]['board_observation']['image_points_px'], np.float32) for i in training]
    fixed = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL |
        cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL |
        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 |
        cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_FIX_TANGENT_DIST |
        cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY)
    definitions = [('factory_distortion_intrinsics_only', fixed, D0, 4),
        ('brown_two_radial', cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3, np.zeros(5), 6),
        ('brown_three_radial', cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST, np.zeros(5), 7)]
    models, candidates = {}, []
    n = sum(len(o)*2 for o in obj)
    for name, flags, initial_D, parameter_count in definitions:
        try:
            print(f'Fitting {name} on {len(training)} training views ...', flush=True)
            rms, K, D, _, _, deviations, _, _ = cv2.calibrateCameraExtended(obj, pixels, size,
                K0.copy(), initial_D.copy(), flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9))
            if not np.isfinite(np.r_[K.ravel(), D.ravel()]).all() or min(K[0, 0], K[1, 1]) <= 0:
                raise ValueError('Intrinsic fit returned invalid parameters')
            baseline = [reprojection(selected[i], K0, D0) for i in testing]
            errors = [reprojection(selected[i], K, D) for i in testing]
            score = float(n*np.log(max(rms*rms/2., np.finfo(float).eps))+parameter_count*np.log(n))
            report = dict(training_rms_px=float(rms), training_bic=score,
                held_out_rms_px=float(np.sqrt(np.mean(np.square(errors)))), held_out_max_rms_px=float(max(errors)),
                held_out_per_view_rms_px=errors, factory_held_out_per_view_rms_px=baseline,
                factory_held_out_rms_px=float(np.sqrt(np.mean(np.square(baseline)))),
                camera_matrix=K.tolist(), distortion_coefficients=D.ravel().tolist(),
                intrinsic_parameter_standard_deviations=deviations.ravel().tolist())
            models[name] = report
            candidates.append((score, name, report))
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            models[name] = dict(failure=str(exc))
    if not candidates:
        raise ValueError(f'All intrinsic models failed: {models}')
    _, name, chosen = min(candidates, key=lambda c: c[0])
    report = dict(kind='multiview_intrinsic_refit', camera=load_yaml(session/'recording/recording.yaml')['camera'],
        model=name, camera_matrix=chosen['camera_matrix'], distortion_coefficients=chosen['distortion_coefficients'],
        image_size=dict(width=size[0], height=size[1]), serial_number=first['serial'],
        status='provisional_multiview_intrinsic_estimate', accepted_for_calibration=False,
        held_out_reprojection_passed=chosen['held_out_max_rms_px'] <= 2.,
        validation=chosen, model_reports=models,
        model_selection='Training BIC only; whole-pose holdout excluded from all model fitting and selection',
        training_sequences=[selected[i]['source_sequence'] for i in training],
        held_out_sequences=[selected[i]['source_sequence'] for i in testing],
        held_out_pose_groups=sorted(holdout_groups),
        factory_camera_matrix=K0.tolist(), factory_distortion_coefficients=D0.tolist(),
        source_session=str(session), source_frames_sha256=digest(frames_file),
        assumptions=['Fixed focus, unchanged image scaling, correct planar metric panel geometry',
                     'This motion does not provide full image-field coverage or independent intrinsic ground truth'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Selected {name}; held-out RMS {chosen["held_out_rms_px"]:.3f} px '
          f'(factory {chosen["factory_held_out_rms_px"]:.3f} px). Saved {output}', flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feature-source', choices=('charuco', 'marker-centers'), default='charuco')
    a = p.parse_args()
    if a.feature_source == 'marker-centers':
        from ..calibration.marker_center_intrinsics import fit_archive
        fit_archive(a.session, a.output)
    else:
        fit(a.session, a.output)


if __name__ == '__main__':
    main()
