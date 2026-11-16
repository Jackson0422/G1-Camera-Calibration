"""Estimate a camera-to-state receipt-clock association from rigid-body motion.

Rotation magnitudes are invariant to a fixed camera mounting rotation. We fit
an affine clock map using camera/board rotations and actual URDF FK rotations.
No extrinsic guess, network latency bound or external timing instrument is used.
The resulting interval is statistical, not a deterministic transport guarantee.

The angular-rate norm principle is also used by Kalibr's time-offset prior:
https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py
"""

import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import chi2, norm

from .stability import model, policy
from .transforms import checked
from ..common import canonical_digest, digest, load_yaml


def has_motion_alignment(camera):
    value = camera.get('motion_time_alignment')
    if not isinstance(value, dict):
        return False
    try:
        covariance = np.asarray(value['covariance_b_span_s2'], float)
        interval = np.asarray(value['camera_source_interval_s'], float)
        numeric = [value[key] for key in ('camera_origin_s', 'camera_span_s', 'state_origin_s',
            'mapped_camera_span_s', 'camera_to_state_rate', 'empirical_time_scatter_95_s',
            'state_sample_gap_margin_s', 'held_out_rotation_rms_deg', 'held_out_rotation_max_deg',
            'rotation_rms_limit_deg')]
        return bool(value.get('status') == 'motion_alignment_validated'
            and value.get('validation_passed') is True
            and value.get('method') == 'affine_clock_from_rigid_rotation_magnitudes'
            and value.get('state_time_axis') == 'local_monotonic_state_receipt'
            and value.get('deterministic_transport_bound') is False
            and covariance.shape == (2, 2) and np.isfinite(covariance).all()
            and np.min(np.linalg.eigvalsh(covariance)) >= -np.finfo(float).eps
            and np.isfinite(numeric).all() and interval.shape == (2,) and np.isfinite(interval).all()
            and interval[0] == value['camera_origin_s'] and interval[1] > interval[0]
            and np.isclose(interval[1] - interval[0], value['camera_span_s'], atol=1e-9, rtol=1e-12)
            and value['mapped_camera_span_s'] > 0 and value['camera_span_s'] > 0
            and np.isclose(value['mapped_camera_span_s'] / value['camera_span_s'], value['camera_to_state_rate'])
            and 0 <= value['held_out_rotation_rms_deg'] <= value['rotation_rms_limit_deg']
            and 0 <= value['held_out_rotation_max_deg'] <= 2 * value['rotation_rms_limit_deg']
            and value['held_out_pair_count'] >= 4 and value['training_pair_count'] >= 12
            and np.isfinite([value['camera_to_state_rate'], value['empirical_time_scatter_95_s'],
                             value['state_sample_gap_margin_s']]).all()
            and value['camera_to_state_rate'] > 0 and value['empirical_time_scatter_95_s'] >= 0
            and 0 < value['state_sample_gap_margin_s'] <= .05)
    except (KeyError, TypeError, ValueError):
        return False


def verify_derived_timing_source(source, camera):
    """Require intact raw data, the actual fitted report, and disjoint images."""
    camera_config = source['camera_configs'][camera]
    if not has_motion_alignment(camera_config):
        raise ValueError('Invalid motion-estimated clock model')
    alignment = camera_config['motion_time_alignment']
    if source.get('motion_timing_derived') is not True:
        raise ValueError('Motion time alignment requires a derived observation dataset')
    report_path = Path(source['time_alignment_report']).resolve(strict=True)
    original_path = Path(source['original_observations']).resolve(strict=True)
    manifest_path = Path(alignment['timing_trace_manifest']).resolve(strict=True)
    if (digest(report_path) != source['time_alignment_report_sha256']
            or canonical_digest(load_yaml(report_path)) != canonical_digest(alignment)
            or digest(original_path) != source['original_observations_sha256']
            or digest(original_path) != alignment['original_observations_sha256']
            or original_path != Path(alignment['original_observations']).resolve()
            or digest(manifest_path) != alignment['timing_trace_manifest_sha256']):
        raise ValueError('Motion timing provenance changed after estimation')
    from ..execution.timing_trace import load_trace
    _, states, _ = load_trace(manifest_path.parent, camera)
    original = load_yaml(original_path)
    if any(canonical_digest(source[key]) != canonical_digest(original[key])
           for key in ('robot_config', 'board_config', 'board_sha256')):
        raise ValueError('Derived timing configuration differs from its raw acquisition')
    original_camera = original['camera_configs'][camera]
    if canonical_digest({k: v for k, v in camera_config.items() if k != 'motion_time_alignment'}) != canonical_digest(original_camera):
        raise ValueError('Derived camera configuration differs from its raw acquisition')
    excluded = {int(r['sequence']) for r in original['cameras'][camera]['samples']}
    training = {int(s) for pair in alignment['training_frame_sequences'] for s in pair}
    held_out = {int(s) for pair in alignment['held_out_frame_sequences'] for s in pair}
    if (excluded != set(alignment['excluded_calibration_image_sequences'])
            or training.intersection(held_out) or (training | held_out).intersection(excluded)):
        raise ValueError('Timing fitting reused held-out or saved calibration image frames')
    for key, count in (('training_frame_sequences', 'training_pair_count'), ('held_out_frame_sequences', 'held_out_pair_count')):
        pairs = alignment[key]
        endpoints = [int(s) for pair in pairs for s in pair]
        if len(pairs) != alignment[count] or len(endpoints) != 2*len(pairs) or len(set(endpoints)) != len(endpoints):
            raise ValueError('Timing validation pairs must have independent camera endpoints')
    robot = source['robot_config']
    criteria = policy(robot)
    if (criteria is None or alignment['urdf_sha256'] != model(robot).sha256
            or alignment['rotation_rms_limit_deg'] != criteria['rotation_limit_deg']
            or alignment['held_out_rotation_rms_deg'] > criteria['rotation_limit_deg']
            or alignment['held_out_rotation_max_deg'] > 2 * criteria['rotation_limit_deg']):
        raise ValueError('Timing model does not meet the approved accuracy criteria')
    records = source['cameras'][camera]['samples']
    originals = original['cameras'][camera]['samples']
    derived_keys = {'capture_alignment_method', 'capture_alignment_failure', 'capture_aligned_state',
        'T_reference_mount_from_capture_aligned_state', 'stance_monitoring',
        'original_capture_alignment_failure', 'original_stance_monitoring',
        'motion_estimated_capture_alignment'}
    if len(records) != len(originals):
        raise ValueError('Derived timing must preserve all original capture attempts')
    times = states['received_s']
    for record, raw in zip(records, originals):
        if canonical_digest({k: v for k, v in record.items() if k not in derived_keys}) != canonical_digest({k: v for k, v in raw.items() if k not in derived_keys}):
            raise ValueError('Derived timing changed a raw capture observation')
        if record.get('capture_alignment_failure') is not None:
            continue
        center, radius = associate_time(alignment, record['source_timestamp_s'])
        start = int(np.searchsorted(times, center-radius-float(robot['settle_window_s']), side='right'))-1
        end = int(np.searchsorted(times, center+radius, side='left'))
        if start < 0 or end >= len(times):
            raise ValueError('Derived capture window leaves the original state history')
        index = start + int(np.argmin(np.abs(times[start:end+1]-center)))
        saved = record['capture_aligned_state']
        for key in ('q', 'dq', 'imu_gyro', 'tick', 'mode_pr', 'mode_machine'):
            if not np.array_equal(saved[key], states[key][index]):
                raise ValueError('Derived capture did not retain its original nearest state sample')
        from .stability import measure
        timing = record['motion_estimated_capture_alignment']
        if (saved['state_received_s'] != float(times[index])
                or timing['state_interval_s'] != [float(times[start]), float(times[end])]
                or canonical_digest(timing['mount_stability']) != canonical_digest(measure(states['q'][start:end+1], robot))):
            raise ValueError('Derived capture stability differs from the complete raw state window')


def associate_time(alignment, source_timestamp_s):
    """Return the estimated corresponding state receipt time and 95% radius."""
    if alignment.get('status') != 'motion_alignment_validated' or alignment.get('validation_passed') is not True:
        raise ValueError('Motion time alignment did not pass validation')
    value = float(source_timestamp_s)
    interval = alignment['camera_source_interval_s']
    if not np.isfinite(value) or not interval[0] <= value <= interval[1]:
        raise ValueError('Camera timestamp is outside the fitted timing interval')
    u = (value - alignment['camera_origin_s']) / alignment['camera_span_s']
    vector = np.array([1., u])
    covariance = np.asarray(alignment['covariance_b_span_s2'], float)
    radius = float(norm.ppf(.975) * np.sqrt(max(0., vector @ covariance @ vector)))
    radius += alignment['empirical_time_scatter_95_s'] + alignment['state_sample_gap_margin_s']
    value = alignment['state_origin_s'] + alignment['mapped_camera_span_s'] * u
    if not np.isfinite([value, radius]).all() or radius <= 0:
        raise ValueError('Invalid estimated timing interval')
    return float(value), radius


def fit_motion_clock(state_times, state_q, frames, robot, *, excluded_sequences=()):
    """Fit on moving-frame pairs, then test disjoint held-out camera pairs.

    Every camera endpoint belongs to at most one pair. Saved calibration image
    sequences are excluded entirely. First/last state support defines the search
    domain; no asserted sensor-to-receipt delay is inferred from packet arrival.
    """
    criteria = policy(robot)
    if criteria is None:
        raise ValueError('Motion timing needs an approved sampling accuracy policy')
    times, q = np.asarray(state_times, float), np.asarray(state_q, float)
    if (times.ndim != 1 or len(times) < 5 or q.shape != (len(times), 29)
            or not np.isfinite(times).all() or not np.isfinite(q).all()
            or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > .05):
        raise ValueError('Timing estimation needs complete, strictly advancing PR state history')
    epoch = float(times[0])
    times = times - epoch
    if not frames:
        raise ValueError('Timing trace contains no camera frames')
    identities = {(f['source_clock'], f['serial'], f['optical_frame']) for f in frames}
    if len(identities) != 1:
        raise ValueError('Camera identity or source clock changed within the timing trace')
    all_source = np.array([f['source_timestamp_s'] for f in frames], float)
    all_sequence = np.array([f['sequence'] for f in frames])
    if (not np.isfinite(all_source).all() or np.any(np.diff(all_source) <= 0)
            or np.any(np.diff(all_sequence) <= 0)):
        raise ValueError('Camera timing timestamps and sequence IDs must strictly advance')
    excluded = set(excluded_sequences)
    usable = [f for f in frames if f.get('T_camera_board') is not None and f['sequence'] not in excluded]
    if len(usable) < 32:
        raise ValueError('Too few target-visible frames independent of saved calibration images for timing estimation')
    source = np.array([f['source_timestamp_s'] for f in usable], float)
    received = np.array([f['received_monotonic_s'] for f in usable], float)
    if not np.isfinite(received).all() or np.any(np.diff(received) <= 0):
        raise ValueError('Camera frame receipt times do not advance')
    rotations = np.array([checked(f['T_camera_board'])[:3, :3] for f in usable])
    origin, source_span = float(source[0]), float(source[-1] - source[0])
    approximate_rate = (received[-1] - received[0]) / source_span
    pair_span = float(robot['settle_window_s'])
    if not np.isfinite([source_span, approximate_rate, pair_span]).all() or min(source_span, approximate_rate, pair_span) <= 0:
        raise ValueError('Invalid timing trace duration')
    rotation_budget = float(np.deg2rad(criteria['rotation_limit_deg']))
    pairs, start = [], 0
    while start < len(usable) - 1:
        end = int(np.searchsorted(source, source[start] + pair_span / approximate_rate))
        if end >= len(usable):
            break
        angle = float(Rotation.from_matrix(rotations[start].T @ rotations[end]).magnitude())
        if angle > rotation_budget:
            pairs.append((start, end, angle))
        start = end + 1
    if len(pairs) < 20:
        raise ValueError(f'Timing offset is not observable: only {len(pairs)} independent moving-frame pairs; keep the target visible during diverse waist transitions')
    first = np.array([p[0] for p in pairs], int)
    last = np.array([p[1] for p in pairs], int)
    observed = np.array([p[2] for p in pairs])
    u0, u1 = (source[first] - origin) / source_span, (source[last] - origin) / source_span
    held_out = np.arange(len(pairs)) % 5 == 0
    training = np.flatnonzero(~held_out)
    testing = np.flatnonzero(held_out)
    fk = model(robot)

    def predict(parameters, indices):
        b, span = parameters
        t0, t1 = b + span * u0[indices], b + span * u1[indices]
        query = np.r_[t0, t1]
        clipped = np.clip(query, times[0], times[-1])
        angles = np.stack([np.interp(clipped, times, q[:, index]) for index in range(29)], axis=1)
        poses = fk.compute_many(angles)[:, :3, :3]
        count = len(indices)
        relative = np.swapaxes(poses[:count], 1, 2) @ poses[count:]
        return Rotation.from_matrix(relative).magnitude(), query

    def residual(parameters, indices=training):
        predicted, query = predict(parameters, indices)
        outside = np.maximum(times[0] - query, 0.) + np.maximum(query - times[-1], 0.)
        return np.r_[(predicted - observed[indices]) / rotation_budget, outside / pair_span]

    history_span = float(times[-1] - times[0])
    approximate_span = min(float(received[-1] - received[0]), history_span * (1. - 1e-8))
    upper_b = float(times[-1] - approximate_span)
    grid = np.linspace(times[0], upper_b, min(801, max(25, int((upper_b - times[0]) / np.median(np.diff(received))) + 1)))
    coarse_indices = training[np.linspace(0, len(training) - 1, min(160, len(training))).astype(int)]
    scores = [float(np.mean(residual([b, approximate_span], coarse_indices) ** 2)) for b in grid]
    seeds = []
    for index in np.argsort(scores):
        b = float(grid[index])
        if all(abs(b - seed) >= pair_span for seed in seeds):
            seeds.append(b)
        if len(seeds) >= 8:
            break
    fits = [least_squares(residual, [b, approximate_span],
        bounds=([times[0], np.min(np.diff(times))], [times[-1], history_span]),
        loss='soft_l1', f_scale=1., max_nfev=120, x_scale='jac') for b in seeds]
    fits.sort(key=lambda fit: fit.cost)
    best = fits[0]
    if not best.success or not np.isfinite(best.x).all():
        raise ValueError('Motion clock fit did not converge')
    predicted, query = predict(best.x, np.arange(len(pairs)))
    if np.min(query) < times[0] or np.max(query) > times[-1]:
        raise ValueError('Fitted camera times leave the recorded state history')
    train_error = (predicted - observed)[training] / rotation_budget
    jacobian = best.jac[:len(training)]
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[-1] <= singular[0] * np.sqrt(np.finfo(float).eps):
        raise ValueError('Timing offset or clock rate is not identifiable from this motion')
    variance = max(float(np.sum(train_error ** 2) / (len(training) - 2)), np.finfo(float).eps)
    covariance = np.linalg.inv(jacobian.T @ jacobian) * variance
    sensitivity = np.abs(jacobian[:, 0])
    useful = sensitivity >= max(float(np.median(sensitivity)), np.sqrt(np.finfo(float).eps))
    if np.count_nonzero(useful) < 4:
        raise ValueError('Motion timing has insufficient offset sensitivity')
    scatter = float(np.quantile(np.abs(train_error[useful] / jacobian[useful, 0]), .95))
    state_gap = float(np.max(np.diff(times)))
    radii = [norm.ppf(.975) * np.sqrt(max(0., v @ covariance @ v)) + scatter + state_gap
             for v in (np.array([1., 0.]), np.array([1., 1.]))]
    for other in fits[1:]:
        if other.success and other.cost <= best.cost + chi2.ppf(.95, 2) * variance:
            delta = other.x - best.x
            if max(abs(delta[0]), abs(delta.sum())) > 2 * max(radii):
                raise ValueError('Ambiguous timing: distinct clock mappings explain the observed motion')
    validation_error = np.rad2deg((predicted - observed)[testing])
    rms, maximum = float(np.sqrt(np.mean(validation_error ** 2))), float(np.max(np.abs(validation_error)))
    if rms > criteria['rotation_limit_deg'] or maximum > 2 * criteria['rotation_limit_deg']:
        raise ValueError(f'Independent timing motion validation failed: rotation RMS {rms:.6f} deg, maximum {maximum:.6f} deg')
    identity = next(iter(identities))
    return dict(schema_version=1, status='motion_alignment_validated', validation_passed=True,
        method='affine_clock_from_rigid_rotation_magnitudes',
        state_time_axis='local_monotonic_state_receipt',
        camera_source_clock=identity[0], serial=identity[1], optical_frame=identity[2],
        camera_origin_s=origin, camera_span_s=source_span,
        camera_source_interval_s=[float(source[0]), float(source[-1])],
        state_origin_s=float(best.x[0] + epoch), mapped_camera_span_s=float(best.x[1]),
        camera_to_state_rate=float(best.x[1] / source_span),
        covariance_b_span_s2=covariance.tolist(), empirical_time_scatter_95_s=scatter,
        state_sample_gap_margin_s=state_gap,
        uncertainty_kind='95_percent_normal_approximation_plus_empirical_scatter_and_sample_gap',
        deterministic_transport_bound=False,
        held_out_rotation_rms_deg=rms, held_out_rotation_max_deg=maximum,
        rotation_rms_limit_deg=float(criteria['rotation_limit_deg']),
        moving_pair_count=len(pairs), training_pair_count=len(training), held_out_pair_count=len(testing),
        training_frame_sequences=[[int(usable[first[i]]['sequence']), int(usable[last[i]]['sequence'])] for i in training],
        held_out_frame_sequences=[[int(usable[first[i]]['sequence']), int(usable[last[i]]['sequence'])] for i in testing],
        excluded_calibration_image_sequences=sorted(int(v) for v in excluded),
        urdf_sha256=fk.sha256,
        assumptions=['Camera mounting and board placement remain rigid and fixed',
                     'An affine source clock model holds within this session',
                     'State receipt timing jitter is represented by the empirical motion residuals'],
        estimation_changes_robot_control=False)
