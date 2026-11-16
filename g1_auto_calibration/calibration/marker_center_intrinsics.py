"""Fit image intrinsics without camera-dependent ChArUco interpolation."""

import json
from pathlib import Path

import cv2
import numpy as np

from ..common import digest, load_yaml
from .board_detector import BoardDetector, detect


def marker_centers(corners, K, D):
    """Intersect marker diagonals in the undistorted plane, then redistort.

    An arithmetic mean of image vertices is not the projective marker centre.
    The supplied distortion model remains fixed during the intrinsic fit.
    """
    rays = []
    for corner in corners:
        points = cv2.undistortPoints(np.asarray(corner, float).reshape(4, 1, 2), K, D).reshape(4, 2)
        h = np.c_[points, np.ones(4)]
        center = np.cross(np.cross(h[0], h[2]), np.cross(h[1], h[3]))
        if not np.isfinite(center).all() or abs(center[2]) < 1e-12:
            raise ValueError('Degenerate marker diagonals')
        rays.append(center/center[2])
    return cv2.projectPoints(np.asarray(rays), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


def fit_fixed_distortion(views, training, testing, K0, D, image_size):
    """Four intrinsic parameters; whole-pose validation never fits parameters."""
    if len(D) != 14:
        raise ValueError('This OAK refit requires the recorded 14-coefficient model')
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL |
        cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL |
        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 |
        cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_FIX_TANGENT_DIST |
        cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY)
    if set(training).intersection(testing) or min(len(training), len(testing)) < 4:
        raise ValueError('Need disjoint training and validation views')
    K = np.array(K0, float)
    for _ in range(3):
        objects = [np.asarray(views[i]['object_points_m'], np.float32) for i in training]
        pixels = [marker_centers(views[i]['marker_corners_px'], K, D).astype(np.float32) for i in training]
        rms, K, fitted_D, _, _, deviations, _, _ = cv2.calibrateCameraExtended(
            objects, pixels, image_size, K.copy(), D.copy(), flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9))
        if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError('Invalid fitted intrinsic matrix')
        if not np.allclose(fitted_D.ravel(), D, rtol=0, atol=1e-12):
            raise ValueError('Fixed distortion coefficients changed')

    def errors(indices, matrix):
        result = []
        for i in indices:
            obj = np.asarray(views[i]['object_points_m'], float)
            pixels = marker_centers(views[i]['marker_corners_px'], matrix, D)
            ok, r, t = cv2.solvePnP(obj, pixels, matrix, D)
            if not ok or np.min((cv2.Rodrigues(r)[0] @ obj.T+t)[2]) <= 0:
                raise ValueError('Invalid marker-centre board pose')
            predicted = cv2.projectPoints(obj, r, t, matrix, D)[0].reshape(-1, 2)
            result.append(float(np.sqrt(np.mean(np.sum((predicted-pixels)**2, axis=1)))))
        return result

    held, factory = errors(testing, K), errors(testing, K0)
    return dict(camera_matrix=K.tolist(), distortion_coefficients=D.tolist(),
        training_rms_px=float(rms), held_out_rms_px=float(np.sqrt(np.mean(np.square(held)))),
        held_out_max_rms_px=max(held), held_out_per_view_rms_px=held,
        factory_held_out_rms_px=float(np.sqrt(np.mean(np.square(factory)))),
        intrinsic_parameter_standard_deviations=deviations.ravel().tolist())


def fit_archive(session, output):
    session, output = Path(session).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    archive = load_yaml(session/'recording/recording.yaml')
    if archive.get('complete') is not True:
        raise ValueError('Need a complete closed recording')
    index = session/'recording/frames.jsonl'
    if digest(index) != archive['frame_index_sha256']:
        raise ValueError('Raw frame index changed')
    frames = [json.loads(line) for line in index.read_text().splitlines()]
    events = [json.loads(line) for line in (session/'reference_timing.jsonl').read_text().splitlines()]
    events = [e for e in events if e['original_reference'] and e['playing']]
    holds = load_yaml(session.parent.parent/'motion/prepared_reference.yaml')['holds']
    detector = BoardDetector(load_yaml(session/'recording/configuration.yaml')['board'])
    positions = {i: np.mean(p, axis=0) for i, p in detector.markers.items()}
    times = np.array([f['received_monotonic_s'] for f in frames])
    ef, et = [e['source_frame'] for e in events], [e['local_monotonic_s'] for e in events]
    views, seen, rejected = [], set(), []
    size = None
    for group, hold in enumerate(holds):
        for offset in (25, 60):
            frame = max(hold['reference_start_frame'], hold['reference_end_frame']-offset)
            if not ef[0] <= frame <= ef[-1]:
                continue
            packet = frames[int(np.argmin(abs(times-np.interp(frame, ef, et))))]
            if packet['source_sequence'] in seen:
                continue
            seen.add(packet['source_sequence'])
            path = session/'recording'/packet['image_file']
            if digest(path) != packet['image_sha256']:
                raise ValueError('Raw image changed')
            image = cv2.imread(str(path))
            size = image.shape[1], image.shape[0]
            corners, ids, _ = detect(image, detector.config['dictionary'])
            if len(set(ids)) != len(ids):
                rejected.append(dict(sequence=packet['source_sequence'], reason='Ambiguous repeated marker ID'))
                continue
            pairs = [(int(i), c.reshape(4, 2)) for i, c in zip(ids, corners) if int(i) in positions]
            if len(pairs) < 8:
                continue
            views.append(dict(sequence=packet['source_sequence'], pose_group=group,
                image_file=packet['image_file'], image_sha256=packet['image_sha256'],
                marker_ids=[i for i, _ in pairs], marker_corners_px=[p.tolist() for _, p in pairs],
                object_points_m=[positions[i].tolist() for i, _ in pairs]))
    held_groups = set(np.random.default_rng(42).permutation(len(holds))[:max(4, len(holds)//5)].tolist())
    training = [i for i, v in enumerate(views) if v['pose_group'] not in held_groups]
    testing = [i for i, v in enumerate(views) if v['pose_group'] in held_groups]
    if len(training) < 15 or len(testing) < 4:
        raise ValueError('Too few independent marker-centre views')
    K0, D = np.array(frames[0]['K']), np.array(frames[0]['distortion'])
    validation = fit_fixed_distortion(views, training, testing, K0, D, size)
    report = dict(kind='multiview_intrinsic_refit', camera=archive['camera'],
        model='factory_distortion_marker_center_intrinsics',
        camera_matrix=validation['camera_matrix'], distortion_coefficients=D.tolist(),
        factory_camera_matrix=K0.tolist(), factory_distortion_coefficients=D.tolist(),
        image_size=dict(width=size[0], height=size[1]), serial_number=frames[0]['serial'],
        status='provisional_multiview_intrinsic_estimate', accepted_for_calibration=False,
        held_out_reprojection_passed=validation['held_out_max_rms_px'] <= 2., validation=validation,
        training_sequences=[views[i]['sequence'] for i in training],
        held_out_sequences=[views[i]['sequence'] for i in testing],
        held_out_pose_groups=sorted(held_groups), views=views, rejected_views=rejected,
        source_session=str(session), source_frames_sha256=digest(index),
        model_selection='One preselected fixed-distortion model; only training views fit fx, fy, cx, cy',
        feature_source='ArUco projective marker centres; no camera-dependent ChArUco interpolation',
        assumptions=['Correct planar metric marker-centre layout and constant focus/output mapping',
            'Factory distortion retained; independent full-field lens calibration remains unverified'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Marker-centre intrinsic fit: held-out RMS {validation["held_out_rms_px"]:.3f} px; saved {output}', flush=True)
    return report
