"""Dictionary diagnostics do not establish metric layout or physical marker scale."""
import cv2
import numpy as np
from .transforms import transform


def detect(image, name):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.detectInvertedMarker = True  # Spot panel black background; diagnose both polarities.
    corners, ids, rejected = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(image)
    return corners, np.array([], int) if ids is None else ids.ravel(), len(rejected)


def dictionary_candidates(image):
    # Nested dictionaries share codewords. Return ALL matches; never auto-verify 4X4_50.
    names = [n for n in dir(cv2.aruco) if n.startswith('DICT_') and isinstance(getattr(cv2.aruco, n), int)]
    result = []
    for name in names:
        corners, ids, rejected = detect(image, name)
        if len(ids):
            result.append(dict(dictionary=name, ids=ids.tolist(), corners_px=[c.reshape(4, 2).tolist() for c in corners],
                               detections=len(ids), rejected_candidates=rejected,
                               total_area_px2=float(sum(abs(cv2.contourArea(c)) for c in corners))))
    return sorted(result, key=lambda x: (x['detections'], x['total_area_px2']), reverse=True)


class BoardDetector:
    def __init__(self, config):
        self.config = config
        if not config.get('dictionary_verified') or not config.get('dictionary'):
            raise ValueError('Board dictionary unverified; run identify_board first')
        if not config.get('geometry_verified') or not config.get('geometry_source') or config.get('units') != 'm':
            raise ValueError('Metric marker geometry missing; panel outline is insufficient')
        self.charuco = None
        if config.get('kind') == 'charuco':
            self.charuco = cv2.aruco.CharucoBoard((int(config['squares_x']), int(config['squares_y'])),
                float(config['square_length_m']), float(config['marker_length_m']),
                cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config['dictionary'])))
            self.charuco.setLegacyPattern(bool(config['legacy_pattern']))
            # Geometry AND ID order are taken from the same OpenCV board, not hand-written.
            self.markers = {int(i): np.array(v, float) for i, v in zip(self.charuco.getIds(), self.charuco.getObjPoints())}
        else:
            self.markers = {int(k): np.array(v, float) for k, v in config['markers'].items()}
        if len(self.markers) < config.get('min_markers', 3):
            raise ValueError('Insufficient marker layout')
        for v in self.markers.values():
            if v.shape != (4, 3) or not np.isfinite(v).all() or np.linalg.norm(np.cross(v[1]-v[0], v[2]-v[0])) < 1e-8:
                raise ValueError('Invalid marker XYZ corners')

    def estimate(self, frame, *, retain_quality_rejected=False):
        corners, ids, rejected = detect(frame.bgr, self.config['dictionary'])
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate marker IDs are ambiguous')
        pairs = [(i, c.reshape(4, 2)) for i, c in zip(ids, corners)
                 if i in self.markers and abs(cv2.contourArea(c)) >= self.config.get('min_marker_area_px2', 100)]
        if len(pairs) < self.config.get('min_markers', 3):
            raise ValueError('Too few reliable known markers')
        obj = np.concatenate([self.markers[i] for i, _ in pairs])
        img = np.concatenate([c for _, c in pairs]).astype(float)
        charuco_ids = []
        if self.charuco is not None:
            cp = cv2.aruco.CharucoParameters()
            cp.cameraMatrix = frame.K
            cp.distCoeffs = frame.distortion
            dp = cv2.aruco.DetectorParameters()
            dp.detectInvertedMarker = True
            detector = cv2.aruco.CharucoDetector(self.charuco, cp, dp)
            cc, ci, _, _ = detector.detectBoard(frame.bgr)
            if ci is None or len(ci) < self.config.get('min_charuco_corners', 8):
                raise ValueError('Too few ChArUco chessboard corners')
            if self.charuco.checkCharucoCornersCollinear(ci):
                raise ValueError('ChArUco corners are collinear')
            obj = np.asarray(self.charuco.getChessboardCorners()[ci.ravel()], float)
            img = np.asarray(cc, float).reshape(-1, 2)
            charuco_ids = ci.ravel().tolist()
        # This implementation uses IPPE and requires board XY plane in metric layout.
        if not np.allclose(obj[:, 2], 0, atol=1e-7):
            raise ValueError('Planar IPPE target must have Z=0; convert layout into board plane')
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, frame.K, frame.distortion, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise ValueError('PnP failed')
        poses = []
        for rvec, tvec in zip(rvecs, tvecs):
            R = cv2.Rodrigues(rvec)[0]
            if np.min((R @ obj.T+tvec).T[:, 2]) <= 0:
                continue
            projection = cv2.projectPoints(obj, rvec, tvec, frame.K, frame.distortion)[0].reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((projection-img)**2, axis=1))))
            poses.append((rms, rvec, tvec))
        poses.sort(key=lambda x: x[0])
        if not poses or (not retain_quality_rejected and poses[0][0] > self.config.get('max_reprojection_rms_px', 1.5)):
            raise ValueError('Reprojection error too high or target behind camera')
        gap = None if len(poses) < 2 else poses[1][0]-poses[0][0]
        ambiguous = gap is not None and gap < self.config.get('min_pose_error_gap_px', .25)
        if ambiguous and not retain_quality_rejected:
            raise ValueError('Ambiguous planar pose; acquire a more oblique view')
        rms, r, t = poses[0]
        r, t = cv2.solvePnPRefineLM(obj, img, frame.K, frame.distortion, r, t)
        R = cv2.Rodrigues(r)[0]
        proj = cv2.projectPoints(obj, r, t, frame.K, frame.distortion)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((proj-img)**2, axis=1))))
        if np.min((R @ obj.T+t).T[:, 2]) <= 0:
            raise ValueError('Refined pose failed quality limits')
        quality_passed = rms <= self.config.get('max_reprojection_rms_px', 1.5) and not ambiguous
        if not quality_passed and not retain_quality_rejected:
            raise ValueError('Refined pose failed quality limits')
        reason = None if quality_passed else ('Ambiguous planar pose' if ambiguous else 'Reprojection error exceeds unchanged board quality limit')
        return dict(T_camera_board=transform(R, t), reprojection_rms_px=rms, pose_error_gap_px=gap,
                    marker_ids=[int(i) for i, _ in pairs], object_points_m=obj, image_points_px=img,
                    charuco_ids=charuco_ids,
                    rejected_candidates=rejected, geometric_gates_passed=quality_passed,
                    quality_rejection_reason=reason, candidate_only=bool(retain_quality_rejected),
                    confidence=('raw candidate; acceptance not granted' if retain_quality_rejected else
                                'passed geometric gates; not a calibrated probability'))


class RawBoardDetector(BoardDetector):
    """Retain usable corners and PnP even when quality fails; never accept them."""

    def estimate(self, frame):
        return super().estimate(frame, retain_quality_rejected=True)
