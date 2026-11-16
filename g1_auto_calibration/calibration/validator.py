import cv2
import numpy as np
from .transforms import inverse, pose_errors


def validate(G, C, X, Y, indices, observations=None):
    if len(indices) < 4:
        raise ValueError('Need at least four independent validation poses')
    mm, deg = pose_errors(Y, [G[i] @ X @ C[i] for i in indices])
    reproj = []
    if observations is not None:
        for i in indices:
            o = observations[i]
            # Predict camera<-board from fixed training board and calibrated mount.
            expected = inverse(G[i] @ X) @ Y
            points = np.asarray(o['object_points_m'], float)
            if np.any((expected[:3, :3] @ points.T+expected[:3, 3:4])[2] <= 0):
                raise ValueError('Predicted validation target behind camera')
            pixels = cv2.projectPoints(points, cv2.Rodrigues(expected[:3, :3])[0], expected[:3, 3],
                                       np.asarray(o['K'], float), np.asarray(o['distortion'], float))[0].reshape(-1, 2)
            reproj.append(float(np.sqrt(np.mean(np.sum((pixels-np.asarray(o['image_points_px']))**2, axis=1)))))
    passed = bool(np.sqrt(np.mean(mm**2)) <= 5 and np.max(mm) <= 10 and
                  np.sqrt(np.mean(deg**2)) <= .5 and np.max(deg) <= 1 and
                  reproj and max(reproj) <= 2)
    return dict(passed=passed, sample_indices=indices, translation_mm=mm.tolist(), rotation_deg=deg.tolist(),
                translation_rms_mm=float(np.sqrt(np.mean(mm**2))), translation_max_mm=float(max(mm)),
                rotation_rms_deg=float(np.sqrt(np.mean(deg**2))), rotation_max_deg=float(max(deg)),
                predicted_reprojection_rms_px=reproj,
                thresholds=dict(translation_rms_mm=5, translation_max_mm=10, rotation_rms_deg=.5,
                                rotation_max_deg=1, predicted_reprojection_max_px=2),
                note='Consistency test conditional on metric target, timing, rigid mount and independently fixed base; not absolute ground truth')


def plot_quality(report, path):
    image = np.full((660, 1100, 3), 255, np.uint8)
    series = [(report['translation_mm'], 'Translation error (mm)'),
              (report['rotation_deg'], 'Rotation error (deg)'),
              (report['predicted_reprojection_rms_px'], 'Predicted reprojection RMS (px)')]
    for k, (values, label) in enumerate(series):
        y0 = 60+k*210
        cv2.putText(image, label, (30, y0-15), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
        cv2.line(image, (60, y0), (60, y0+145), (100, 100, 100), 1)
        cv2.line(image, (60, y0+145), (1060, y0+145), (100, 100, 100), 1)
        top = max(max(values, default=1), .001)
        for j, value in enumerate(values):
            x = int(90+j*940/max(len(values)-1, 1))
            y = int(y0+145-130*value/top)
            cv2.circle(image, (x, y), 5, (180, 70, 10), -1)
            cv2.putText(image, f'{value:.3f}', (x-20, y-8), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    if not cv2.imwrite(str(path), image):
        raise OSError('Could not write validation plot')
