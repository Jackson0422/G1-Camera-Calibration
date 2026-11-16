import cv2
import numpy as np
import pytest

from g1_auto_calibration.calibration.marker_center_intrinsics import marker_centers, fit_fixed_distortion


def test_projective_centres_and_intrinsic_fit_with_wrong_initial_aspect():
    cv2.setNumThreads(1)
    board = cv2.aruco.CharucoBoard((9, 4), .115, .09,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    board.setLegacyPattern(True)
    markers = np.array(board.getObjPoints(), float)
    objects = markers.mean(axis=1)
    K = np.array([[1100., 0., 985.], [0., 820., 534.], [0., 0., 1.]])
    D = np.array([12., 9.7, .001, -.0016, -2.67, 12., 13.5, -2.42,
                  0., 0., 0., 0., -.008, -.015])
    guess = np.array([[1153., 0., 973.], [0., 1153., 523.], [0., 0., 1.]])
    rng = np.random.default_rng(81)
    views = []
    for i in range(40):
        r = rng.uniform(-.4, .4, 3)
        t = np.array([rng.uniform(-.65, -.2), rng.uniform(-.3, .05), rng.uniform(1.8, 2.6)])
        corners = cv2.projectPoints(markers.reshape(-1, 3), r, t, K, D)[0].reshape(-1, 4, 2)
        actual = cv2.projectPoints(objects, r, t, K, D)[0].reshape(-1, 2)
        np.testing.assert_allclose(marker_centers(corners, K, D), actual, atol=.02)
        corners += rng.normal(0., .025, corners.shape)
        views.append(dict(object_points_m=objects, marker_corners_px=corners))
    training, testing = list(range(32)), list(range(32, 40))
    fit = fit_fixed_distortion(views, training, testing, guess, D, (1920, 1080))
    np.testing.assert_allclose(fit['camera_matrix'], K, atol=1.5)
    assert fit['held_out_rms_px'] < .1
    # Changes to held-out image measurements cannot affect the fitted matrix.
    views[32]['marker_corners_px'] = views[32]['marker_corners_px']*1.01
    changed = fit_fixed_distortion(views, training, testing, guess, D, (1920, 1080))
    np.testing.assert_allclose(changed['camera_matrix'], fit['camera_matrix'], atol=1e-10)
    assert changed['held_out_rms_px'] > fit['held_out_rms_px']
    with pytest.raises(ValueError, match='disjoint'):
        fit_fixed_distortion(views, training, training, guess, D, (1920, 1080))
