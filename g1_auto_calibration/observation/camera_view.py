"""Live camera check before a trial: image, detected board corners and PnP error."""
import time

import cv2
import numpy as np

CONTINUE_KEYS = (13, 10, 32)       # Enter, Return, Space
ABORT_KEYS = (27, ord('q'), ord('Q'))


def annotate(frame, observation, error, *, width=1280):
    """Return a display-sized BGR image with board corners and a status banner."""
    image = frame.bgr.copy()
    if observation is not None:
        for x, y in np.asarray(observation['image_points_px'], float):
            cv2.circle(image, (int(round(x)), int(round(y))), 5, (0, 255, 0), -1)
        distance = float(np.linalg.norm(observation['T_camera_board'][:3, 3]))
        good = observation['geometric_gates_passed']
        status = (f"Board: {len(observation['image_points_px'])} corners, "
                  f"RMS {observation['reprojection_rms_px']:.2f} px, distance {distance:.2f} m"
                  + ('' if good else f" ({observation['quality_rejection_reason']})"))
        colour = (0, 200, 0) if good else (0, 200, 255)
    else:
        status, colour = f'Board not detected: {error}', (0, 0, 255)
    scale = width/image.shape[1]
    image = cv2.resize(image, (width, int(round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
    lines = [f'{frame.optical_frame}  serial {frame.serial}  seq {frame.sequence}', status,
             'Enter/Space or close window: continue    q/Esc: abort']
    cv2.rectangle(image, (0, 0), (width, 28*len(lines)+10), (0, 0, 0), -1)
    for index, text in enumerate(lines):
        cv2.putText(image, text, (12, 26+28*index), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    colour if index == 1 else (255, 255, 255), 2, cv2.LINE_AA)
    return image


def window_visible(window):
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def show_camera_view(camera, config, board, *, opener=None, detector=None,
                     window='Camera check', imshow=cv2.imshow, wait_key=cv2.waitKey,
                     destroy=cv2.destroyWindow, visible=window_visible, timeout_s=600.):
    """Show the live stream until the operator continues (True) or aborts (False).

    Enter, Space or closing the window continues; q or Esc aborts.
    """
    if opener is None:
        from ..hardware.base import open_camera as opener
    if detector is None:
        from ..calibration.board_detector import RawBoardDetector
        detector = RawBoardDetector(board)
    print(f'Camera check ({camera}): Enter/Space or close the window to continue, q/Esc to abort', flush=True)
    reader = opener(config)
    deadline = time.monotonic()+timeout_s
    shown = False  # only a window that has appeared can count as closed by the operator
    try:
        while time.monotonic() < deadline:
            frame = reader.read()
            try:
                observation, error = detector.estimate(frame), None
            except ValueError as exc:
                observation, error = None, exc
            imshow(window, annotate(frame, observation, error))
            key = wait_key(1) & 0xFF
            if key in CONTINUE_KEYS:
                return True
            if key in ABORT_KEYS:
                return False
            if visible(window):
                shown = True
            elif shown:
                return True
        return False
    finally:
        reader.close()
        try:
            destroy(window)
            wait_key(1)
        except cv2.error:
            pass
