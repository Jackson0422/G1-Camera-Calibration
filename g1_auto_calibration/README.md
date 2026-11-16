G1 camera extrinsic calibration with SONIC
==========================================

Estimates `T_torso_camera_optical` for the G1's D435i and head OAK cameras. The
unmodified SONIC balance controller moves the waist through the operator's 40
calibration poses while a fixed ChArUco board is filmed. The poses are listed in
`config/calibration_poses_40.yaml`: yaw up to ±20°, roll and pitch up to ±10°,
each held for 2 s, about 3.2 minutes in total. Hand-eye calibration is then
solved from the recorded images and measured joint angles, using one frame per
pose. On hardware SONIC tracks yaw and roll but barely moves waist pitch.
`--poses <file>` selects another pose list in the same format.

Run all commands from `/home/kaixin/camera_calibration`. Scripts that import
SONIC must use its Python 3.10 environment directly; `bash run.sh` (ros_jazzy,
Python 3.12) is for the offline solve and the tests.

```
SIM=/home/kaixin/GR00T-WholeBodyControl/.venv_sim/bin/python
```

Layout
------

| Path | Purpose |
| --- | --- |
| `config/` | Board, camera endpoints, robot joint maps, G1 URDF |
| `execution/sonic_observation.py` | Trial preparation, MuJoCo preview, hardware run and raw capture |
| `hardware/` | Camera clients (network/OAK), Unitree state reader, SONIC PC session |
| `observation/` | Camera worker, board detection per frame, continuous recording |
| `calibration/` | Board detector, FK, hand-eye solver, time alignment, intrinsics refit |
| `scripts/` | Command-line entry points listed below |
| `results/` | Calibration outputs (see below); new trials are also written here |

1. Camera streams
-----------------

The OAK is plugged into the robot over USB and is published by `serve_oak` on
`tcp://192.168.123.164:5567`. `run_sonic_observation`, `resume_sonic_observation`
and `preview_cameras` handle this automatically (`hardware/robot_oak_server.py`,
settings under `oak.robot_server` in `config/cameras.yaml`):

- Just before capture they copy the package to the robot over SSH, which also
  works after a reboot clears `/tmp`. They then start `serve_oak` and wait for
  `OAK ready`, which takes about 5 s.
- When capture ends, including on errors or Ctrl+C, they stop it and the OAK is
  released. If the PC side dies, the SSH session drops and `serve_oak` exits by
  itself within a few seconds.
- If something is already serving port 5567, for example a server started by
  hand, it is reused and left running.

Only one process can open the OAK. If someone else holds it, startup fails with
their output and a hint, and no robot command is sent. Agree on a time slot.
Passwordless SSH to `unitree@192.168.123.164` is required.

The D435 stream (`tcp://192.168.123.164:5556`) comes from the robot's existing
camera service and needs no setup.

Check the stream and board visibility:

```bash
bash run.sh -m g1_auto_calibration.scripts.preview_cameras --camera oak --duration 30
```

2. Record a trial
-----------------

```bash
env __GL_SYNC_TO_VBLANK=0 $SIM -B -m g1_auto_calibration.scripts.run_sonic_observation \
  --standby --hardware-after-preview --camera oak --local-ip 192.168.123.123
```

The run goes through these steps:

1. **Camera check.** A window shows the live image with detected board corners,
   PnP RMS and board distance. Press Enter or Space to continue, or q or Esc to
   abort. Closing the window also continues. Skip it with `--no-camera-view`.
2. **Preview prompt.** Type `y`/`Y` to watch the full realtime MuJoCo preview, or
   `n`/`N` to skip it. When skipping, the newest earlier successful preview of
   exactly the same motion is reused. It must match reference, sources, profile,
   models, decoder, tools and physics files hash for hash, and is rebound to this
   run's files. If none matches, the preview runs anyway.
3. **Confirmation.** Type `yes` to confirm the robot is supported.
4. **Capture check.** The camera and joint streams open. The run fails before
   any robot command if no fresh frame with a visible board arrives within 15 s.
5. **Execution.** The robot-local decoder is deployed. On the robot, `X` starts
   the motion after balanced standby, `Y` returns early, and `U` requests a
   damped stop (support the robot). After `U` the program saves the raw data
   without encoding a video.
6. **Automatic solve.** If the motion window is complete, the extrinsics are
   solved with the intrinsics recorded in each frame (the camera defaults). The
   console prints the position, downward tilt and held-out validation. The result,
   `<prefix>_calibration.json` and the diagram are written to
   `results/<camera>_calibration_<session>/`. Use `--no-solve` to skip this step.

Camera frames are processed on three detection threads, and lossless PNG
encoding is part of the same parallel step, so a 15 fps stream is recorded
without dropped frames.

Use `--preview` instead of `--hardware-after-preview` for an offline-only run, and
`--camera d435` for the D435. If deployment fails after approval,
`resume_sonic_observation --trial ... --session ... --local-ip ...` retries with
the same approved preview.

Each attempt writes `sessions/<UTC>/recording/`, which holds lossless frames,
`frames.jsonl` with board detections and joint brackets, and `timing_trace/`.
`review_sonic_observation` (step 3 below) renders `review.mp4` from the frames.

The OAK publishes its default image: the 3840x2160 (16:9) sensor mode the factory
calibration was made in, scaled to 1920x1080. The factory intrinsics in each
frame therefore apply directly, with a board PnP RMS of about 0.55 px. Earlier
recordings (before 2026-11-16 afternoon) stretched the full 4056x3040 (4:3)
sensor instead, and needed refitted intrinsics.

3. Solve offline
----------------

The trial already solves once with the default intrinsics (step 6 above). Use
these commands to review a session, render its video, or solve again:

```bash
S=g1_auto_calibration/results/sonic_observation_<UTC>/sessions/<UTC>
bash run.sh -m g1_auto_calibration.scripts.review_sonic_observation --session $S
# Optional, recommended for the OAK: refit intrinsics from the recorded views
bash run.sh -m g1_auto_calibration.scripts.refine_observation_intrinsics \
  --session $S --output g1_auto_calibration/results/<name>_intrinsics.json --feature-source marker-centers
bash run.sh -m g1_auto_calibration.scripts.estimate_sonic_camera --session $S \
  --output g1_auto_calibration/results/<name> \
  [--intrinsics g1_auto_calibration/results/<name>_intrinsics.json --redetect-corners]
bash run.sh -m g1_auto_calibration.scripts.export_d435_estimate \
  --result-directory g1_auto_calibration/results/<name> --camera-name OAK --file-prefix OAK_head
```

`export_d435_estimate` works for either camera; it writes
`<prefix>_calibration.json` and a torso diagram.

Kept results
------------

Both are marked `provisional_estimate_not_accepted_for_calibration`. Their raw
recordings were deleted, so the `source.session` paths in `result.yaml` no longer
exist and these results cannot be re-solved.

| Camera | File | Position in `torso_link` (fwd, left, up) | Held-out validation |
| --- | --- | --- | --- |
| D435i (640x480), 2026-11-12 | `results/sonic_camera_estimate_20261112/D435i_calibration.json` | 4.8, 4.3, 39.3 cm | not passed |
| OAK head (1920x1080), 2026-11-16 | `results/oak_head_calibration_20261116T090904_558804Z/OAK_head_calibration.json` | 11.0, 3.1, 45.8 cm; 24.7° down | 13.0 mm / 0.60° (limits 5 mm / 0.5°) |

The OAK was re-mounted before 2026-11-16 and now looks about 25° down, at about
45 cm. Its result uses the default 4K-mode image with factory intrinsics and the
40-pose motion; the robust solver succeeded. The D435i result is only valid if
that camera has not been moved since 2026-11-12.

Diagnostics and tests
---------------------

- `observe_sonic`: read-only check of camera streams and robot state, no motion.
- `check_hardware`: read-only SDK/camera environment check.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 bash run.sh -m pytest -q tests
```

Plugin autoload is disabled because the ros_jazzy `dash` pytest plugin fails to
import with the installed pydantic.

Background: [SONIC calibration log](../docs/SONIC_CALIBRATION.md),
[SONIC interface audit](docs/SONIC_INTERFACE_AUDIT.md),
[hardware inspection](../docs/HARDWARE_INSPECTION.md),
[URDF provenance](../docs/MODEL_PROVENANCE.md).
