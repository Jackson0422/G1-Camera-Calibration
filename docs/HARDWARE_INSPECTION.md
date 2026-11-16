Hardware inspection and verification — 2026-11-10
================================================

All inspection and tests were read-only with respect to robot control. Zero robot
motion commands were issued. No existing service, factory process, controller mode,
or camera driver was replaced or restarted. An independent OAK reader ran for a
bounded interval from a separate directory under the robot's `/tmp` and exited
normally. No persistent OAK service was installed.

Workstation and environment
---------------------------

- Project: `/home/kaixin/camera_calibration` (initially empty).
- Requested Python: `/home/kaixin/miniconda3/envs/ros_jazzy/bin/python`, Python 3.12.15.
- Existing packages: NumPy 2.5.4, PyYAML 6.0.3, OpenCV 5.0.0, ROS 2 `rclpy`.
- OpenCV provides ArUco/ChArUco/PnP but does not export `calibrateHandEye`.
- Project-local `.deps`: DepthAI 3.12.0, SciPy 1.18.1, pyzmq 27.3.0,
  MessagePack 1.2.4 and CycloneDDS Python 0.10.2. The latter was compiled using
  the existing Conda `libddsc.so.0.10.5`. Exact versions are in `DEPENDENCIES.json`.
- Unitree SDK2 Python source: `/home/kaixin/unitree_sdk2_python`, setup version 1.0.1;
  it is loaded from source, not installed into the shared Conda environment.
- Ethernet: `eno1`, `192.168.123.123/24`; Wi-Fi: `192.168.0.108/24`.
- No OAK or RealSense USB device is attached to this workstation.

Robot and devices
------------------

- SSH-reachable host: `unitree@192.168.123.164`, hostname `unitree-g1-nx`, aarch64.
- Robot Ethernet: `enP8p1s0`, `192.168.123.164/24`.
- Robot Wi-Fi: `wlx94ba06f265a9`, `192.168.0.130/24`.
- Existing RealSense process: PID 6416 at inspection, running the existing
  `.venv_camera/bin/python -m gear_sonic.camera.composed_camera` with serial
  `419522071149` and original port 5555. It was preserved.
- Robot USB reports Intel RealSense D435i (`8086:0b3a`) and Movidius MyriadX
  (`03e7:2485`). Both cameras are hosted on the robot.
- OAK SDK discovery reports device ID `19443010D173EE4800`.
- Important model discrepancy: OAK EEPROM reports product **OAK-D-W**, board
  **DM9098**, rather than the task's stated OAK-D S2. The implementation uses
  this device's own calibration. Confirm the physical product/label when practical;
  no S2-specific intrinsics or installation extrinsics were substituted.
- Native robot DepthAI version: 3.12.0. Existing
  `/home/unitree/miniconda3/envs/g1_wireless/bin/python` has DepthAI, cv2, NumPy,
  zmq and msgpack and can run the OAK server without adding packages.

Actual acquisition tests
------------------------

| Source | Verified outcome | Evidence |
| --- | --- | --- |
| G1 joint state | `eno1`, DDS domain 0, `rt/lowstate`, `unitree_hg.msg.dds_.LowState_`, PR mode 0, all 29 joint positions | `results/state_live.yaml` |
| Joint-state rate | 2,982 unique tick samples in approximately 3 s, approximately 1 kHz; duplicate ticks discarded; no state-buffer error | `results/state_live.yaml` |
| D435i | 48 RGB frames in approximately 5 s, 9.48 Hz, 640x480, serial `419522071149`, actual K and five zero distortion coefficients | `results/hardware_live.yaml`, `results/d435_latest.jpg` |
| OAK | 47 RGB frames over a 3 s observation interval, 15.24 Hz, 1920x1080, device ID `19443010D173EE4800`, actual K and 14 distortion coefficients | `results/oak_probe.yaml`, `results/oak_latest.jpg`, `results/oak_probe.server.log` |
| OAK shutdown | Independent robot server exited normally with exit code 0 | `results/oak_probe.yaml` |

The first `hardware_live.yaml` run recorded duplicate DDS ticks as an error. The
buffer was corrected to discard duplicates while retaining reset/backwards-tick
rejection; `state_live.yaml` supersedes that first state diagnostic. The earlier
`oak_live.yaml` timeout occurred after a short-lived server had already exited
between separate test commands. The synchronized finite probe in `oak_probe.yaml`
is the successful end-to-end OAK acquisition test.

Both saved real camera images were inspected. Neither shows the Spot target.
The dictionary candidate diagnostics found no board markers in these images.
No real calibration pose or installation transform was produced.

D435 K for the verified 640x480 stream:

```
607.1337280273438   0                 320.9698791503906
0                   607.0213623046875 247.58245849609375
0                   0                 1
```

OAK K for the verified 1920x1080 stream:

```
1153.460693359375  0                 972.8737182617188
0                 1152.9884033203125 523.3448486328125
0                 0                 1
```

These matrices are diagnostic snapshots, not hardcoded acquisition calibration.
Every received frame retrieves calibration metadata from its source.

Camera interfaces and timing
----------------------------

The existing D435 receiver/publisher implementations were inspected before writing
a client:

- `/home/kaixin/G1-demo/hardware/camera.py`
- `/home/kaixin/G1-demo/hardware/realsense_server.py`
- `/home/kaixin/G1-demo/hardware/sonic_metric_tap.py`
- `/home/kaixin/GR00T-WholeBodyControl/gear_sonic/camera/drivers/realsense.py`

Working transport is ZeroMQ SUB/PUB with single MessagePack packets, not an
invented TCP image framing scheme. Existing calibrated side-channel endpoints
listen at `192.168.123.164:5556` and `192.168.0.130:5556`. Metadata carries
intrinsics, serial, timestamps, clock domain and frame numbers. Legacy port 5555
is preserved; its depth previews are irrelevant to this RGB hand-eye solver.

The independent OAK server publishes the same RGB metadata envelope on its own
port 5567 while reading the USB camera directly through native DepthAI. The v3
DeviceInfo identifier is `getDeviceId()`; EEPROM camera model is read from
`getEepromData().cameraData[CAM_A].cameraType`. Both were checked against the
installed bindings. The returned model was perspective and the 14-term OpenCV
distortion profile was preserved, including sensor tilt terms.

The D435 capture clock reports `timestamp_domain.global_time`. OAK capture clock
is DepthAI's host-synchronized clock on the robot. Workstation DDS timestamps are
local monotonic receipt times. None of these were equated. Preview is verified;
calibration-grade cross-host synchronization or transport bounds are not yet
established. All live reports therefore retain `synchronized_for_calibration: false`.

Robot model and joint mapping
-----------------------------

Source URDF:
`/home/kaixin/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf`.
An exact byte copy is included for FK reproducibility. Its SHA256 is
`56d0c7b4e44351ee8e89e7b1f1fcc8b02d5ec5cd191e43239d2b06f039e75860`.
Mesh files were not copied; this model snapshot is for FK, not visual rendering.
See `MODEL_PROVENANCE.md` for origin and notices.

The mount chain is `pelvis -> waist_yaw_link -> waist_roll_link -> torso_link`.
The model does have `head_link`; its existence was established from the XML.
`d435_link` and `mid360_link` are fixed children of `torso_link`. Nominal sensor
transforms are not used as calibrated values. Unitree's installed
`example/g1/low_level/g1_low_level_example.py` and `read_g1_state.py` establish the
29-DOF index order. Waist yaw, roll, pitch are 12, 13, 14 in PR coordinates;
AB motor coordinates are rejected. Physical OAK attachment still needs confirmation.

Target source verification
---------------------------

The user supplied a RAI source lead, independently verified against
[RAI's Spot ChArUco source](https://github.com/bdaiinstitute/spot_wrapper/blob/main/spot_wrapper/calibration/charuco_board_detection.py):
9x4 checkers, 0.115 m checker edge, 0.09 m marker edge, `DICT_4X4_50`, legacy true.
The configured board has 18 markers and 24 internal corners. The active region
is 1.035x0.460 m. The generated layout is under `results/board_reference`.
No marker dimensions were inferred from the outer panel.

This source verifies the expected Spot default geometry; the actual physical board
has not yet been seen by these cameras. `physical_pattern_verified` remains false.
No external RAI controller or automatic motion code was imported or executed.

Offline verification
---------------------

Eight numerical/integration tests pass: exact hand-eye direction for two independent
linear solvers; single-axis degeneracy; real URDF chain and yaw mapping; Spot legacy
IDs and dimensions; synthetic rendered ChArUco pose recovery; noisy hand-eye data
with a rejected training outlier and disjoint holdout; timing and AB-mode rejection;
and network packet color/calibration/serial/distortion validation.
`compileall` passes. Synthetic dataset generation, solving, numerical validation and
plot generation were executed end to end.

The reproducible 28-pose synthetic run gives held-out consistency RMS
**0.895 mm / 0.059 degrees**. Estimated installation-transform error against
synthetic ground truth is **1.251 mm / 0.026 degrees**. These are simulation results,
not claims about this robot's calibration accuracy. The synthetic result remains
`candidate_only`, and TF export rejects it. Real collection with the default
unverified prerequisite configuration was also executed and correctly refused
before opening hardware or creating a session directory.

Outstanding prerequisites for real calibration
----------------------------------------------

1. Put the stationary Spot target in each camera's view and check actual IDs/layout.
2. Confirm each camera is rigid to the selected URDF mounting link; no ruler needed.
3. Establish cross-host timing or evidence-backed camera/DDS latency bounds.
4. Verify pelvis fixation relative to the stationary target; controller balancing
   can move the pelvis even when feet remain planted. Independent moving-base
   compensation is not implemented.
5. Explicitly approve physical movement and validate an external controller before
   moving the robot. A 25-pose offline waist proposal is provided; it is not a
   validated safe trajectory and no execution API exists here.
6. Collect 20-30 valid, settled, diverse poses per camera, then solve and pass held-out
   consistency/reprojection checks before exporting calibrated mount TF.

The delivered implementation is executable and the real read-only acquisition
paths are verified. Installation extrinsics and production TF publication remain
unverified until the above physical data and constraints are available.
