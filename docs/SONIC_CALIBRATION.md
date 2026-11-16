SONIC standing calibration integration
=====================================

> Historical log. The 2026-11-16 cleanup kept only the SONIC observation workflow
> (see `g1_auto_calibration/README.md`). Scripts such as `sonic_calibration`,
> `verify_sonic_policy` and `replay_sonic_trace`, and most linked `results/` files,
> no longer exist.

Two-minute observation trial (2026-11-12)
----------------------------------------

The current trial uses a 120-second, 50 Hz reference with waist yaw, roll and
pitch bounded by ±15 degrees, slow quintic transitions, and twenty hold blocks.
The reference keeps all other joints at SONIC's default standing angles. The
released SONIC policy still adjusts all twenty-nine actual motor targets for
balance. Initialization moves from measured joints to SONIC's default posture;
this trial does not initialize to the separate saved native-control posture.

`scripts/run_sonic_observation.py` prepares its own genuine motion source CSV,
encoder reference, dedicated profile, model hashes and decoder deployment
bundle. The upstream gains, joint limits and motor ownership remain unchanged.
Preparation is local. Hardware mode first shows a fresh, complete, realtime
MuJoCo preview, retains the existing interactive `yes` confirmation, checks
camera/state readiness, and then deploys and runs the matching robot-local
decoder. The PC only encodes references and subscribes to measured state.

Use the simulation environment directly; `run.sh` adds Python-3.12 dependencies
that must not be imported into SONIC's Python-3.10 environment:

```bash
/home/kaixin/GR00T-WholeBodyControl/.venv_sim/bin/python -B \
  -m g1_auto_calibration.scripts.run_sonic_observation \
  --output g1_auto_calibration/results/sonic_observation_new
```

To run a prepared trial with its exact profile and source:

```bash
/home/kaixin/GR00T-WholeBodyControl/.venv_sim/bin/python -B \
  -m g1_auto_calibration.scripts.run_sonic_observation \
  --trial g1_auto_calibration/results/sonic_observation_15deg_120s_v2 \
  --hardware-after-preview --camera d435 --local-ip 192.168.123.123
```

Add `--preview` instead of `--hardware-after-preview` for a completely offline
preview. Each attempt gets a new `sessions/<UTC>/` directory. Use the existing
runtime keys `Y` for an early return and `U` for the guarded controller stop.
The runtime confirms its default standing hold before requesting the final
supported stop; its session remains bounded to 180 seconds.

Acquisition begins on the first playing reference token and lasts 120 seconds.
The resulting `recording/` directory retains every received frame as a lossless
PNG and a matching `frames.jsonl` record containing board corner/marker IDs,
PnP, camera intrinsics, source timestamps and actual measured joint brackets.
`timing_trace/` retains the continuous measured joint history, while
`reference_timing.jsonl` stamps selected reference frames in the same PC clock.
`review.mp4` has exactly one encoded frame per indexed PNG. Its presentation
times are for viewing; the original timestamps remain in the frame index.
Early failures preserve the partial archive and report the incomplete interval.

These archives are candidate data for onsite observation, with no automatic
hand-eye solve or acceptance. A frame ID ties the image and its detected board
features exactly; exposure-time association with joint samples still requires
validated camera/state clock alignment. Receipt-time brackets are labeled as
provisional. Actual leg joint changes can enter FK compensation; common world
foot motion needs an independently known trajectory. No simulated world motion
is substituted for hardware measurements.

The completed ±15-degree dynamics diagnostic consumed all 6001 references,
returned to default and recorded no falls. Its left foot changed by up to
5.329 mm and 3.017 degrees from the first hold. It therefore does not establish
the fixed-foot calibration assumption. Smaller tested amplitudes did not
monotonically reduce foot drift. The saved offline video is
[`review_120s.mp4`](../g1_auto_calibration/results/sonic_stationary_120s/descending_15deg/review_120s.mp4).
The synthetic calibration diagnostics are explicitly ideal virtual-camera
experiments, not measurements of the physical D435.

The subsequent real observation trial completed the entire reference and
returned to default, confirmed five seconds of standing hold, and shut down
cleanly after the operator confirmed that someone could support the robot.
It recorded 1054 indexed images, all with board detections, and 168762 raw
state samples including startup/return (119937 within the 120-second image
window). Image hashes and video frame count were checked. The maximum board
PnP RMS was 0.566 pixels. Actual yaw ranged from -13.44 to +12.21 degrees and
roll from -15.50 to +15.54 degrees; actual pitch stayed between -0.48 and
+1.47 degrees despite its ±15-degree reference. The controller's successful
run is therefore not a claim of accurate pitch tracking. Leg joint changes
were recorded; world foot displacement was not measured. No calibration was
performed and exposure-time joint association remains pending.
See the [review summary](../g1_auto_calibration/results/sonic_observation_15deg_120s_v2/sessions/20261112T130641_098824Z/review_summary.yaml)
and [measured motion plot](../g1_auto_calibration/results/sonic_observation_15deg_120s_v2/sessions/20261112T130641_098824Z/joint_board_review.png).

On 2026-11-12 the operator asked to reassess the existing SONIC controller as
an alternative to native joint-position control. Leg joint motion is compatible
with this calibration: use synchronized measured joints for
`T_left_foot_torso(q)`, while the left reference foot and panel remain stationary.
The legs need not retain a fixed angle vector. Stepping or common foot slip
invalidates the current stationary-reference model; relative-foot monitoring
alone cannot observe common slip. Final target angles are never substituted for
measured kinematics.

The source mode-0 `Mode0Reference` accepts joint position and velocity arrays
with 29 columns plus root WXYZ quaternion references at 50 Hz. A socket-blocked
check using the actual encoder accepted all forty original waist targets with
the saved twenty-six non-waist references unchanged. See
[`sonic_saved_pose_input_check_20261112T120518Z/report.json`](../g1_auto_calibration/results/sonic_saved_pose_input_check_20261112T120518Z/report.json).
These were stationary input probes, not validated transitions or motor tracking.
The policy independently generates all twenty-nine final motor targets.

The subsequent saved-standing dynamics experiment consumed all forty references
without a fall. Across its capture windows, the world left-foot pose changed
by up to 16.334 mm and 3.970 degrees from the first hold. This invalidates the
fixed-left-foot assumption for that simulated run; measured joints alone cannot
compensate motion of their own world reference. The maximum within-window
left-foot angular diameter was 0.825 degrees. These are measured simulation
quantities, not new URDF limits or claimed final calibration errors.
See the [full diagnostic](../g1_auto_calibration/results/sonic_saved_standing_policy_20261112121212Z/verification.yaml)
and [independent foot audit](../g1_auto_calibration/results/sonic_saved_standing_foot_audit_20261112T121500Z/report.json).
The final factory-default return also timed out; that is reported separately
from the operator's requirement for stability during the forty-pose execution.
No SONIC controller was launched on hardware. The operator subsequently
confirmed that the robot was offline; hardware attempts remain stopped.

Reproduce the saved-standing experiment without robot or network I/O:

```bash
/home/kaixin/GR00T-WholeBodyControl/.venv_sim/bin/python -B \
  -m g1_auto_calibration.scripts.verify_sonic_policy \
  --standing-reference g1_auto_calibration/config/default_standing_pose.yaml \
  --output g1_auto_calibration/results/sonic_saved_standing_new
```

For the externally supported development-mode setup, the new executable
connection is described in [Automatic acquisition](AUTOMATIC_ACQUISITION.md).
It actively holds measured leg and arm targets and changes only the waist.
The SONIC-specific entry below retains its read-only behavior.

The requested workflow is: initialize a known standing pose, preview all 40
motions in MuJoCo, enter exactly `yes`, execute them with SONIC maintaining
balance, and independently record each camera at measured settled poses. Work on
2026-11-11 was restricted to read-only robot tests. No SONIC controller was
launched, robot initialized, control mode changed, or motion command transmitted.

Implemented components
----------------------

`hardware/sonic.py` encodes the installed native SONIC planner protocol, holds a
frozen arm reference, validates joint limits and parses debug feedback. Its
publisher is inert until explicitly activated after preview, confirmation,
controller-contract and wired-route gates. It sends policy references, not exact
motor targets. The protocol and running-controller contract are separate checks.
See [the interface audit](../g1_auto_calibration/docs/SONIC_INTERFACE_AUDIT.md) for source fingerprints.

`planning/sonic_sequence.py` is a pure 40-pose coordinator. It requires a complete
preview receipt tied to the plan and frozen reference, then exact `yes` and
validated prerequisites. It uses measured waist arrival and a complete settling
window, collects the two cameras independently, and returns to reference. A
timeout aborts new decisions. This is not a claim that closing a reference
publisher physically stops the robot. No direct low-level writer is used.

`scripts/observe_sonic.py` opens both wired streams concurrently and overlays
detected board features. It records candidate images, marker/corner IDs, PnP,
camera calibration, actual joint angles and timestamp metadata independently.
A missing board in one camera does not reject the other camera's observation.
Standalone observations do not infer commanded pose IDs and are not solver
sessions. They remain `candidate_only: true` and `calibration_ready: false`.

The top-level `scripts/sonic_calibration.py` exposes read-only preflight, nominal
40-pose preview and observation. Automatic physical initialization/execution is
not connected. Entering `yes` cannot override the current readiness failures.
The existing zero-arm nominal preview cannot approve a different measured arm
reference. Such a reference needs a new endpoint and transition screening.

The current acquisition preference is sequential: D435 first, OAK second. Each
round has its own fixed panel placement, output directory and independent set of
accepted images. The panel may be repositioned between the rounds; it must stay
fixed within each round. The 40 reference poses may be reused, while each camera
accepts its own actual observations. `--camera d435` does not open, test or wait
for OAK. `--camera oak` does not require the D435 stream. The workflow entry point
defaults to D435; `--camera all` remains available for dual preview.

Run from the project root:

```
bash run.sh -m g1_auto_calibration.scripts.sonic_calibration
bash run.sh -m g1_auto_calibration.scripts.sonic_calibration --preview
bash run.sh -m g1_auto_calibration.scripts.replay_sonic_trace
bash run.sh -m g1_auto_calibration.scripts.sonic_calibration --camera d435 --observe --duration 120
bash run.sh -m g1_auto_calibration.scripts.sonic_calibration --camera oak --observe --duration 120
```

The second command previews nominal target geometry; the third replays the
saved actual SONIC simulation, including floating-base and joint motion.
Neither can authorize hardware. The observer requires only its selected streams
to be running. Use `Q` or window close to exit the live video. The OAK-only round
still needs the camera-only publisher; this does not start a motion controller.

Fixed-foot reference
--------------------

The user confirmed that both feet stay planted without lifting, turning or
displacement and that the calibration panel stays fixed. These confirmations
are recorded in `config/robot_sonic.yaml`; the original configuration remains
unchanged. The new reference is `left_ankle_roll_link` rather than `pelvis`.

Full URDF FK now evaluates arbitrary relative link transforms:

```
G_i = T_left_foot_torso(q_i)
G_i X C_i = T_left_foot_board
X = T_torso_camera_optical
```

Leg joint changes therefore compensate pelvis motion caused by balancing. The
calibrated installation transform is still torso-to-camera. A relative-foot
monitor rejects changes exceeding 5 mm or 1 degree and checks complete state
windows. These are diagnostic thresholds, not measured guarantees of foot
fixation. Common motion of both feet cannot be detected from joint state and
must remain excluded by the confirmed physical setup. Camera/state timing,
physical board layout and actual rigid camera attachment still require evidence.

Read-only Ethernet results
--------------------------

The workstation uses `eno1`, 192.168.123.123, to reach 192.168.123.164. LowState
was received at approximately 1000 Hz in PR mode with `mode_machine: 5`. D435
streaming worked at approximately 8.83 Hz, 640x480. OAK-D-W streaming worked at
approximately 14.89 Hz, 1920x1080, with the real EEPROM calibration. The existing
D435 forwarder addresses the robot's own Wi-Fi-assigned IP locally through `lo`;
it does not transmit that upstream connection over the Wi-Fi radio.

A concurrent 15-second dual-camera observer test completed with zero commands.
Raw reported joint velocity exceeded the existing settling threshold, so no
candidate was incorrectly accepted as settled. The threshold was not relaxed.
OAK used a finite, camera-only publisher which exited normally after 60 seconds;
no persistent service was installed. Future previews require that camera-only
publisher to be running again. The inspected processes included a SONIC camera
service but no active SONIC motion controller.

Relevant reports live under `g1_auto_calibration/results/`:

* `sonic_readonly_20261111.yaml`
* `sonic_robot_inventory_20261111.yaml`
* `sonic_readonly_snapshot_20261111.yaml`
* `sonic_oak_readonly_20261111/hardware.yaml`
* `sonic_dual_readonly_20261111/observations.yaml`
* `sonic_reference_assessment_20261111.yaml`

Current measured reference screening failed: the wrist collision proxy overlaps
a hip proxy, and the frozen leg model does not put all foot proxies on the
floor. This is a model/reference mismatch, not proof of an actual physical
collision. No clearance or support tolerance was relaxed to pass it.

Actual released-policy diagnostics
---------------------------------

The new offline policy test runs the released encoder and portable guarded
decoder with floating-base MuJoCo in memory. It creates no DDS or network
connection. This differs from the native ZMQ planner path and cannot certify
that path or the installed robot controller. The policy test uses a separate
existing simulation environment because `ros_jazzy` lacks ONNX Runtime; the
calibration application continues using `ros_jazzy`.

With the measured arm snapshot, all 40 reference holds were consumed but the
final return did not settle before its deadline. With standard SONIC arms, the
complete reference and balanced return succeeded without a fall. Waist pitch
still followed the requested references poorly. Off-target observations can
nevertheless be useful for calibration if actual FK, stability, board visibility
and independent excitation checks pass. Diagnostic quiet windows are not valid
calibration samples by themselves. Reports and tracking plots are in
`results/sonic_policy_preview/` and `results/sonic_policy_preview_defaultarms/`.

The standard-arm simulation also rotated and displaced its feet: relative foot
pose changed by up to 23.376 mm and 5.959 degrees over the route. It exceeded
the fixed-foot gate at pose 2, despite remaining upright and completing the
return. This does not establish the real robot's stance behavior, but means that
this simulated route has not validated the confirmed stationary-foot assumption.
`feet_verification.yaml` records the measured world and relative foot transforms.

Physical execution remains pending verification of the native controller,
initial reference, actual stance, input ownership, timeout/return behavior and
capture synchronization. A visual approval or a successful policy simulation
cannot silently enable those conditions.

OAK head-camera automatic-start trial, 2026-11-12
------------------------------------------------

The observation runner now starts the single 120-second reference automatically
after the original SONIC supervisor reports fresh, qualified stable standby.
There is no second X confirmation. The original U/Y controls, motor fault,
temperature and state-freshness checks remain active. Preview and its initial
hardware approval still occur before starting the motor owner.

For a new trial, use the simulation environment directly; `run.sh` adds Python
dependencies belonging to a different interpreter:

```
env __GL_SYNC_TO_VBLANK=0 /home/kaixin/GR00T-WholeBodyControl/.venv_sim/bin/python -B -m g1_auto_calibration.scripts.run_sonic_observation --standby --hardware-after-preview --camera oak --local-ip 192.168.123.123
```

This command prepares a fresh hash-bound trial. Existing prepared trials retain
their recorded configuration and must pass source checks. The upstream console
still prints its generic X instructions; the local adapter automatically makes
the start request only when the original readiness checks permit it.

Real execution succeeded in
`g1_auto_calibration/results/oak_head_autostart_20261112/sessions/20261112T142619_515107Z`.
The session recorded exactly one automatic start request, completed the full
reference timeline, returned to neutral, confirmed five seconds of balanced
post-standby and stopped via U with the previously confirmed on-site support.
The hardware report has no failure or remote faults. The independent read-only
temperature monitor recorded a maximum of 51 degrees C and no 75-degree events.

The closed archive contains 120 seconds of video, 1,795 lossless images and
167,839 raw robot-state samples including startup/return. Board corners were
detected in 1,779 images. Every video frame matches a lossless image and recorded
corner/joint-receipt record. Exposure-time alignment and world foot fixation
remain unverified; the recording is not itself an accepted calibration.

Factory-intrinsic OAK diagnostics are exported in
`results/oak_head_factory_estimate_20261112/OAK_head_calibration.json` with an
English torso-frame diagram. **This is a rejected/provisional numerical
estimate, not a validated mounting position.** Held-out consistency is
20.85 mm translation RMS and 3.08 degrees rotation RMS. Independent timing
validation also failed. An offline intrinsic refit reduced held-out corner
reprojection RMS from 3.02 to 2.53 pixels, still above the existing criteria;
it is saved separately under `results/oak_head_estimate_20261112/` and did not
replace the factory intrinsics or raw observations. Its alternative extrinsic
estimate also failed validation. No D435i calibration was overwritten.

OAK intrinsic-mapping correction and independent repeat
-----------------------------------------------------

The previous 87.8 cm mounting-height estimate is rejected. Its ChArUco
interpolation used the same mismatched packet intrinsics as its pose solver;
refitting those already biased chessboard corners did not correct the problem.
The camera output currently needs an independently estimated intrinsic matrix.
The specific sensor/ISP scaling cause has not been independently established.

`refine_observation_intrinsics --feature-source marker-centers` now extracts
raw ArUco vertices, intersects their diagonals in the undistorted plane and
fits fx/fy/cx/cy using their known metric grid centres. All 14 factory distortion
coefficients remain fixed. Four complete pose groups are excluded from fitting,
including all their image views. The detector never uses the operator's
approximate mounting height. The code is numerically tested with a deliberately
wrong initial aspect ratio, perspective distortion, noisy marker corners and
perturbed held-out views that cannot change the fitted matrix.

Reprocessing the first full recording gives a held-out marker-centre RMS of
0.568 pixels. Redetecting ChArUco corners with these fitted intrinsics passes
the original geometric gates in 1,775 of 1,795 images, validates conditional
motion timing at 0.274 degrees RMS and estimates an optical-centre height of
48.52 cm. First-/second-half solutions differ from the primary by about 9 mm.
Full extrinsic accuracy checks still fail; this corrected numerical result
remains provisional.

A second real 120-second trial completed normally in
`results/oak_head_recalibration_motion_20261112/sessions/20261112T144932_333349Z`.
It contains 1,792 lossless images and video frames with 159,292 raw state samples
including startup/return. It auto-started after qualified balance, returned and
confirmed post-standby, and stopped normally via U. A new intrinsic fit using
only this repeat recording gives held-out RMS 0.469 pixels. Raw packet metadata,
images, and the original D435i calibration remain untouched.

Use the fitted intrinsics to **redetect** raw image corners, rather than merely
re-solving the old ChArUco observations:

```
/home/kaixin/GR00T-WholeBodyControl/.venv_sim/bin/python -B -m g1_auto_calibration.scripts.refine_observation_intrinsics --session SESSION --output NEW_INTRINSICS.json --feature-source marker-centers
/home/kaixin/GR00T-WholeBodyControl/.venv_sim/bin/python -B -m g1_auto_calibration.scripts.estimate_sonic_camera --session SESSION --output NEW_RESULT_DIRECTORY --intrinsics NEW_INTRINSICS.json --redetect-corners
```

Repeat-recording output is under
`results/oak_head_recalibrated_extrinsics_20261112/`, with fitted intrinsics in
`results/oak_head_recalibrated_20261112/intrinsics_refit.json`.

The independent repeat estimates the optical centre at
`[0.08285477, 0.01231887, 0.48606410]` metres in torso coordinates (forward,
left, up), with viewing direction tilted 4.08 degrees downward. The previous
corrected run differs by 5.79 mm in position and 1.47 degrees in rotation;
the height difference alone is 0.89 mm. These differences are repeatability
measurements, not absolute accuracy bounds. The repeat's held-out extrinsic
consistency remains 24.68 mm translation RMS and 1.05 degrees rotation RMS,
with predicted reprojection above 2 pixels. Accordingly, the English JSON and
PNG/SVG diagrams explicitly retain `accepted: false` and provisional labels.
The source recording and prior failed estimates are preserved for comparison.
