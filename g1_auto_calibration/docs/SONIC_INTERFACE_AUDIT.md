# SONIC interface audit

> Historical log. The 2026-11-16 cleanup kept only the SONIC observation workflow
> (see `g1_auto_calibration/README.md`). Scripts such as `sonic_calibration`,
> `verify_sonic_policy` and `replay_sonic_trace`, and most linked `results/` files,
> no longer exist.

Audit date: 2026-11-11. This inspection read local source files only. It did not
start SONIC, create a real command publisher, change robot modes, or send a
robot control message.

## Supported reference interface

The local C++ SONIC deployment supports an external upper-body reference in
planner mode. `ZMQManager::OnPlannerReceived` accepts
`upper_body_position[17]` and `upper_body_velocity[17]`.
`GatherInputInterfaceData` reads these buffers, and the motion-observation
gatherers replace the corresponding reference joint positions and velocities.
The policy still generates the final motor commands for the whole robot.

This is a suitable integration boundary to investigate for stationary camera
calibration. It avoids introducing a competing Unitree command writer. It does
not guarantee exact waist tracking or balance at any requested posture.

Relevant files under `/home/kaixin/GR00T-WholeBodyControl`:

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp:776`:
  planner decoder; optional position and velocity buffers begin at lines 892
  and 923. Required fields are `mode`, `movement`, and `facing`.
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp:80`:
  upper-body IsaacLab-to-SDK mapping.
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp:78`:
  standing IDLE is mode `0`.
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:1109`:
  external upper-body positions replace motion-reference values. Velocity
  replacement begins at line 1187; input buffering begins at line 3809.
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp:588`:
  a one-second planner timeout clears upper-body control and returns the
  locomotion request to IDLE. This is not a validated physical stop strategy.

The 17-element vector is ordered as follows:

| Vector index | SDK joint index | Joint |
| --- | --- | --- |
| 0 | 12 | Waist yaw |
| 1 | 13 | Waist roll |
| 2 | 14 | Waist pitch |
| 3 | 15 | Left shoulder pitch |
| 4 | 22 | Right shoulder pitch |
| 5 | 16 | Left shoulder roll |
| 6 | 23 | Right shoulder roll |
| 7 | 17 | Left shoulder yaw |
| 8 | 24 | Right shoulder yaw |
| 9 | 18 | Left elbow |
| 10 | 25 | Right elbow |
| 11 | 19 | Left wrist roll |
| 12 | 26 | Right wrist roll |
| 13 | 20 | Left wrist pitch |
| 14 | 27 | Right wrist pitch |
| 15 | 21 | Left wrist yaw |
| 16 | 28 | Right wrist yaw |

The calibration adapter freezes the fourteen arm references from a measured
29-joint PR snapshot, changes only the three waist references, and checks the
whole snapshot against the actual URDF limits. It requests mode 0, zero
movement, zero speed, a constant horizontal facing direction, and default
height. In native SONIC, static-mode speed is subsequently normalized to the
planner's default value; mode IDLE, rather than the numeric speed alone,
selects stationary behavior.

## Packet and network direction

The native packed input format is one ZMQ message containing:

```text
ASCII topic prefix + 1280-byte NUL-padded JSON header + little-endian arrays
```

The header has `v=1`, `endian=le`, `count=1`, and field descriptors with names,
types and shapes. Topic `planner` contains the reference. Topic `command`
contains `start:u8[1]`, `stop:u8[1]`, and `planner:u8[1]`; the adapter always
uses `planner=1` and never switches to full-body streamed motion.

`ZMQPackedMessageSubscriber::PollOnce` at lines 260-340 verifies the topic,
reads the fixed header, and splits consecutive array buffers. `OnPlannerReceived`
uses those buffers. The Python adapter's tests independently inspect each
array offset and unpack the primitive values, and compare the mapping with
the actual local SONIC source.

The SONIC controller is a SUB client. The calibration publisher must bind to a
specific workstation Ethernet address, and the already verified controller
must be configured to connect to that address and port. The robot camera's
PUB endpoint on port 5556 is a separate service on the robot. An arbitrary
publisher cannot attach to that camera endpoint to control SONIC.

## Feedback limitations and startup

Native output uses `g1_debug` followed by a msgpack map. The relevant source
is `include/output_interface/zmq_output_handler.hpp:263`. `body_q` and
`body_dq` are measured positions and velocities converted to MuJoCo/SDK
ordering; the position default-angle offsets have already been restored.
`body_q_target` is separate and must not be used as measured state.

Debug feedback has an increasing logger index, measured joint state and base
IMU orientation. It has no command acknowledgement, controller ownership
proof, deployment hash, current program state, or independently bounded sensor
capture time. Local receipt freshness alone cannot supply those facts.

The C++ state machine publishes full debug state only after policy control has
started (`src/g1_deploy_onnx_ref.cpp:5080`). During INIT and WAIT_FOR_CONTROL,
it republishes only `robot_config` (lines 4875-4912). Consequently, requiring
fresh `g1_debug` is appropriate when attaching to an existing active SONIC
standing session, but cannot bootstrap an unstarted controller. The adapter
does not fabricate startup feedback from DDS. A separately verified controller
startup/status path is needed to automate initialization.

Starting the native C++ deployment itself is an actuator-affecting action:
its INIT phase ramps to the configured standing posture, and its motor writer
owns the robot command interface. It must not be launched as a read-only test.

The separate `robot_decoder` Python deployment is a different integration
boundary. Its passive token protocol explicitly has no joint-target command;
its motion service requires a pinned profile and a new session authorization.
The existence of that deployment does not prove that the native C++ planner
socket is running on the robot.

## Existing evidence against assuming precise tracking

The local historical simulation report
`recordings/waist_pitch_sim_20261027/REPORT.md` records a SONIC mode-0 reference
change of -0.01 rad in waist pitch. In that run, the measured settled change
was only -0.000141 rad, approximately 1.4 percent of the request. The run used
the reference-motion input rather than the new planner packet adapter. It
therefore does not test this adapter, but it directly demonstrates that a
valid waist reference cannot be treated as a reached motor angle.

The 40-pose calibration must verify actual waist motion and settling from
measured state. Policy tracking of the complete route, actual controller
ownership, deployment parity, initialization, entry/exit, and lost-input
behavior remain separate requirements. Passing packet tests or a kinematic
MuJoCo preview cannot mark them verified.

## Local adapter status

`hardware/sonic.py` contains pure packet builders, a read-only feedback
subscriber, and a publisher with no socket side effects at construction.
Activation requires a new completed preview, exact `yes` confirmation,
independently verified controller contract, and a verified wired route.
Each request also requires recent parsed SONIC feedback. Closing the transport
only closes it; it does not claim that the physical robot has stopped.

The caller must not set contract verification merely because the source code
contains these fields. Native feedback has no ownership or binary identity
claim. A source-to-running-binary and mode check must establish that contract.

Seven focused adapter tests passed with mocked publisher sockets. These tests
did not create any real command sockets or connect to the robot:

```bash
bash run.sh -m unittest tests.test_sonic -v
```

## Complete policy diagnostics on 2026-11-11

The project wrapper `scripts/verify_sonic_policy.py` ran the released mode-0
ONNX encoder and decoder through the existing portable guarded controller in
floating-base MuJoCo. Tokens and decoder messages were processed in memory.
Python socket creation was prohibited, and no DDS participant or native
controller subprocess was created. The existing SONIC environment was used
because it already contains ONNX Runtime; `ros_jazzy` does not currently contain
that dependency. All new artifacts were written in this calibration workspace.

Each run retained the complete 182.78125-second original waist route with
two-second holds. A derived diagnostic reference explicitly replaced the
old kinematic zero-leg reference with the pinned SONIC standing legs. It added
three seconds at the default pose, a five-second arm-reference blend and two
seconds at the frozen arm reference. Supported initialization, unsupported
policy dynamics, final return and hold were included. Original plans and user
review decisions were preserved.

| Diagnostic arm reference | Full 40-pose reference | Final return | Median target error, worst axis/pose | Quiet waist holds |
| --- | --- | --- | --- | --- |
| Measured read-only arm snapshot | Complete | Failed to settle before the 30-second deadline | 18.119 degrees | 12 / 40 |
| Pinned SONIC default arms | Complete | Completed, including five-second final hold | 11.007 degrees | 19 / 40 |

The quiet-hold count examines only the final half second of each hold, with
waist speed at most 0.015 rad/s and waist range at most 0.005 rad. It does not
apply the full acquisition history, all-joint, IMU and camera-timing gates;
these are not counts of accepted calibration samples. A one-degree position
tolerance was used only to summarize target tracking. Calibration can use
off-target poses when measured kinematics, board visibility, excitation and
stationary reference assumptions are all valid.

Neither diagnostic fell. With default arms, actual yaw and roll motion supplied
two-axis excitation in the saved settled waist windows, while actual pitch
stayed close to zero despite requests of plus or minus ten degrees. This is
evidence about this mode-0 pipeline and model. It does not establish behavior
of the native ZMQ planner interface or prove a need for policy retraining.

The fixed-foot assumption was checked independently using each saved floating
base pose and joint vector with the exact dynamics model. Across the final
one-second windows of the default-arm holds:

- Left-foot position changed by up to 7.926 mm and orientation by 3.937 degrees
  from the first hold.
- Right-foot position changed by up to 8.412 mm and orientation by 3.183 degrees.
- Relative left-to-right foot position changed by 23.376 mm and orientation by
  5.959 degrees. The configured 5 mm / 1 degree stance monitor would reject the
  route at pose 2, where relative rotation already reaches 1.791 degrees.
- Common translation averaged over both feet was only 1.837 mm. Most measured
  foot rotation was yaw; hold-window foot height changed by less than 0.12 mm.
  The saved simulation shows flat feet turning or spreading, rather than just
  the pelvis balancing over identical stationary foot frames.

This prevents using the default-arm policy result as validation of a stationary
foot reference. It does not override a user's observation of the actual robot;
the simulation and hardware require their own evidence. Relative stance checks
also cannot detect a common rigid slip of both feet.

Artifacts:

- `results/sonic_policy_preview/verification.yaml`: measured-arm diagnostic.
- `results/sonic_policy_preview_defaultarms/verification.yaml`: default-arm
  diagnostic, raw dynamics and per-pose waist metrics.
- `results/sonic_policy_preview_defaultarms/feet_verification.yaml`: actual world
  foot motion and relative stance checks.
- `results/sonic_policy_preview_defaultarms/waist_tracking.png`: comparison of
  requested and actual waist angles.

Run another new diagnostic output directory using the existing SONIC Python:

```bash
/home/kaixin/GR00T-WholeBodyControl/.venv_sim/bin/python -B \
  -m g1_auto_calibration.scripts.verify_sonic_policy \
  --arm-reference sonic-default \
  --output g1_auto_calibration/results/sonic_policy_preview_new
```

Replay the saved actual body and floating-base dynamics without running the
policy or opening a robot connection:

```bash
bash run.sh -m g1_auto_calibration.scripts.replay_sonic_trace
```

The saved trace contains all forty references and the final return/hold, about
210 seconds total. Space pauses, R restarts and Q exits. Finger joint positions
are model defaults because the body trace records only the 29 body joints.
`--check-only` verified loading 10,492 finite samples without opening a GUI.
The GUI itself was not launched during this audit. This is an offline dynamics
replay, and it does not grant physical execution approval.

## Provenance fingerprints

The inspected source repository reports HEAD
`b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`, with local changes in
`zmq_manager.hpp` and `g1_deploy_onnx_ref.cpp`. HEAD alone does not identify
this source snapshot, and no running robot binary was verified by this audit.

| File suffix | SHA-256 |
| --- | --- |
| `include/input_interface/zmq_manager.hpp` | `e0ba11f9a3897d7b9d05ec9e06dd4fc27e1b8d0fec0583ce41c03bda82a8eee8` |
| `include/policy_parameters.hpp` | `b9332adf07c2c9b75c9b1e0756e57c7a1c2a890d8bb0aa53c3f1905fb739b791` |
| `include/input_interface/zmq_packed_message_subscriber.hpp` | `35d5142bfce39c0632e96dbb6f17316a0d200f044c62f1f7e9edabfa351452c4` |
| `include/output_interface/zmq_output_handler.hpp` | `50c39ae3d7f10eb1ce8e9ef328615f0c8baf6b45851c89beb9d8c105d91f1ccf` |
| `src/g1_deploy_onnx_ref.cpp` | `546e208c43dc63814ba0bd76e8155605fec61a51dad278bf3c4856920c76fc14` |
