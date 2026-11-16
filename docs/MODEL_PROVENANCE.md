URDF provenance
===============

`g1_auto_calibration/config/g1_29dof.urdf` is an unmodified local snapshot from
`/home/kaixin/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf`.
The source repository is [NVIDIA GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
The local source repository's top-level LICENSE and NOTICE documents, when present,
are retained alongside this provenance as `SOURCE_REPOSITORY_LICENSE` and
`SOURCE_REPOSITORY_NOTICE`. The source URDF contains no separate copyright header.
This records provenance; it does not assign a new license to third-party assets.

The snapshot preserves nominal fixed sensor links for model traceability, but the
calibration code uses only the robot joint chain to the mount. Nominal camera and
LiDAR extrinsics in this model are not treated as measurements or calibration priors.
