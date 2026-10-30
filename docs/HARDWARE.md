# Hardware integration boundary

No physical arm is connected or controlled by this release. Choose the exact arm,
controller, gripper, and communication protocol before implementing its driver.
`configs/panda.json` describes the current simulated robot; it is not a profile
for an arbitrary hobby arm.

## Reuse the planner/backend separation

`RobotBackend` in `domain.py` defines three methods:

| Method | Hardware responsibility |
|---|---|
| `observe()` | Return measured/verified object states, tray names/locations, gripper state, and execution history. |
| `execute(action)` | Validate the request, execute one high-level primitive through the device driver, wait for feedback, and return an honest success/error result. |
| `close()` | Stop pending motion according to the device's supported procedure and release resources. |

The model chooses symbolic actions. Motor trajectories, inverse kinematics,
gripper actuation, and workspace/joint limits belong to the device controller.
Do not substitute an expert planning policy when a model decision fails.

Start with a measured arrangement of fixed block and tray locations. Camera
perception can later supply the same observation contract, with calibration
between the camera and robot coordinate frames. Keep physical coordinates in the
backend; the new model input uses semantic object state only.

A hardware backend needs a real transport adapter, calibrated units/joint
conventions, reachable poses, motion completion feedback/timeouts, validated
limits, and access to the arm's stop mechanism. A four-axis hobby arm cannot
reuse Panda joint angles or assume the same end-effector orientation and reach.

PyBullet currently assists grasps with a fixed attachment. Physical grasp success
must instead be checked from real feedback or observation. State must not be
updated to say a block moved merely because a command was sent. Controller tests
and simulator success do not establish hardware safety or real grasp reliability.

## Robot profiles

Copy `configs/panda.json` to document the candidate workspace and calibration.
The current simulator reads its URDF, joint/link indices, home positions, trays,
workspace, gripper settings, forces, and motion heights. Only compatible URDFs
and controllers can use that simulator implementation. Add an actual device
backend to `backends.py` once its protocol is known; no placeholder backend
pretends to control real equipment.
