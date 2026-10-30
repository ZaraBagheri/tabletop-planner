"""Panda controller with explicitly assisted grasps; not a dexterous-grasp benchmark."""
import math
from pathlib import Path
import time

from .symbolic import SymbolicBackend


class BulletBackend(SymbolicBackend):
    def __init__(self, episode, gui=False, record=False, config=None):
        super().__init__(episode, config=config)
        config = self.config
        import pybullet as p
        import pybullet_data
        self.p = p
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        self.gui, self.capture = gui, record
        self.frames = []
        self.tick = 0
        self.constraint = None
        self.body_ids = {}
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
            p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            p.setTimeStep(1 / 240, physicsClientId=self.client)
            p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.client)
            p.loadURDF("plane.urdf", [0, 0, -.07], physicsClientId=self.client)
            self._box([.40, 0, -.025], [.65, .48, .025], [0.78, .80, .76, 1])
            self.arm = p.loadURDF(config.arm_urdf, useFixedBase=True,
                                  physicsClientId=self.client)
            self.ee = config.end_effector_link
            self.home = config.home
            count = p.getNumJoints(self.arm, physicsClientId=self.client)
            if any(index >= count for index in config.joint_indices + config.finger_joints + [self.ee]):
                raise ValueError("Robot profile references joints/links absent from the URDF")
            movable = [i for i in range(count) if p.getJointInfo(self.arm, i, physicsClientId=self.client)[3] >= 0]
            self.ik_indices = [movable.index(joint) for joint in config.joint_indices]
            self.joint_limits = [p.getJointInfo(self.arm, joint, physicsClientId=self.client)[8:10]
                                 for joint in config.joint_indices]
            for joint, value in zip(config.joint_indices, self.home):
                p.resetJointState(self.arm, joint, value, physicsClientId=self.client)
            self._joints(self.home)
            self._fingers(config.gripper_open)
            self.orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
            self.tray_bodies = []
            for name, position in self.trays.items():
                x, y, _ = position
                color = [.50, .72, .66, 1] if name == "left_tray" else [.67, .61, .81, 1]
                size = config.tray_half_size
                self.tray_bodies.append(self._box([x, y, .004], [size, size, .004], color))
                for dx, dy, half in [(0, size, [size+.003, .004, .016]),
                                     (0, -size, [size+.003, .004, .016]),
                                     (size, 0, [.004, size, .016]),
                                     (-size, 0, [.004, size, .016])]:
                    self.tray_bodies.append(self._box([x+dx, y+dy, .016], half, color))
            colors = {"red": [.85, .18, .15, 1], "blue": [.15, .38, .85, 1],
                      "green": [.22, .65, .28, 1], "yellow": [.93, .72, .12, 1]}
            for obj in self.objects.values():
                self.body_ids[obj["id"]] = self._box(obj["position"], [config.block_half_size]*3, colors[obj["color"]], .04)
            p.resetDebugVisualizerCamera(.95, 40, -42, [.43, 0, .1], physicsClientId=self.client)
            self._step(180)
        except Exception:
            self.close()
            raise

    def _box(self, position, half, color, mass=0):
        p = self.p
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client)
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision,
                                 baseVisualShapeIndex=visual, basePosition=position, physicsClientId=self.client)
        p.changeDynamics(body, -1, lateralFriction=1.0, physicsClientId=self.client)
        return body

    def _joints(self, positions):
        if any(not math.isfinite(value) or (low < high and not low <= value <= high)
               for value, (low, high) in zip(positions, self.joint_limits)):
            raise ValueError("Requested joint positions exceed URDF joint limits")
        count = len(self.config.joint_indices)
        self.p.setJointMotorControlArray(self.arm, self.config.joint_indices, self.p.POSITION_CONTROL,
                                         targetPositions=list(positions), forces=[self.config.arm_force]*count,
                                         positionGains=[.15]*count, physicsClientId=self.client)

    def _fingers(self, width):
        self.finger_width = width
        count = len(self.config.finger_joints)
        self.p.setJointMotorControlArray(self.arm, self.config.finger_joints, self.p.POSITION_CONTROL,
                                         targetPositions=[width]*count, forces=[self.config.gripper_force]*count,
                                         physicsClientId=self.client)

    def _step(self, steps):
        for _ in range(steps):
            self.p.stepSimulation(physicsClientId=self.client)
            self.tick += 1
            if self.capture and self.tick % 16 == 0:
                self.frames.append(self.image())
            if self.gui:
                time.sleep(1 / 240)

    def _ee_position(self):
        return self.p.getLinkState(self.arm, self.ee, computeForwardKinematics=True,
                                   physicsClientId=self.client)[4]

    def _move(self, destination):
        # Bounded Cartesian waypoints, inverse kinematics and position motor control.
        if not self.config.contains(destination):
            return False
        start = self._ee_position()
        segments = max(10, int(math.dist(start, destination) / .012))
        for i in range(1, segments+1):
            point = [a+(b-a)*i/segments for a, b in zip(start, destination)]
            joints = self.p.calculateInverseKinematics(
                self.arm, self.ee, point, self.orientation, maxNumIterations=120,
                residualThreshold=.0001, physicsClientId=self.client)
            try:
                self._joints([joints[index] for index in self.ik_indices])
            except ValueError:
                return False
            self._step(12)
        self._step(100)
        return math.dist(self._ee_position(), destination) < .025

    def observe(self):
        # Read object poses from the physics engine; do not infer success from commands.
        for obj_id, body in self.body_ids.items():
            position = self.p.getBasePositionAndOrientation(body, physicsClientId=self.client)[0]
            obj = self.objects[obj_id]
            obj["position"] = [round(v, 3) for v in position]
            obj["zone"] = "held" if self.holding == obj_id else "table"
            if not self.holding == obj_id:
                for name, center in self.trays.items():
                    if (abs(position[0]-center[0]) < self.config.tray_membership_half_size and abs(position[1]-center[1]) < self.config.tray_membership_half_size
                            and 0 < position[2] < .09):
                        obj["zone"] = name
        return super().observe()

    def execute(self, action):
        self.observe()
        error = self.validate(action)
        if error:
            return self.record(action, error)
        p = self.p
        if action.action == "pick":
            body = self.body_ids[action.target]
            xyz = p.getBasePositionAndOrientation(body, physicsClientId=self.client)[0]
            self._fingers(self.config.gripper_open)
            if not self._move([xyz[0], xyz[1], self.config.approach_height]) or not self._move([xyz[0], xyz[1], max(self.config.workspace_lower[2], self.config.block_half_size+.015, xyz[2]+.008)]):
                return self.record(action, "APPROACH_FAILED")
            self._fingers(self.config.gripper_closed)
            self._step(100)
            # Attach only after a close approach. This is a deliberate grasp aid,
            # not a claim that friction/contact grasping has been learned.
            ee = p.getLinkState(self.arm, self.ee, computeForwardKinematics=True, physicsClientId=self.client)
            obj_pos, obj_rot = p.getBasePositionAndOrientation(body, physicsClientId=self.client)
            if math.dist(ee[4], obj_pos) > .055:
                return self.record(action, "GRASP_MISSED")
            inv = p.invertTransform(ee[4], ee[5])
            rel_pos, rel_rot = p.multiplyTransforms(inv[0], inv[1], obj_pos, obj_rot)
            self.constraint = p.createConstraint(self.arm, self.ee, body, -1, p.JOINT_FIXED,
                                                   [0, 0, 0], rel_pos, [0, 0, 0],
                                                   parentFrameOrientation=rel_rot,
                                                   physicsClientId=self.client)
            p.changeConstraint(self.constraint, maxForce=120, physicsClientId=self.client)
            self.holding = action.target
            if not self._move([xyz[0], xyz[1], self.config.approach_height]):
                return self.record(action, "LIFT_FAILED")
            self.observe()
            return self.record(action)
        if action.action == "place":
            obj_id = self.holding
            x, y, _ = self.trays[action.target]
            if not self._move([x, y, self.config.approach_height]) or not self._move([x, y, self.config.release_height]):
                return self.record(action, "PLACE_APPROACH_FAILED")
            p.removeConstraint(self.constraint, physicsClientId=self.client)
            self.constraint = None
            self.holding = None
            self._fingers(self.config.gripper_open)
            self._step(120)
            retreated = self._move([x, y, self.config.approach_height])
            self._step(120)
            self.observe()
            error = None if self.objects[obj_id]["zone"] == action.target else "MISSED_TRAY"
            if not retreated and error is None:
                error = "RETREAT_FAILED"
            return self.record(action, error, obj_id)
        return self.record(action)

    def image(self):
        from PIL import Image
        import numpy as np
        p = self.p
        view = p.computeViewMatrix([1.25, -1.25, 1.0], [.30, 0, .30], [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(49, 4/3, .05, 3)
        rgba = p.getCameraImage(640, 480, view, projection, renderer=p.ER_TINY_RENDERER,
                                physicsClientId=self.client)[2]
        # Source-built PyBullet may return tuples rather than a NumPy array.
        pixels = np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)
        return Image.fromarray(pixels).convert("RGB")

    def save_media(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.image().save(directory / "final.png")
        if self.frames:
            frames = [im.resize((480, 360)) for im in self.frames]
            frames[0].save(directory / "replay.gif", save_all=True, append_images=frames[1:],
                           duration=67, loop=0, optimize=False)

    def close(self):
        if self.p.isConnected(self.client):
            self.p.disconnect(self.client)
