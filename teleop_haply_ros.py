"""
Haply Inverse3 as leader for a Franka follower running the ProteCT `haply_force_teleop` controller.

The Haply cursor (and VerseGrip orientation) is streamed to the robot through rosbridge (ROS 1),
so no local ROS installation is needed. The follower's external wrench comes back the same way
and is rendered on the Haply with the force-feedback logic of the Franka-to-Franka teleop:
contact gating, guiding/idle scaling, local damping, rate limiting and fade-out on stale data.
The follower side (Cartesian impedance, contact dependent force limits, force lock along z) runs
at 1 kHz on the robot, see README.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import orjson
import tyro
import websockets

TARGET_TOPIC = "haply_force_teleop/target"
FORCE_LOCK_TOPIC = "haply_force_teleop/force_lock"
WRENCH_TOPIC = "haply_force_teleop/wrench"
STATE_TOPIC = "haply_force_teleop/state"
SWITCH_GOAL_TOPIC = "switch_control/goal"
SWITCH_RESULT_TOPIC = "switch_control/result"
CONTROLLER_TYPE = "haply_force_teleop"

HAPLY_TIMEOUT = 0.1  # [s] stop sending targets if the Haply stream stalls
STATE_TIMEOUT = 0.5  # [s] controller state is considered stale after this
LOCK_CONFIRM_TIMEOUT = 0.5  # [s] the controller must report the force lock within this time
LOCK_REPUBLISH_PERIOD = 0.05  # [s]
DETENT_RAMP_TIME = 0.3  # [s]
ACTIVATION_DELAY = 1.0  # [s] let ROS connect the new publishers before sending a goal
# Both servers keep streaming while we shut down, which stalls the closing handshake.
CLOSE_TIMEOUT = 0.5  # [s]
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion w, x, y, z


@dataclass
class Config:
    """Haply Inverse3 -> ProteCT follower force-feedback teleop via rosbridge."""

    rosbridge_uri: str = "ws://localhost:9090"
    """rosbridge websocket of the robot PC (started by protect_robots.launch)."""
    haply_uri: str = "ws://localhost:10001"
    """Haply Inverse Service websocket."""
    controller_ns: str = "/protect/multi_mode_controller"
    """Namespace of the multi mode controller."""
    frame_id: str = "follower_link0"
    """Frame id of the published targets (follower base frame)."""
    follower: str = "follower"
    """Robot resource name of the follower, used by --activate."""
    activate: bool = False
    """Switch the follower to haply_force_teleop on start and back to stop on exit."""

    scale: float = 1.0
    """Robot metres per Haply metre."""
    position_axes: str = "-x,-y,z"
    """Robot base x,y,z expressed in Haply axes (same mapping as teleop_with_haply.py)."""
    orientation_axes: str = "z,-x,-y"
    """Robot base rotation axes expressed in VerseGrip axes (same mapping as teleop_with_haply.py)."""
    use_orientation: bool = True
    """Send VerseGrip rotations. Disable to teleoperate the position only."""
    workspace_half_extent: tuple[float, float, float] = (0.3, 0.3, 0.3)
    """[m] Box around the activation pose. Keep equal to cuboid/2 of haply_force_teleop."""

    clutch_mode: Literal["hold", "toggle", "always"] = "hold"
    """hold: the follower moves only while the clutch button is held (deadman). toggle: press to
    engage/disengage. always: always engaged. Releasing the clutch lets you reposition the Haply."""
    clutch_button: str = "a"
    """VerseGrip button used as clutch."""
    force_lock_button: str = "b"
    """VerseGrip button that toggles holding the current contact force along base z."""

    force_scale: float = 0.2
    """Haply force per follower force while guiding (clutch engaged)."""
    force_scale_idle: float = 0.0
    """Haply force per follower force while not guiding (clutch released)."""
    force_sign: float = -1.0
    """-1 renders -O_F_ext_hat_K, i.e. the force the environment applies to the follower."""
    contact_threshold: float = 4.0
    """[N] Follower force at which the feedback is half faded in."""
    contact_ramp_increase: float = 0.5
    """[1/N] Steepness of the contact ramp."""
    feedback_filter_time_constant: float = 0.01
    """[s] Low-pass filter of the received wrench."""
    max_feedback_rate: float = 40.0
    """[N/s] Rate limit of the force feedback."""
    damping: float = 1.5
    """[N s/m] Local damping of the Haply."""
    wall_stiffness: float = 800.0
    """[N/m] Virtual walls at the workspace limits, rendered on the Haply."""
    lock_detent_stiffness: float = 300.0
    """[N/m] Holds the hand along robot z while the force lock is active."""
    max_force: float = 3.0
    """[N] Maximum force commanded to the Haply."""
    wrench_timeout: float = 0.1
    """[s] Fade out the force feedback if no wrench arrives for this long."""

    publish_hz: float = 250.0
    """Rate of the targets sent to the robot."""
    status_hz: float = 2.0
    """Rate of the status line."""


def axis_matrix(spec: str) -> np.ndarray:
    """Signed permutation matrix M with robot = M @ device for a spec like "-x,-y,z"."""
    parts = [part.strip().lower() for part in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Axis spec '{spec}' must have three comma separated entries")
    matrix = np.zeros((3, 3))
    for row, part in enumerate(parts):
        axis = part.lstrip("+-")
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Axis spec '{spec}': unknown axis '{part}'")
        matrix[row, "xyz".index(axis)] = -1.0 if part.startswith("-") else 1.0
    if not np.allclose(np.abs(matrix).sum(axis=0), 1.0):
        raise ValueError(f"Axis spec '{spec}' must use every axis exactly once")
    return matrix


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-9 else IDENTITY.copy()


def map_rotation(matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Expresses the rotation q in the frame given by the signed permutation matrix."""
    vector = np.linalg.det(matrix) * (matrix @ q[1:])
    return quat_normalize(np.concatenate([[q[0]], vector]))


def contact_factor(force_norm: float, threshold: float, increase: float) -> float:
    """Same tanh contact ramp as the Franka teleop controllers (0 below, 1 above threshold)."""
    return 0.5 * (1.0 + math.tanh(increase * (force_norm - threshold)))


def vec3(value: dict | None) -> np.ndarray | None:
    if not value:
        return None
    return np.array([value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)], dtype=float)


def quat_from_dict(value: dict | None) -> np.ndarray | None:
    if not value:
        return None
    q = np.array([value.get("w", 0.0), value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)])
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
        return None
    return quat_normalize(q)


class HaplyLeader:
    """Clutch, virtual leader pose and haptic rendering. Pure logic without any I/O."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.position_map = axis_matrix(cfg.position_axes)
        self.orientation_map = axis_matrix(cfg.orientation_axes)
        self.half_extent = np.asarray(cfg.workspace_half_extent, dtype=float)

        # Clutch and virtual leader (robot base frame). Only its increments are used by the
        # follower, so it continues smoothly across clutching.
        self.engaged = False
        self._clutch_toggled = False
        self._previous_buttons: dict[str, bool] = {}
        self._anchor_position: np.ndarray | None = None
        self._anchor_orientation: np.ndarray | None = None
        self._offset_position = np.zeros(3)
        self._offset_orientation = IDENTITY.copy()
        self._wall_overshoot = np.zeros(3)
        self.virtual_position = np.zeros(3)
        self.virtual_orientation = IDENTITY.copy()
        self._last_position: np.ndarray | None = None
        self.velocity = np.zeros(3)

        # Feedback from the follower.
        self.wrench = np.zeros(6)
        self.wrench_time = -math.inf
        self.state_time = -math.inf
        self.input_ok = False
        self.lock_active = False
        self.f_des = 0.0
        self.f_meas = 0.0

        # Force lock request (the controller engages on a rising edge if in contact).
        self.lock_requested = False
        self._lock_request_time = -math.inf
        self._lock_seen_active = False

        # Rendering state.
        self._filtered_feedback = np.zeros(3)
        self._feedback = np.zeros(3)
        self._detent_anchor: np.ndarray | None = None
        self._detent_weight = 0.0
        self.force = np.zeros(3)
        self.messages: list[str] = []

    def reset_device(self) -> None:
        """Called when the Haply (re)connects: re-anchor the clutch and render from zero."""
        self.engaged = False
        self._clutch_toggled = False
        self._anchor_position = None
        self._anchor_orientation = None
        self._wall_overshoot = np.zeros(3)
        self._last_position = None
        self.velocity = np.zeros(3)
        self._filtered_feedback = np.zeros(3)
        self._feedback = np.zeros(3)
        self._detent_anchor = None
        self._detent_weight = 0.0
        self.force = np.zeros(3)

    def update_device(
        self,
        now: float,
        dt: float,
        position: np.ndarray,
        velocity: np.ndarray | None,
        orientation: np.ndarray | None,
        buttons: dict,
    ) -> None:
        """Processes one Haply sample: buttons, clutch and the virtual leader pose."""
        cfg = self.cfg

        def pressed(name: str) -> bool:
            return bool(buttons.get(name, False))

        def rising(name: str) -> bool:
            return pressed(name) and not self._previous_buttons.get(name, False)

        if cfg.clutch_mode == "hold":
            engaged = pressed(cfg.clutch_button)
        elif cfg.clutch_mode == "toggle":
            if rising(cfg.clutch_button):
                self._clutch_toggled = not self._clutch_toggled
            engaged = self._clutch_toggled
        else:
            engaged = True
        if rising(cfg.force_lock_button):
            self.toggle_force_lock(now)
        self._previous_buttons = {name: bool(value) for name, value in buttons.items()}

        if velocity is None:
            # Device without velocity in its state: filtered finite differences.
            if self._last_position is not None and dt > 0.0:
                raw = (position - self._last_position) / dt
                self.velocity = 0.8 * self.velocity + 0.2 * raw
        else:
            self.velocity = velocity
        self._last_position = position.copy()

        use_orientation = cfg.use_orientation and orientation is not None
        if engaged and (not self.engaged or self._anchor_position is None):
            # (Re)engage: continue from the current virtual pose, so the follower does not jump.
            self._anchor_position = position.copy()
            self._offset_position = self.virtual_position.copy()
            self._anchor_orientation = orientation.copy() if use_orientation else None
            self._offset_orientation = self.virtual_orientation.copy()
            if self._detent_anchor is not None:
                self._detent_anchor = position.copy()
        elif engaged and use_orientation and self._anchor_orientation is None:
            # The VerseGrip showed up while engaged.
            self._anchor_orientation = orientation.copy()
            self._offset_orientation = self.virtual_orientation.copy()
        self.engaged = engaged

        if not engaged:
            self._wall_overshoot = np.zeros(3)
            return
        raw = self._offset_position + cfg.scale * (
            self.position_map @ (position - self._anchor_position)
        )
        clamped = np.clip(raw, -self.half_extent, self.half_extent)
        self._wall_overshoot = raw - clamped
        self.virtual_position = clamped
        if use_orientation and self._anchor_orientation is not None:
            delta = quat_mul(orientation, quat_conj(self._anchor_orientation))
            self.virtual_orientation = quat_normalize(
                quat_mul(map_rotation(self.orientation_map, delta), self._offset_orientation)
            )

    def toggle_force_lock(self, now: float) -> None:
        if self.lock_requested:
            self.lock_requested = False
            self.messages.append("Force lock: release requested")
        else:
            self.lock_requested = True
            self._lock_request_time = now
            self._lock_seen_active = False
            self.messages.append("Force lock: requested")

    def update_wrench(self, now: float, wrench: np.ndarray) -> None:
        self.wrench = wrench
        self.wrench_time = now

    def update_controller_state(self, now: float, data: list[float]) -> None:
        """data = [input_ok, force_lock_engaged, lock_blend, f_des_z, f_meas_z, contact]."""
        if len(data) < 6:
            return
        self.state_time = now
        self.input_ok = data[0] > 0.5
        active = data[1] > 0.5
        if active and not self.lock_active:
            self.messages.append(f"Force lock engaged at {data[3]:.1f} N")
        elif not active and self.lock_active:
            self.messages.append("Force lock released")
        self.lock_active = active
        self._lock_seen_active = self._lock_seen_active or active
        self.f_des = data[3]
        self.f_meas = data[4]

    def check_force_lock(self, now: float) -> None:
        """Clears the request when the controller refused or released the lock."""
        if not self.lock_requested:
            return
        if self._lock_seen_active and not self.lock_active:
            self.lock_requested = False
        elif not self._lock_seen_active and now - self._lock_request_time > LOCK_CONFIRM_TIMEOUT:
            self.lock_requested = False
            self.messages.append(
                "Force lock refused: the follower needs contact along z (see the robot log)"
            )

    def render(self, now: float, dt: float, position: np.ndarray) -> np.ndarray:
        """Force [N] to command to the Haply, in Haply axes."""
        cfg = self.cfg
        dt = min(max(dt, 1e-4), 0.05)
        to_device = self.position_map.T

        # Force feedback, gated by the follower contact and scaled by guiding/idle like the
        # Franka leader. Stale data fades out through the filter.
        target = np.zeros(3)
        if now - self.wrench_time < cfg.wrench_timeout:
            follower_force = self.wrench[:3]
            contact = contact_factor(
                float(np.linalg.norm(follower_force)), cfg.contact_threshold, cfg.contact_ramp_increase
            )
            gain = cfg.force_scale if self.engaged else cfg.force_scale_idle
            target = contact * gain * cfg.force_sign * (to_device @ follower_force)
        alpha = dt / (cfg.feedback_filter_time_constant + dt)
        self._filtered_feedback += alpha * (target - self._filtered_feedback)
        step = self._filtered_feedback - self._feedback
        max_step = cfg.max_feedback_rate * dt
        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            step *= max_step / step_norm
        self._feedback += step
        force = self._feedback.copy()

        # Virtual walls at the workspace limits (overshoot converted back to Haply metres).
        force -= cfg.wall_stiffness * (to_device @ self._wall_overshoot) / cfg.scale

        # Detent along robot z while the force lock holds the contact force, so the operator's
        # hand rests instead of drifting.
        want_detent = self.lock_active and self.engaged
        if want_detent and self._detent_anchor is None:
            self._detent_anchor = position.copy()
        ramp = dt / DETENT_RAMP_TIME
        self._detent_weight = min(1.0, self._detent_weight + ramp) if want_detent else max(
            0.0, self._detent_weight - ramp
        )
        if self._detent_anchor is not None:
            if self._detent_weight <= 0.0:
                self._detent_anchor = None
            else:
                axis = to_device @ np.array([0.0, 0.0, 1.0])
                offset = float(axis @ (position - self._detent_anchor))
                force -= self._detent_weight * cfg.lock_detent_stiffness * offset * axis

        # Local damping, like the leader damping of the Franka teleop.
        force -= cfg.damping * self.velocity

        force_norm = np.linalg.norm(force)
        if force_norm > cfg.max_force:
            force *= cfg.max_force / force_norm
        self.force = force
        return force


def ros_publish(topic: str, msg: dict) -> str:
    return orjson.dumps({"op": "publish", "topic": topic, "msg": msg}).decode()


def ros_stamp() -> dict:
    now = time.time()
    return {"secs": int(now), "nsecs": int((now % 1.0) * 1e9)}


class Bridge:
    """Connects the Haply Inverse Service and rosbridge."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.leader = HaplyLeader(cfg)
        self.ns = cfg.controller_ns.rstrip("/")
        self.device_id: str | None = None
        self.haply_time = -math.inf
        self.haply_count = 0
        self.wrench_count = 0
        self.activated = False
        self._goal_counter = 0

    def topic(self, name: str) -> str:
        return f"{self.ns}/{name}"

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self.haply_loop()),
            asyncio.create_task(self.ros_loop()),
            asyncio.create_task(self.status_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---------------------------------------------------------------- Haply
    async def haply_loop(self) -> None:
        while True:
            try:
                async with websockets.connect(self.cfg.haply_uri, close_timeout=CLOSE_TIMEOUT) as ws:
                    print(f"[Haply] Connected to {self.cfg.haply_uri}")
                    self.device_id = None
                    self.leader.reset_device()
                    try:
                        await self._haply_session(ws)
                    finally:
                        # Never leave a force on the device.
                        if self.device_id is not None:
                            with contextlib.suppress(Exception):
                                await ws.send(self._force_command(np.zeros(3)))
                        self.leader.reset_device()
            except Exception as error:  # noqa: BLE001 - keep the bridge alive, retry
                print(f"[Haply] Connection error: {error!r}. Retrying in 1 s.")
            await asyncio.sleep(1.0)

    async def _haply_session(self, ws) -> None:
        last_time: float | None = None
        warned_buttons = False
        while True:
            data = orjson.loads(await ws.recv())
            now = time.monotonic()
            devices = data.get("inverse3") or []
            if not devices:
                continue
            device = devices[0]
            if self.device_id is None:
                self.device_id = device.get("device_id")
                handedness = device.get("config", {}).get("handedness", "unknown")
                print(f"[Haply] Device ID: {self.device_id}, Handedness: {handedness}")
            state = device.get("state", {})
            position = vec3(state.get("cursor_position"))
            if position is None:
                continue
            grips = data.get("wireless_verse_grip") or []
            grip_state = grips[0].get("state", {}) if grips else {}
            buttons = grip_state.get("buttons") or {}
            if not warned_buttons and self.cfg.clutch_mode != "always" and self.cfg.clutch_button not in buttons:
                print(
                    f"[Haply] WARNING: no VerseGrip button '{self.cfg.clutch_button}' found. "
                    "The follower will not move (use --clutch-mode always to teleoperate without it)."
                )
                warned_buttons = True

            dt = now - last_time if last_time is not None else 1e-3
            last_time = now
            self.leader.update_device(
                now,
                dt,
                position,
                vec3(state.get("cursor_velocity")),
                quat_from_dict(grip_state.get("orientation")),
                buttons,
            )
            force = self.leader.render(now, dt, position)
            await ws.send(self._force_command(force))
            self.haply_time = now
            self.haply_count += 1

    def _force_command(self, force: np.ndarray) -> bytes:
        return orjson.dumps(
            {
                "inverse3": [
                    {
                        "device_id": self.device_id,
                        "commands": {
                            "set_cursor_force": {
                                "values": {"x": float(force[0]), "y": float(force[1]), "z": float(force[2])}
                            }
                        },
                    }
                ]
            }
        )

    # ---------------------------------------------------------------- ROS
    async def ros_loop(self) -> None:
        while True:
            try:
                async with websockets.connect(self.cfg.rosbridge_uri, close_timeout=CLOSE_TIMEOUT) as ws:
                    print(f"[ROS] Connected to rosbridge {self.cfg.rosbridge_uri}")
                    try:
                        await self._ros_session(ws)
                    except asyncio.CancelledError:
                        if self.activated:
                            print(f"[ROS] Switching {self.cfg.follower} back to stop")
                            with contextlib.suppress(Exception):
                                await ws.send(self._switch_goal("stop"))
                                await asyncio.sleep(0.2)
                        raise
            except Exception as error:  # noqa: BLE001 - keep the bridge alive, retry
                print(f"[ROS] rosbridge connection error: {error!r}. Retrying in 1 s.")
            await asyncio.sleep(1.0)

    async def _ros_session(self, ws) -> None:
        operations = [
            {"op": "advertise", "topic": self.topic(TARGET_TOPIC), "type": "geometry_msgs/PoseStamped", "queue_size": 1},
            {"op": "advertise", "topic": self.topic(FORCE_LOCK_TOPIC), "type": "std_msgs/Bool", "queue_size": 1},
            {"op": "subscribe", "topic": self.topic(WRENCH_TOPIC), "type": "geometry_msgs/WrenchStamped", "queue_length": 1, "throttle_rate": 0},
            {"op": "subscribe", "topic": self.topic(STATE_TOPIC), "type": "std_msgs/Float64MultiArray", "queue_length": 1, "throttle_rate": 0},
        ]
        if self.cfg.activate:
            operations += [
                {"op": "advertise", "topic": self.topic(SWITCH_GOAL_TOPIC), "type": "dual_panda_multi_mode_controllers/SwitchControlActionGoal", "queue_size": 1},
                {"op": "subscribe", "topic": self.topic(SWITCH_RESULT_TOPIC), "type": "dual_panda_multi_mode_controllers/SwitchControlActionResult"},
            ]
        for operation in operations:
            await ws.send(orjson.dumps(operation).decode())

        tasks = [asyncio.create_task(self._ros_receive(ws)), asyncio.create_task(self._ros_publish(ws))]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()  # re-raise connection errors
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _ros_receive(self, ws) -> None:
        wrench_topic = self.topic(WRENCH_TOPIC)
        state_topic = self.topic(STATE_TOPIC)
        result_topic = self.topic(SWITCH_RESULT_TOPIC)
        async for raw in ws:
            message = orjson.loads(raw)
            op = message.get("op")
            if op == "status":
                print(f"[ROS] rosbridge {message.get('level', '')}: {message.get('msg', '')}")
                continue
            if op != "publish":
                continue
            topic = message.get("topic")
            body = message.get("msg", {})
            now = time.monotonic()
            if topic == wrench_topic:
                force = body["wrench"]["force"]
                torque = body["wrench"]["torque"]
                self.leader.update_wrench(
                    now,
                    np.array([force["x"], force["y"], force["z"], torque["x"], torque["y"], torque["z"]], dtype=float),
                )
                self.wrench_count += 1
            elif topic == state_topic:
                self.leader.update_controller_state(now, body.get("data", []))
            elif topic == result_topic:
                status = body.get("status", {}).get("status")
                error = body.get("result", {}).get("error", "")
                ok = status == 3  # actionlib_msgs/GoalStatus.SUCCEEDED
                print(f"[ROS] switch_control {'succeeded' if ok else f'failed (status {status})'} {error}")

    async def _ros_publish(self, ws) -> None:
        target_topic = self.topic(TARGET_TOPIC)
        lock_topic = self.topic(FORCE_LOCK_TOPIC)
        period = 1.0 / self.cfg.publish_hz
        connected_time = time.monotonic()
        last_lock_time = -math.inf
        last_lock_value: bool | None = None
        next_time = time.monotonic()
        while True:
            now = time.monotonic()
            # Only stream targets while the Haply is alive, otherwise the follower holds its pose.
            if now - self.haply_time < HAPLY_TIMEOUT:
                position = self.leader.virtual_position
                q = self.leader.virtual_orientation
                await ws.send(
                    ros_publish(
                        target_topic,
                        {
                            "header": {"stamp": ros_stamp(), "frame_id": self.cfg.frame_id},
                            "pose": {
                                "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
                                "orientation": {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])},
                            },
                        },
                    )
                )
                self.leader.check_force_lock(now)
                lock = self.leader.lock_requested
                if lock != last_lock_value or now - last_lock_time > LOCK_REPUBLISH_PERIOD:
                    await ws.send(ros_publish(lock_topic, {"data": lock}))
                    last_lock_value = lock
                    last_lock_time = now
                if self.cfg.activate and not self.activated and now - connected_time > ACTIVATION_DELAY:
                    print(f"[ROS] Switching {self.cfg.follower} to {CONTROLLER_TYPE}")
                    await ws.send(self._switch_goal(CONTROLLER_TYPE))
                    self.activated = True
            next_time += period
            delay = next_time - time.monotonic()
            if delay < -period:
                next_time = time.monotonic()  # fell behind, do not burst
            await asyncio.sleep(max(delay, 0.0))

    def _switch_goal(self, controller_type: str) -> str:
        self._goal_counter += 1
        zero_stamp = {"secs": 0, "nsecs": 0}  # the action server stamps it with its own clock
        return ros_publish(
            self.topic(SWITCH_GOAL_TOPIC),
            {
                "header": {"stamp": zero_stamp, "frame_id": ""},
                "goal_id": {"stamp": zero_stamp, "id": f"haply_bridge_{controller_type}_{self._goal_counter}_{int(time.time())}"},
                "goal": {
                    "ctrl_modes": {
                        "mode_list": [{"ctrl_type": controller_type, "ctrl_resources": [self.cfg.follower]}]
                    }
                },
            },
        )

    # ---------------------------------------------------------------- Status
    async def status_loop(self) -> None:
        period = 1.0 / self.cfg.status_hz
        while True:
            await asyncio.sleep(period)
            leader = self.leader
            for message in leader.messages:
                print(f"[Teleop] {message}")
            leader.messages.clear()
            now = time.monotonic()
            follower_ok = leader.input_ok and now - leader.state_time < STATE_TIMEOUT
            lock = "ACTIVE" if leader.lock_active else ("requested" if leader.lock_requested else "off")
            print(
                f"[Teleop] haply {self.haply_count / period:5.0f} Hz | "
                f"{'ENGAGED ' if leader.engaged else 'released'} | "
                f"follower {'tracking' if follower_ok else 'no input'} | lock {lock:9s} | "
                f"wrench {self.wrench_count / period:4.0f} Hz |F| {np.linalg.norm(leader.wrench[:3]):5.1f} N | "
                f"haply |F| {np.linalg.norm(leader.force):4.2f} N"
            )
            self.haply_count = 0
            self.wrench_count = 0


def main(cfg: Config) -> None:
    print("=== Haply -> ProteCT follower force-feedback teleop ===")
    print(f"rosbridge: {cfg.rosbridge_uri} | controller: {cfg.controller_ns}/{CONTROLLER_TYPE}")
    print(
        f"Clutch: {cfg.clutch_mode} on '{cfg.clutch_button}', force lock toggle on '{cfg.force_lock_button}'. "
        "Stop with Ctrl+C."
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(Bridge(cfg).run())
    print("[Teleop] Done.")


if __name__ == "__main__":
    main(tyro.cli(Config))
