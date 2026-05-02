"""
Haply Inverse3 to Panda End-Effector Teleop Control

Uses the Haply Inverse3 haptic device to control the Panda arm's
end-effector position in simulation via inverse kinematics.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple
import numpy as np
import orjson
import robotic as ry
import tyro
import websockets


@dataclass
class HaplyState:
    """Thread-safe container for Haply device state."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(4))
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    connected: bool = False
    device_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_position(self, x: float, y: float, z: float) -> None:
        with self.lock:
            self.position = np.array([x, y, z])

    def update_orientation(self, w: float, x: float, y: float, z: float) -> None:
        with self.lock:
            self.orientation = np.array([w, x, y, z])

    def update_buttons(self, buttons: np.ndarray) -> None:
        with self.lock:
            self.buttons = buttons

    def get_position(self) -> np.ndarray:
        with self.lock:
            return self.position.copy()

    def get_orientation(self) -> np.ndarray:
        with self.lock:
            return self.orientation.copy()

    def get_buttons(self) -> np.ndarray:
        with self.lock:
            return self.buttons.copy()

    def update_force(self, force: np.ndarray) -> None:
        with self.lock:
            # Ensure native Python floats for JSON serialization.
            force_magnitude = np.linalg.norm(force)
            force_direction = force / (force_magnitude + 1e-8)
            force_clipped = np.clip(force_magnitude, 0.0, 0.1) * force_direction
            self.force = np.asarray(force_clipped, dtype=float)


class HaplyClient:
    """Async WebSocket client for Haply Inverse3 device."""

    def __init__(self, state: HaplyState, uri: str = "ws://localhost:10001"):
        self.state = state
        self.uri = uri
        self._running = False

    async def run(self) -> None:
        """Main async loop to read Haply position and send zero-force commands."""
        self._running = True
        first_message = True

        try:
            async with websockets.connect(self.uri) as ws:
                self.state.connected = True
                print(f"[Haply] Connected to {self.uri}")

                while self._running:
                    # Receive state from device
                    response = await ws.recv()
                    data = orjson.loads(response)

                    # Get Inverse3 device data
                    inverse3_devices = data.get("inverse3", [])
                    if not inverse3_devices:
                        if first_message:
                            print("[Haply] No Inverse3 device found!")
                        continue
                    inverse3_data = inverse3_devices[0]

                    verse_grip_devices = data.get("wireless_verse_grip", [])
                    verse_grip_data = (
                        verse_grip_devices[0] if verse_grip_devices else None
                    )

                    # First message: get device ID and config
                    if first_message:
                        first_message = False
                        self.state.device_id = inverse3_data.get("device_id")
                        handedness = inverse3_data.get("config", {}).get(
                            "handedness", "unknown"
                        )
                        print(
                            f"[Haply] Device ID: {self.state.device_id}, Handedness: {handedness}"
                        )

                    # Extract cursor position
                    cursor_pos = inverse3_data.get("state", {}).get(
                        "cursor_position", {}
                    )
                    x = cursor_pos.get("x", 0.0)
                    y = cursor_pos.get("y", 0.0)
                    z = cursor_pos.get("z", 0.0)
                    self.state.update_position(x, y, z)

                    # Extract buttons and orientation from Wireless Verse Grip device
                    if verse_grip_data:
                        buttons = verse_grip_data.get("state", {}).get("buttons", {})
                        orientation = verse_grip_data.get("state", {}).get(
                            "orientation", {}
                        )
                    else:
                        buttons = np.zeros(1)
                        orientation = np.zeros(4)
                    self.state.update_buttons(buttons)
                    self.state.update_orientation(
                        orientation["w"],
                        orientation["x"],
                        orientation["y"],
                        orientation["z"],
                    )

                    request_msg = {
                        "inverse3": [
                            {
                                "device_id": self.state.device_id,
                                "commands": {
                                    "set_cursor_force": {
                                        "values": {
                                            "x": float(self.state.force[0]),
                                            "y": float(self.state.force[1]),
                                            "z": float(self.state.force[2]),
                                        }
                                    }
                                },
                            }
                        ]
                    }
                    await ws.send(orjson.dumps(request_msg))

        except Exception as e:
            print(f"[Haply] Connection error: {e}")
        finally:
            self.state.connected = False
            print("[Haply] Disconnected")

    def stop(self) -> None:
        self._running = False


def run_haply_client_thread(client: HaplyClient) -> None:
    """Run the Haply client in a separate thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(client.run())
    finally:
        loop.close()


def compute_komo_ik(
    C: ry.Config,
    target_pose: np.ndarray,
    qHome: np.ndarray,
    ee_frame: str = "l_gripper",
    with_collisions: bool = False,
) -> Tuple[np.ndarray, ry.SolverReturn]:
    """
    Compute IK using KOMO optimization.

    Args:
        C: Robot configuration
        target_pos: Desired end-effector position [x, y, z]
        ee_frame: Name of the end-effector frame
        with_collisions: Enable collision avoidance constraints

    Returns:
        Joint configuration if IK is feasible, None otherwise
    """
    q0 = C.getJointState()

    # Update target frame position
    C.getFrame("target").setPose(target_pose.tolist())

    # Create KOMO for single-step IK
    komo = ry.KOMO(C, 1, 1, 0, with_collisions)

    # Stay close to current configuration (regularization)
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0)

    # Stay close to home position
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], qHome)

    # End-effector pose should match target
    komo.addObjective([], ry.FS.poseDiff, [ee_frame, "target"], ry.OT.eq, [1e1])

    # Add collision avoidance if enabled
    if with_collisions:
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])

    solver = ry.NLP_Solver(komo.nlp(), verbose=0)
    solver.setInitialization(q0)
    ret = solver.solve()

    q_sol = komo.getPath()[0]
    return q_sol, ret


def haply_to_robot_pose(
    haply_pos: np.ndarray,
    scale: float,
    offset: np.ndarray,
    haply_orientation: np.ndarray,
) -> np.ndarray:
    """
    Map Haply device pose to robot end-effector workspace pose.

    Args:
        haply_pos: Position from Haply device [x, y, z]
        scale: Scaling factor (Haply is small workspace, robot is large workspace)
        offset: Base offset in robot frame [x, y, z]
        haply_orientation: Orientation from Haply device [w, x, y, z]

    Returns:
        Pose in robot frame [x, y, z, w, x, y, z]
    """

    robot_pos = np.array(
        [
            -haply_pos[0] * scale, 
            -haply_pos[1] * scale, 
            haply_pos[2] * scale, 
        ]
    )

    robot_pos = robot_pos + offset
    robot_pose = np.concatenate(
        [
            robot_pos,
            [
                haply_orientation[0],
                haply_orientation[3],
                -haply_orientation[1],
                -haply_orientation[2],
            ],
        ]
    )

    return robot_pose


def main(
    teleop_in_real: bool = False,
    haply_to_robot_workspace_scale: float = 5.0,
    move_to_time_cost: float = 20.0,
    sync_hz: float = 120.0,
    gripper_speed: float = 0.4,
    force_feedback: bool = False,
    avoid_self_collisions: bool = False,
    wrench_log_frequency_hz: float = 2.0,
) -> None:
    """
    Main teleop loop.

    Args:
        teleop_in_real: If True, teleop in real robot, otherwise in simulation
        haply_to_robot_workspace_scale: Scaling factor for Haply workspace to robot workspace
        move_to_time_cost: Time cost for moveTo operation
        sync_hz: Teleop control loop frequency
        gripper_speed: Gripper speed
        force_feedback: Enable force feedback from robot to Haply
        avoid_self_collisions: Enable collision avoidance
        wrench_log_frequency_hz: Frequency for logging wrench
    """
    print("=== Haply Panda Ranger Teleop ===")
    print(f"Mode: {'REAL' if teleop_in_real else 'SIMULATION'}")

    # Load robot configuration
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/panda_ranger.g"))
    C.selectJoints([f"l_panda_joint{i}" for i in range(1, 8)])
    ry.params_add({"botsim/verbose": 2})
    bot = ry.BotOp(C, useRealRobot=teleop_in_real, )
    bot.moveTo(bot.get_qHome(), timeCost=1.0)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # Add target marker frame
    target = C.addFrame("target")
    target.setShape(ry.ST.marker, [0.1])
    target.setColor([1, 0, 0])  # Red marker
    target_initial_position = C.getFrame("l_gripper").getPosition()
    target.setPosition(target_initial_position.tolist())  # Initial position

    # Initialize Haply client in separate thread
    haply_state = HaplyState()
    haply_client = HaplyClient(haply_state)
    haply_thread = threading.Thread(
        target=run_haply_client_thread, args=(haply_client,), daemon=True
    )
    haply_thread.start()

    # Wait briefly for Haply connection
    print("[Haply] Waiting for device connection...")
    time.sleep(1.0)

    # Store neutral position for relative control
    haply_neutral_position = np.zeros(3)
    if not haply_state.connected:
        print(
            "[Haply] WARNING: Device not connected. Running in demo mode (target stays fixed)."
        )
    else:
        haply_neutral_position = haply_state.get_position()
        print(f"[Haply] Neutral position: {haply_neutral_position}")

    print("\n[Teleop] Starting control loop. Press Ctrl+C to exit.")
    print("[Teleop] Move the Haply device to control the Panda end-effector.\n")

    try:
        gripper_mode = 'open'
        last_wrench_log = 0.0
        first_q_new = None
        while True:
            bot.sync(C, 1.0 / sync_hz)

            # Get Haply position and map to robot coordinates
            if haply_state.connected:
                haply_pos = haply_state.get_position() - haply_neutral_position
                haply_orientation = haply_state.get_orientation()
                ee_target_pose = haply_to_robot_pose(
                    haply_pos,
                    scale=haply_to_robot_workspace_scale,
                    offset=target_initial_position,
                    haply_orientation=haply_orientation,
                )
                haply_buttons = haply_state.get_buttons()
                if haply_buttons['a']:
                    gripper_mode = 'close'
                elif haply_buttons['b']:
                    gripper_mode = 'open'
            else:
                raise ValueError("Haply device not connected")

            q_new, ret = compute_komo_ik(
                C,
                ee_target_pose,
                bot.get_qHome(),
                ee_frame="l_gripper",
                with_collisions=avoid_self_collisions,
            )
            if ret.feasible:
                bot.moveTo(q_new, timeCost=move_to_time_cost, overwrite=True)
                if gripper_mode == 'close':
                    bot.gripperMove(ry._left, width=0.0, speed=gripper_speed)
                elif gripper_mode == 'open':
                    bot.gripperMove(ry._left, width=0.06, speed=gripper_speed)
            else:
                print(f"[Teleop] IK failed: {ret}")

            # ---------------- In alpha development ----------------
            if teleop_in_real and force_feedback:
                wrench_external = bot.get_wrenchExternal()
                haply_state.update_force(wrench_external[0:3])
                if time.time() - last_wrench_log > 1.0 / wrench_log_frequency_hz:
                    print(f"[Teleop] Haply force [Fx,Fy,Fz]: {wrench_external[0:3]}")
                    last_wrench_log = time.time()
            # ----------------------------------------------------

    except KeyboardInterrupt:
        print("\n[Teleop] Shutting down...")
    finally:
        haply_client.stop()
        del bot
        del C
        print("[Teleop] Done.")


if __name__ == "__main__":
    tyro.cli(main)
