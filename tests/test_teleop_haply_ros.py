"""Tests for teleop_haply_ros.py. Run with: uv run --with pytest pytest tests"""

import asyncio
import math
import sys
from pathlib import Path

import numpy as np
import orjson
import pytest
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teleop_haply_ros as bridge_module  # noqa: E402
from teleop_haply_ros import (  # noqa: E402
    IDENTITY,
    Bridge,
    Config,
    HaplyLeader,
    axis_matrix,
    map_rotation,
    quat_mul,
)

DT = 0.001


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * axis])


def step(leader, now, position, buttons, orientation=None, velocity=np.zeros(3)):
    leader.update_device(now, DT, np.asarray(position, dtype=float), velocity, orientation, buttons)
    return leader.render(now, DT, np.asarray(position, dtype=float))


def test_axis_matrix():
    np.testing.assert_array_equal(axis_matrix("-x,-y,z"), np.diag([-1.0, -1.0, 1.0]))
    np.testing.assert_array_equal(axis_matrix("z,-x,-y"), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    for bad in ("x,y", "x,x,z", "x,y,w"):
        with pytest.raises(ValueError):
            axis_matrix(bad)


def test_rotation_mapping_matches_original_script():
    # teleop_with_haply.py maps the VerseGrip quaternion as [w, z, -x, -y].
    q = np.array([0.9, 0.1, 0.2, 0.3])
    q /= np.linalg.norm(q)
    expected = np.array([q[0], q[3], -q[1], -q[2]])
    np.testing.assert_allclose(map_rotation(axis_matrix("z,-x,-y"), q), expected)


def test_clutch_moves_virtual_leader_only_while_engaged_and_never_jumps():
    leader = HaplyLeader(Config(scale=2.0))
    step(leader, 0.0, [0.0, 0.0, 0.0], {"a": True})
    step(leader, 0.1, [0.01, 0.0, 0.02], {"a": True})
    # Default mapping: robot = (-x, -y, z) * scale.
    np.testing.assert_allclose(leader.virtual_position, [-0.02, 0.0, 0.04])

    step(leader, 0.2, [0.05, 0.05, 0.05], {"a": False})  # repositioning, clutch released
    np.testing.assert_allclose(leader.virtual_position, [-0.02, 0.0, 0.04])

    step(leader, 0.3, [0.05, 0.05, 0.05], {"a": True})  # re-engage somewhere else: no jump
    np.testing.assert_allclose(leader.virtual_position, [-0.02, 0.0, 0.04])
    step(leader, 0.4, [0.05, 0.05, 0.06], {"a": True})
    np.testing.assert_allclose(leader.virtual_position, [-0.02, 0.0, 0.06])


def test_reconnect_reanchors_and_renders_from_zero():
    leader = HaplyLeader(Config(clutch_mode="always"))
    step(leader, 0.0, [0.0, 0.0, 0.0], {})
    step(leader, 0.1, [0.0, 0.0, 0.01], {})
    for i in range(200):
        leader.update_wrench(0.1 + i * DT, np.array([0, 0, -10.0, 0, 0, 0]))
        step(leader, 0.1 + i * DT, [0.0, 0.0, 0.01], {})
    assert np.linalg.norm(leader.force) > 1.0

    leader.reset_device()  # Haply reconnected, the device moved meanwhile
    force = step(leader, 5.0, [0.05, 0.05, 0.05], {})
    np.testing.assert_allclose(leader.virtual_position, [0.0, 0.0, 0.01])
    np.testing.assert_allclose(force, [0.0, 0.0, 0.0])
    step(leader, 5.1, [0.05, 0.05, 0.06], {})
    np.testing.assert_allclose(leader.virtual_position, [0.0, 0.0, 0.02])


def test_toggle_clutch():
    leader = HaplyLeader(Config(clutch_mode="toggle"))
    step(leader, 0.0, [0.0, 0.0, 0.0], {"a": True})
    step(leader, 0.1, [0.0, 0.0, 0.0], {"a": False})
    assert leader.engaged
    step(leader, 0.2, [0.0, 0.0, 0.01], {"a": False})
    np.testing.assert_allclose(leader.virtual_position, [0.0, 0.0, 0.01])
    step(leader, 0.3, [0.0, 0.0, 0.01], {"a": True})
    assert not leader.engaged


def test_relative_orientation_is_mapped_to_robot_axes():
    leader = HaplyLeader(Config())
    step(leader, 0.0, [0, 0, 0], {"a": True}, orientation=IDENTITY)
    # Rotating the VerseGrip about its z axis rotates the robot about base x ("z,-x,-y").
    step(leader, 0.1, [0, 0, 0], {"a": True}, orientation=axis_angle_quat([0, 0, 1], 0.3))
    np.testing.assert_allclose(leader.virtual_orientation, axis_angle_quat([1, 0, 0], 0.3), atol=1e-12)


def test_orientation_continues_across_clutching():
    leader = HaplyLeader(Config())
    step(leader, 0.0, [0, 0, 0], {"a": True}, orientation=IDENTITY)
    step(leader, 0.1, [0, 0, 0], {"a": True}, orientation=axis_angle_quat([0, 0, 1], 0.2))
    step(leader, 0.2, [0, 0, 0], {"a": False}, orientation=axis_angle_quat([0, 0, 1], -0.5))
    step(leader, 0.3, [0, 0, 0], {"a": True}, orientation=axis_angle_quat([0, 0, 1], -0.5))
    np.testing.assert_allclose(leader.virtual_orientation, axis_angle_quat([1, 0, 0], 0.2), atol=1e-12)
    step(leader, 0.4, [0, 0, 0], {"a": True}, orientation=axis_angle_quat([0, 0, 1], -0.4))
    np.testing.assert_allclose(leader.virtual_orientation, axis_angle_quat([1, 0, 0], 0.3), atol=1e-12)


def settle_feedback(leader, wrench, engaged=True, steps=400):
    force = np.zeros(3)
    for i in range(steps):
        now = 1.0 + i * DT
        leader.update_wrench(now, np.asarray(wrench, dtype=float))
        force = step(leader, now, [0, 0, 0], {"a": engaged})
    return force


def test_force_feedback_direction_and_scale():
    # Follower pressing down on a surface: O_F_ext_hat_K z = -10 N (robot on environment).
    leader = HaplyLeader(Config(force_scale=0.2, max_force=5.0))
    force = settle_feedback(leader, [0, 0, -10, 0, 0, 0])
    contact = 0.5 * (1.0 + math.tanh(0.5 * (10.0 - 4.0)))
    np.testing.assert_allclose(force, [0.0, 0.0, 0.2 * 10.0 * contact], atol=1e-6)

    # Environment pushes the follower towards robot +x -> the hand is pushed towards robot +x too.
    leader = HaplyLeader(Config(force_scale=0.2, max_force=5.0))
    force = settle_feedback(leader, [-10, 0, 0, 0, 0, 0])
    robot_direction = axis_matrix("-x,-y,z") @ force
    assert robot_direction[0] > 1.9 and abs(robot_direction[1]) < 1e-9


def test_force_feedback_is_contact_gated_and_idle_scaled():
    leader = HaplyLeader(Config())
    assert np.linalg.norm(settle_feedback(leader, [0, 0, -1, 0, 0, 0])) < 0.01  # sensor noise level
    leader = HaplyLeader(Config(force_scale_idle=0.0))
    assert np.linalg.norm(settle_feedback(leader, [0, 0, -10, 0, 0, 0], engaged=False)) < 1e-9


def test_force_feedback_is_rate_limited_saturated_and_fades_when_stale():
    cfg = Config(force_scale=1.0, max_force=3.0, max_feedback_rate=40.0, feedback_filter_time_constant=0.0)
    leader = HaplyLeader(cfg)
    step(leader, 0.0, [0, 0, 0], {"a": True})
    leader.update_wrench(0.0, np.array([0, 0, -10.0, 0, 0, 0]))
    first = step(leader, DT, [0, 0, 0], {"a": True})
    assert np.linalg.norm(first) <= 40.0 * DT + 1e-9
    force = settle_feedback(leader, [0, 0, -10, 0, 0, 0], steps=1000)
    assert np.linalg.norm(force) == pytest.approx(3.0)

    # No new wrench: fade out.
    for i in range(1000):
        force = step(leader, 10.0 + i * DT, [0, 0, 0], {"a": True})
    assert np.linalg.norm(force) < 1e-6


def test_workspace_wall_clamps_and_pushes_back():
    cfg = Config(workspace_half_extent=(0.05, 0.05, 0.05), wall_stiffness=100.0, damping=0.0)
    leader = HaplyLeader(cfg)
    step(leader, 0.0, [0, 0, 0], {"a": True})
    force = step(leader, 0.1, [0, 0, 0.07], {"a": True})
    np.testing.assert_allclose(leader.virtual_position, [0, 0, 0.05])
    np.testing.assert_allclose(force, [0, 0, -100.0 * 0.02], atol=1e-9)
    step(leader, 0.2, [0, 0, 0.06], {"a": True})  # coming back leaves the wall immediately
    np.testing.assert_allclose(leader.virtual_position, [0, 0, 0.05])
    step(leader, 0.3, [0, 0, 0.04], {"a": True})
    np.testing.assert_allclose(leader.virtual_position, [0, 0, 0.04])


def test_damping_opposes_velocity():
    leader = HaplyLeader(Config(damping=2.0))
    force = step(leader, 0.0, [0, 0, 0], {"a": True}, velocity=np.array([0.1, 0.0, 0.0]))
    np.testing.assert_allclose(force, [-0.2, 0.0, 0.0])


def test_force_lock_request_is_cleared_when_refused_or_released():
    leader = HaplyLeader(Config())
    step(leader, 0.0, [0, 0, 0], {"a": True, "b": True})
    assert leader.lock_requested
    leader.check_force_lock(0.3)
    assert leader.lock_requested
    leader.check_force_lock(0.6)  # controller never engaged -> refused
    assert not leader.lock_requested

    step(leader, 1.0, [0, 0, 0], {"a": True, "b": False})
    step(leader, 1.1, [0, 0, 0], {"a": True, "b": True})
    assert leader.lock_requested
    leader.update_controller_state(1.2, [1, 1, 0.2, -5.0, -5.0, 1.0])
    leader.check_force_lock(2.0)
    assert leader.lock_requested and leader.lock_active
    leader.update_controller_state(2.1, [1, 0, 1.0, -5.0, -0.1, 0.0])  # contact lost on the robot
    leader.check_force_lock(2.1)
    assert not leader.lock_requested

    step(leader, 3.0, [0, 0, 0], {"a": True, "b": False})
    step(leader, 3.1, [0, 0, 0], {"a": True, "b": True})
    step(leader, 3.2, [0, 0, 0], {"a": True, "b": False})
    step(leader, 3.3, [0, 0, 0], {"a": True, "b": True})  # second press releases
    assert not leader.lock_requested


def test_detent_holds_hand_along_robot_z_while_locked():
    leader = HaplyLeader(Config(damping=0.0, lock_detent_stiffness=300.0))
    step(leader, 0.0, [0, 0, 0], {"a": True})
    leader.update_controller_state(0.0, [1, 1, 1.0, -5.0, -5.0, 1.0])
    for i in range(500):
        force = step(leader, 0.001 * i, [0.0, 0.0, 0.0], {"a": True})
    force = step(leader, 0.6, [0.0, 0.0, 0.004], {"a": True})
    assert force[2] == pytest.approx(-300.0 * 0.004)
    # Motion along robot x is not held.
    force = step(leader, 0.601, [0.004, 0.0, 0.0], {"a": True})
    assert abs(force[0]) < 1e-9


def test_end_to_end_with_fake_haply_and_rosbridge(monkeypatch):
    monkeypatch.setattr(bridge_module, "ACTIVATION_DELAY", 0.1)
    result = asyncio.run(asyncio.wait_for(run_end_to_end(), timeout=20.0))
    ros_ops, haply_commands = result

    def topics(op):
        return {o["topic"] for o in ros_ops if o["op"] == op}

    ns = "/protect/multi_mode_controller/"
    assert {ns + "haply_force_teleop/target", ns + "haply_force_teleop/force_lock"} <= topics("advertise")
    assert {ns + "haply_force_teleop/wrench", ns + "haply_force_teleop/state"} <= topics("subscribe")

    targets = [o["msg"] for o in ros_ops if o["op"] == "publish" and o["topic"].endswith("/target")]
    assert len(targets) > 50
    # The fake Haply moved +2 cm along its x axis while engaged -> robot -2 cm along x.
    assert targets[-1]["pose"]["position"]["x"] == pytest.approx(-0.02, abs=1e-6)
    assert targets[-1]["header"]["frame_id"] == "follower_link0"

    goals = [o["msg"]["goal"]["ctrl_modes"]["mode_list"][0] for o in ros_ops if o["topic"].endswith("switch_control/goal") and o["op"] == "publish"]
    assert goals[0] == {"ctrl_type": "haply_force_teleop", "ctrl_resources": ["follower"]}
    assert goals[-1]["ctrl_type"] == "stop"

    forces = [c["inverse3"][0]["commands"]["set_cursor_force"]["values"] for c in haply_commands]
    assert all(c["inverse3"][0]["device_id"] == "haply-test" for c in haply_commands)
    # Follower presses down with 10 N -> the operator feels about 0.2 * 10 N upwards.
    assert max(f["z"] for f in forces) == pytest.approx(2.0, abs=0.05)
    # Shutdown leaves no force on the device.
    assert forces[-1] == {"x": 0.0, "y": 0.0, "z": 0.0}


async def run_end_to_end():
    ros_ops = []
    haply_commands = []
    ns = "/protect/multi_mode_controller/"

    async def fake_rosbridge(ws, *args):
        async def send_feedback():
            while True:
                await ws.send(orjson.dumps({"op": "publish", "topic": ns + "haply_force_teleop/wrench", "msg": {
                    "header": {}, "wrench": {"force": {"x": 0, "y": 0, "z": -10.0}, "torque": {"x": 0, "y": 0, "z": 0}}}}).decode())
                await ws.send(orjson.dumps({"op": "publish", "topic": ns + "haply_force_teleop/state", "msg": {
                    "layout": {}, "data": [1, 0, 0, 0, -10.0, 1.0]}}).decode())
                await asyncio.sleep(0.004)

        feedback = asyncio.create_task(send_feedback())
        try:
            async for raw in ws:
                ros_ops.append(orjson.loads(raw))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            feedback.cancel()

    async def fake_haply(ws, *args):
        async def receive():
            async for raw in ws:
                haply_commands.append(orjson.loads(raw))

        receiver = asyncio.create_task(receive())
        try:
            for i in range(1500):
                x = min(i, 200) * 0.0001  # move 2 cm, then hold still
                await ws.send(orjson.dumps({
                    "inverse3": [{"device_id": "haply-test", "config": {"handedness": "right"}, "state": {
                        "cursor_position": {"x": x, "y": 0.0, "z": 0.0},
                        "cursor_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}}}],
                    "wireless_verse_grip": [{"state": {"buttons": {"a": True, "b": False},
                                                       "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}}],
                }))
                await asyncio.sleep(0.001)
            await ws.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await asyncio.gather(receiver, return_exceptions=True)

    async with websockets.serve(fake_rosbridge, "127.0.0.1", 0) as ros_server, websockets.serve(
        fake_haply, "127.0.0.1", 0
    ) as haply_server:
        ros_port = list(ros_server.sockets)[0].getsockname()[1]
        haply_port = list(haply_server.sockets)[0].getsockname()[1]
        cfg = Config(
            rosbridge_uri=f"ws://127.0.0.1:{ros_port}",
            haply_uri=f"ws://127.0.0.1:{haply_port}",
            activate=True,
            status_hz=1.0,
        )
        task = asyncio.create_task(Bridge(cfg).run())
        await asyncio.sleep(2.5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0.2)
    return ros_ops, haply_commands
