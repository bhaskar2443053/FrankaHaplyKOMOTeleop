# Mobile Base Teleop

This is a LIS-internal repository which contains a script for teleoperating the arm of the [Panda Ranger](https://github.com/MarcToussaint/rai-robotModels/blob/b68f6c70c06ca19bec64f810a2342024043f8ead/scenarios/panda_ranger.g) mobile base present in the LIS lab using a [Haply Inverse3](https://www.haply.co/inverse3/), a 6D space mouse providing linear haptic feedback.

<p align="center">
  <a href="./panda_ranger_haply_teleop.mov">
    <img src="./panda_ranger_haply_teleop_preview.gif" alt="Teleop demo preview" width="70%" />
  </a>
</p>

## Software Setup

The script was developed and tested on LIS's "**hackathon notebook**", a Lenovo ThinkPad X13 Gen4 running Ubuntu 24.04.3 LTS and a realtime Linux kernel 6.8.1-1015-realtime for interfacing with the Franka control library.
Note that this notebook also runs other OSs, so 
- make sure to boot Ubuntu 24. 
- Then log into the the user "Mobile Manipulator" with the password "panda_ranger".
- The repo is located at /home/mobile_manipulator/git/MobileBaseTeleop
- Opening the folder with a VS-Code-based IDE allows to conveniently execute tasks in .vscode/tasks.json.

In case you want to set it up on a different machine:
- We don't use a local .venv as the robotic installation process installs robotic into the system python (with executable at /usr/bin/python3 and dist packages at /usr/local. But if you only want to work in simulation, you can also use a local .venv.
- [Install the robotic library](https://github.com/MarcToussaint/robotic/blob/master/README.md) from source to allow control of the real robot. Note: For the Panda on the mobile base libfranka needs to >=0.13.3.
- Install Python dependencies with either:
  - `pip install -r requirements.txt --break` (system Python setup), or
  - `uv pip install -r requirements.txt` (recommended if you use `uv` locally).

## Usage

Start teleoperation:

`uv run teleop_with_haply.py`

Show all available CLI options:

`uv run teleop_with_haply.py --help`

Useful options:
- `--teleop-in-real`: Control the real robot instead of simulation.
- `--avoid-self-collisions`: Enable collision-avoidance objective in IK.
- `--haply-to-robot-workspace-scale`: Tune workspace mapping gain.
- `--sync-hz`: Control loop frequency (default: 120 Hz).
- `--gripper-speed`: Gripper opening/closing speed.
- `--force-feedback`: Enable force feedback (alpha; real robot mode only).

Operator controls:
- Move Haply Inverse3 to move the Panda end-effector target.
- Press `a` on the Verse Grip to close the gripper.
- Press `b` on the Verse Grip to open the gripper.
- Stop safely with `Ctrl+C`.

## Hardware Setup

### Haply Calibration

Calibrate the Inverse 3 as described [here](https://docs.haply.co/docs/quick-start/#:~:text=Calibration%E2%80%8B,can%20now%20start%20a%20simulation).

General calibration information on the VerseGrip can be found [here](https://docs.haply.co/docs/quick-start-verse-grip).
But note that the implementation assumes that the VerseGrip stylus is calibrated with the thick end pointing upwards and the buttons facing the operator, orthogonally to the long edge of the desk.
The neutral xzy-position (which translates to the bot's home position) is initialized with the first received Haply position in the script.

### Franka Ranger

- Turn on the black Franka control box mounted on top of the Ranger.
- Visit 172.0.0.2/desk in the browser.
- Unlock the breaks in the the panel on the right.
- Activate FCI (Franka Control Interface): In the upper right click on the header entry showing the IP address and "PANDA 3". In the drop down menu click "Activate FCI".

In case you do this with a new machine you have to manually set your local IP address:
- Open Settings -> Network
- Open the settings of your wired connection to the Franka control box.
- Click on the IPv4 section.
- Select "Manual" as the IPv4 method.
- Enter 172.16.0.1 as the address and 255.255.255.0 as the netmask. Leave the gateway blank and DNS as well as Routes automatic.
- Don't forget to click on Apply.

## Noteworthy Implementation Details

- The Haply input stream is handled via an async WebSocket client in a dedicated thread. A shared `HaplyState` object guarded by a lock provides thread-safe access to position, orientation, buttons, and force commands.
- Haply pose is mapped to robot pose in `haply_to_robot_pose(...)`, including axis sign flips and quaternion component reordering to match robot frame conventions.
- Each control cycle solves IK as a one-phase, one-step KOMO problem (`ry.KOMO(C, 1, 1, 0, ...)`) with:
  - regularization toward current joint state,
  - regularization toward `qHome`,
  - an end-effector pose equality objective (`poseDiff` to the `target` frame),
  - optional self-collision objective.
- If IK is feasible, the resulting joint configuration is sent with `BotOp.moveTo(..., overwrite=True)` and gripper commands are issued from Haply button state.
- Force feedback is currently alpha: in real-robot mode, external wrench data is read and clipped before being sent back to Haply. On [this botop fork](https://github.com/christian-helms/botop/tree/external_wrench_sensing) the sensed external wrench from libranka is forwarded through botop. (get_tauExternal is not enough to get the contact wrench via a LSP involving the transpose of the contact Jacobian as the exact control torques applied by libfranka are not available in botop.) On the "hackathon notebook" this fork is compiled.
