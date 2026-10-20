Some useful commands:

## ReLIC WASD: simulation and optional real Spot

Simulation only (the default; no robot connection):

```bash
python3 scripts/wasd_relic_spot.py
```

To command **both simulation and physical Spot**, configure the SDK credentials
in `BOSDYN_CLIENT_USERNAME` and `BOSDYN_CLIENT_PASSWORD`, then run:

```bash
python3 scripts/wasd_relic_spot.py --real-robot --hostname 192.168.80.3
```

**This flag acquires a lease, powers on Spot, commands standing/open gripper, and
enables ReLIC joint control.** Use the established deployment setup, clear space,
and an independent Spot E-stop. Wait for the panel's “Real Spot ready” message
before selecting a movement key. The worker cannot prompt for credentials.

- Tap W/S for ±x, A/D for ±y, Q/E for ±yaw. Commands remain active after release
  and focus changes. Defaults are 0.20 m/s and 0.40 rad/s; use `--linear_speed`
  and `--turn_speed` to change them.
- Space/Stop commands zero base velocity in both environments; it is **not an
  emergency stop**. The real robot continues running ReLIC to stand.
- Reset stops both commands and resets **only simulation**, keeping the panel
  open. It does not reposition the physical robot or re-latch its arm reference.
- Esc, window close, or Ctrl-C requests standing followed by stream shutdown,
  gripper release, arm stow, SDK self-right, power-off, and lease return.
  Recovery is skipped if already powered off or E-stopped; a failed stow prevents
  self-right, but still attempts power-off. Keep the recovery area clear.
- Real inference runs in its own wall-clock 100 Hz process using
  `SPOT`/`SpotReLICEnvPLAY`, the same classes as `zero_agent_ReLIC_spot.py`.
  It uses real observations and holds the measured startup arm pose. Simulated
  observations, arm poses, and predicted joint targets are never sent to Spot.
- If the panel stops refreshing commands for 0.5 s (including a stalled/paused
  simulation), the worker commands standing. The old motion cannot automatically
  resume with a heartbeat; select a new command after the panel recovers. This
  watchdog does not replace Spot's external E-stop or guarantee balance.

Optional real input/output logging (overwrites the selected file):

```bash
python3 scripts/wasd_relic_spot.py --real-robot --hostname 192.168.80.3 \
  --record-policy-trace --policy-trace-log relic_wasd_trace.txt
```

Hardware-free bridge tests:

```bash
python3 -m unittest discover -s scripts -p 'test_relic_spot_bridge.py' -v
```

## Chair pulling: MPC or ReLIC

`zero_agent_spot.py` keeps the existing AprilTag approach, arm positioning, and
gripper-closing sequence. Without a flag it uses the existing Spot MPC pull loop:

```bash
python3 scripts/zero_agent_spot.py --hostname 192.168.80.3
```

To switch to ReLIC **after grasping**, without simulation, a bridge, or logging:

```bash
python3 scripts/zero_agent_spot.py --hostname 192.168.80.3 --use-relic --steps 500
```

This commands backward velocity `-0.2 m/s` for approximately five seconds of
100 Hz ReLIC updates. The default is 100 updates (about one second in ReLIC mode).
The existing gripper-closing wait is retained; it is not a grasp-success check.
ReLIC holds the measured arm/gripper positions with joint PD control, rather than
preserving the SDK's gripper-force controller. Check grasp retention and use an
external E-stop. ReLIC mode stops joint streaming, releases the gripper, waits
for arm stow, runs SDK self-right, and requests power-off on completion, Ctrl-C,
or error. Recovery is skipped when powered off or E-stopped; power-off is still
attempted if recovery fails. MPC mode retains its existing powered-on end state.

To start a training:
```bash
python scripts/rsl_rl/train.py --task Template-Superq-Alore-v0 --num_envs 4096 --headless
```

To start a joint teleoperation:
```bash
python scripts/windows_joint_teleoperation.py --task Joint-Teleoperation --num_envs 1
```

To start a `zero_agent`:
```bash
python3 ./scripts/zero_agent.py --task Template-Superq-Alore-v0 --num_envs 20

```






# Set of the Project
Everything in README(old).md is deployed, except the Omniverse setup. (The IDE setup may not
be 100% successful though, because it seems that isaacsim can be detected by VScode, but isaaclab cannot)

# Deployment of low-level controller
In your terminal, run 
```bash
python3 ./scripts/zero_agent.py --task Template-Superq-Alore-v0 --num_envs 20
```

Then, you should observe the robot moving forward with a constant speed. 

You can set up the attributes in "./scripts/zero_agent.py". 

# Run the joint-teleoperation

Use the keyboard teleoperation script to command Spot arm joints (and optionally base motion).

```bash
# If Isaac Lab is available in your current Python environment:
python scripts/windows_joint_teleoperation.py --task Joint-Teleoperation --num_envs 1

# If Isaac Lab is not in your active environment, use Isaac Lab's launcher instead:
# <FULL_PATH_TO_isaaclab.bat> -p scripts/joint_teleoperation.py --task Template-Superq-Alore-v0 --num_envs 1
```

## Keyboard controls

- `1..7`: select arm joint index
- `a` / `d`: increase/decrease selected joint target (arm mode)
- `1a`, `1d`, ... `7a`, `7d`: compact joint select + move command
- `b`: toggle base motion mode on/off
- `w` / `a` / `s` / `d` (in motion mode): move base (`+vx`, `+vy`, `-vx`, `-vy`)
- `r`: reset arm targets to initial values
- `p`: print current arm joint targets
- `q`: quit

## Useful options

- `--step_size 0.05`: joint increment in radians (default)
- `--step_size_deg 2.0`: joint increment in degrees (overrides `--step_size`)
- `--motion_speed 0.2`: base speed used by WASD in motion mode
- `--base_lin_vel VX VY WZ`: fixed base velocity command
- `--base_pose PITCH HEIGHT`: fixed base pose command
- `--init_from_zero`: initialize all 7 arm joints from zero

Example with custom step and base speed:

```bash
python scripts/joint_teleoperation.py \
	--task Template-Superq-Alore-v0 \
	--num_envs 1 \
	--step_size_deg 3.0 \
	--motion_speed 0.3
```

Notes:
- This script is intended for keyboard input from a Windows terminal.
- The action vector keeps non-arm components fixed while updating Spot arm targets.
