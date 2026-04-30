# Set of the Project
Everything in README(old).md is deployed, except the Omniverse setup. (The IDE setup may not
be 100% successful though, because it seems that isaacsim can be detected by VScode, but isaaclab cannot)

# Deployment of low-level controller
In your terminal, run 
```
python3 ./scripts/zero_agent.py --task Template-Superq-Alore-v0 --num_envs 20
```

Then, you should observe the robot moving forward with a constant speed. 

You can set up the attributes in "./scripts/zero_agent.py". 

# Run the joint-teleoperation

Use the keyboard teleoperation script to command Spot arm joints (and optionally base motion).

```bash
# If Isaac Lab is available in your current Python environment:
python scripts/joint_teleoperation.py --task Template-Superq-Alore-v0 --num_envs 1

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

