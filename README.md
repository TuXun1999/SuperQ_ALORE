Some useful commands:



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

To train Grasp-TAGS, run from the repository root. In
[pre_grasping.yaml](source/SuperQ_ALORE/SuperQ_ALORE/assets/objects/pre_grasping.yaml),
uncomment one complete object block, including its candidate grasp poses, and
comment out the other objects. The catalog retains multiple object definitions
for extensibility; training uses one object at a time.

```bash
python scripts/rsl_rl/train.py \
  --task Template-Superq-Alore-v0 \
  --agent rsl_rl_grasp_ranking_cfg_entry_point \
  --num_envs 4096 --headless
```

This trains the shared PPO policy across the selected object's grasps, then
distills the critic into the grasp utility network. Omitting `--agent` runs
standard PPO without grasp-utility distillation.


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

