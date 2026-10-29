# QiVi chair physics

The training task loads `qivi.usdc`. `qivi.usda` is the matching editable ASCII
asset; keep both files synchronized when changing the physics. The articulation
has 20 bodies and 19 connections: 18 movable joints and one fixed height joint.
The repository ignores `*.usdc` files, so a fresh checkout also needs the
runtime binary exported from `qivi.usda`. Updating the ASCII source alone does
not update an existing binary.

## Documented specifications

[Steelcase's QiVi cut sheet](https://www.steelcase.com/resources/documents/qivi-cut-sheet/)
lists a five-star chair weight of **25.7 lb**, or 11.657 kg. The asset uses
**11.66 kg**, rounded to two decimal places, as its nominal empty-chair mass.
Specific finishes, base options and casters can change the real weight.

The previous asset totaled 7.4389 kg. Each body mass and its diagonal inertia
were multiplied by the same factor (approximately 1.5674), retaining the
estimated relative mass distribution, centers of mass and principal axes.
The individual link masses and inertias have not been measured.

[Steelcase's seating specification guide](https://www.steelcase.com/content/uploads/2020/10/seating.pdf)
describes a 1-inch seat glide, a 15-degree pivoting back, and hard dual-wheel
casters for carpet. The existing 25.4 mm glide range and 15-degree backrest
limit agree with those travel specifications.

## Seat lowered and locked before grasping

The real setup manually lowers the gas lift fully and locks its height before
the robot grasps the chair. The USD represents this with a **fixed joint** named
`height_hold`, replacing the previous spring-held prismatic joint. Its attachment
uses the original minimum travel of **-0.0385 m** relative to the supplied neutral
pose. The lift, frame, seat and back are authored 3.85 cm lower, so the asset
opens at the intended height and cannot extend or compress during a rollout.
This offset comes from the existing model's travel limit; the absolute seat
height has not been measured against the physical chair.

The seat's fore/aft glide, backrest pivot, chair swivel and caster joints remain
passive and movable. The initial frame offset used by the training task is
updated to `(0, 0, 0.3512)` m to match the lowered asset. Robot placements and
arm configurations remain supplied by `pre_grasping.yaml`; check those against
the grasp points at this height. The legacy `qivi_cfg.py` height-setting helper
is not used by the training task and cannot adjust this fixed-height asset.

## Nominal carpet model: estimates requiring calibration

The intended floor is carpet. Steelcase does not publish the wheel/carpet
friction, rolling resistance, or spring/damper constants used below. These are
model assumptions, not measured product specifications.

| Property | Nominal value | Meaning |
| --- | --- | --- |
| Wheel static / dynamic contact friction | 0.7 / 0.7 | Existing estimate of traction against sliding; retained. |
| Structure static / dynamic contact friction | 0.5 / 0.5 | Existing estimate for other chair contacts; retained. |
| Wheel rolling damping | 0.002 N m s/rad per wheel | Initial carpet rolling-resistance approximation; previously 0.0006 in the USD and effectively zero during training. |
| Caster swivel damping | 0.008 N m s/rad | Existing estimate; retained. |
| Chair swivel damping | 0.04 N m s/rad | Existing estimate; retained. |
| Seat glide spring / damper | 650 N/m / 28 N s/m | Existing estimate; retained. |
| Backrest spring / damper | 18 N m/rad / 0.8 N m s/rad | Existing estimate; retained. |
| Seat height adjustment | Fixed at original minimum (-0.0385 m) | Mechanical lock; no spring or adjustable height DOF. |

The backrest's estimated spring preload is scaled with the mass correction to
approximately preserve its unloaded equilibrium. It is not an extra command
from the robot policy.

Traction friction and rolling resistance are different parameters. Simply
increasing contact friction does not reproduce a carpet's resistance to wheel
rolling. The nominal model uses wheel-axis damping to approximate that drag,
while leaving the wheels free to roll and the casters free to swivel. The
ground remains a rigid plane; carpet deformation and breakaway resistance are
not explicitly simulated.

For ten wheels of radius 0.0325 m rolling straight without slip, this damping
corresponds to approximately **3.8 N of resistance at 0.2 m/s**:

`F = number_of_wheels * damping * speed / radius**2`

This is a prediction of the assumed model, not a measured force for the real
chair. It varies with speed. To calibrate it, measure the force required to
roll the empty chair steadily on the actual carpet at a known speed, with the
casters aligned. Estimate per-wheel damping as
`force * radius**2 / (number_of_wheels * speed)`. Check several speeds and
directions; caster realignment and initial breakaway need separate checks.

## Isaac Lab integration

The chair's implicit actuator uses `stiffness=None, damping=None`, preserving
the gains authored in the USD. Setting these to zero overrides the springs
and rolling damping and previously made the unloaded backrest fall to its
joint limit. The height is now mechanically fixed. Episode resets also restore
authored spring rest targets, converting angular targets from USD degrees to
tensor radians.

USD angular spring and damping values are per degree; the SI values in the
table are per radian. For example, wheel damping 0.002 N m s/rad is authored
as approximately 0.000034906585 N m s/degree. Mass randomization remains off.

Changing these nominal dynamics changes the training task. Policies trained
with the earlier mass and disabled passive mechanisms should be evaluated
again and may require retraining.
