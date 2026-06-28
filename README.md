# rl-policy / go2

Deployable low-level locomotion policy for the **Unitree Go2**, trained with
**DreamWaQ** (CVAE implicit terrain estimator, arXiv:2301.10602) on Isaac Lab.

The policy is **proprioception-only** — no camera, LiDAR or height scan at
inference time. It tracks an external `(vx, vy, vyaw)` velocity command and
handles small-to-moderate stairs (≈ 15 cm steps, ~level 4.8 of the Isaac Lab
pyramid-stairs curriculum) using a CENet that estimates terrain context from a
5-step proprioceptive history.

## Contents

```
models/
├── dwaq_go2_v12.pt        TorchScript policy (1.2 MB, CPU & CUDA)
└── dwaq_go2_v12.json      I/O specification (term order, scales, joint order)
examples/
└── infer_example.py       minimal standalone usage
requirements.txt           torch only
```

## Install

```bash
pip install -r requirements.txt    # torch (CPU is fine; CUDA optional)
```

That is the only dependency — the `.pt` is plain TorchScript, no Isaac
Sim / Isaac Lab / rsl_rl needed to run it.

## Quick use

```python
import torch

policy = torch.jit.load("models/dwaq_go2_v12.pt", map_location="cpu").eval()

# obs_current : (B, 45)   — see models/dwaq_go2_v12.json for the term order
# obs_history : (B, 225)  — last 5 obs_current vectors flattened (oldest first)
action = policy(obs_current, obs_history)             # (B, 12)

joint_pd_target = default_joint_pos + 0.25 * action   # what to send to the PD layer
```

A complete runnable demo with dummy inputs is in [`examples/infer_example.py`](examples/infer_example.py).

## Policy I/O

### Input — `obs_current` (45 dim)

| # | term | dim | scale | source |
|---|---|---|---|---|
| 0 | `base_ang_vel`      | 3  | ×0.2  | IMU gyro (body frame) |
| 1 | `projected_gravity` | 3  | ×1.0  | gravity in body frame |
| 2 | `velocity_commands` | 3  | ×1.0  | **external** `(vx, vy, vyaw)` |
| 3 | `joint_pos_rel`     | 12 | ×1.0  | `joint_pos − default_joint_pos` |
| 4 | `joint_vel_rel`     | 12 | ×0.05 | joint angular velocity |
| 5 | `last_action`       | 12 | ×1.0  | previous policy output |

### Input — `obs_history` (225 dim)

The last **5** `obs_current` vectors (oldest → newest) concatenated, so
`obs_history.shape == (5 * 45,)`. Build it on-robot with a ring buffer.

### Output — `action` (12 dim)

12 joint position deltas. Convert to a PD target as:

```
joint_pd_target = default_joint_pos + 0.25 * action
```

Joint order matches the **Unitree Go2 URDF** used at training time
(`unitree_ros2/robots/go2_description/urdf/go2_description.urdf`). Re-map to
your on-robot SDK indexing as needed.

## Control loop

Run the policy at **50 Hz**. Each step:

1. Read IMU + joint encoders, assemble `obs_current` (same term order & scales).
2. Push it into the 5-slot history ring buffer → `obs_history`.
3. `action = policy(obs_current.unsqueeze(0), obs_history.unsqueeze(0))[0]`
4. `joint_pd_target = default_joint_pos + 0.25 * action`
5. Send PD targets to the motor controller.
6. Remember `action` as `last_action` for the next obs assembly.

## Limits

- Proprioception-only — handles ≈ 15 cm stairs reliably. Beyond that (≥ 20 cm)
  requires perceptive input (height scan / depth camera) and is **not** what
  this policy was trained for.
- Trained in Isaac Lab simulation only — sim-to-real gap on a physical Go2 is
  not yet measured.
- Commanded velocity range clipped during training to `vx, vy, vyaw ∈ [-1, 1]`
  m/s and rad/s. Driving harder is undefined behavior.

## License

Apache-2.0 — see [LICENSE](LICENSE).
