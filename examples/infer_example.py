#!/usr/bin/env python3
"""Minimal standalone usage of the deployed DreamWaQ Go2 policy.

Loads ``models/dwaq_go2_v12.pt`` and runs one forward pass with dummy zero
observations. Use this as a template for plugging the policy into your robot
SDK / ROS2 node — only the ``build_obs_current(...)`` and ``send_pd_target(...)``
helpers need real implementations.
"""

from __future__ import annotations

import collections
import os
from collections import deque

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "models", "dwaq_go2_v12.pt")

# observation layout (must match the order/scales used at training time)
NUM_PROPRIO = 45
HISTORY_LEN = 5
NUM_ACTIONS = 12
ACTION_SCALE = 0.25
CONTROL_HZ = 50

# 12-element default joint pose used at training (radians, Go2 URDF joint order).
# Adjust if your robot has a different default standing pose.
DEFAULT_JOINT_POS = torch.tensor(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    dtype=torch.float32,
)


def build_obs_current(
    base_ang_vel: torch.Tensor,        # (3,) rad/s
    projected_gravity: torch.Tensor,   # (3,)
    velocity_command: torch.Tensor,    # (3,) (vx, vy, vyaw) from your nav / joystick
    joint_pos: torch.Tensor,           # (12,) rad
    joint_vel: torch.Tensor,           # (12,) rad/s
    last_action: torch.Tensor,         # (12,)
) -> torch.Tensor:
    """Assemble the 45-dim obs_current tensor with the trained scales."""
    return torch.cat(
        [
            base_ang_vel * 0.2,
            projected_gravity,
            velocity_command,
            joint_pos - DEFAULT_JOINT_POS,
            joint_vel * 0.05,
            last_action,
        ]
    )


def main():
    policy = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
    print(f"loaded policy: {MODEL_PATH}")

    # rolling history buffer (oldest -> newest)
    history: deque = collections.deque(maxlen=HISTORY_LEN)
    for _ in range(HISTORY_LEN):
        history.append(torch.zeros(NUM_PROPRIO))

    last_action = torch.zeros(NUM_ACTIONS)

    # ---- example control step (replace dummies with real sensor reads) ----
    obs_current = build_obs_current(
        base_ang_vel=torch.zeros(3),
        projected_gravity=torch.tensor([0.0, 0.0, -1.0]),
        velocity_command=torch.tensor([0.5, 0.0, 0.0]),   # forward 0.5 m/s
        joint_pos=DEFAULT_JOINT_POS.clone(),
        joint_vel=torch.zeros(12),
        last_action=last_action,
    )
    history.append(obs_current)
    obs_history = torch.cat(list(history))   # (5 * 45,) = (225,)

    with torch.no_grad():
        action = policy(obs_current.unsqueeze(0), obs_history.unsqueeze(0))[0]

    joint_pd_target = DEFAULT_JOINT_POS + ACTION_SCALE * action

    print(f"obs_current shape : {tuple(obs_current.shape)}")
    print(f"obs_history shape : {tuple(obs_history.shape)}")
    print(f"action            : {action.tolist()}")
    print(f"joint_pd_target   : {joint_pd_target.tolist()}")


if __name__ == "__main__":
    main()
