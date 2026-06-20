"""
Side-by-side robot simulation: Before (original code) vs After (optimized code).

Left panel  — terrain built with src/before/terrain/generator.py
Right panel — terrain built with src/after/terrain/generator.py

Both robots run the same trot-gait CPG on equivalent rough terrain.
Output: results/robot_walk.mp4  (1280×360, 30 fps)
        results/robot_walk.gif  (960×270,  20 fps, 4 s loop)
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
OUT_MP4 = os.path.join(RESULTS, "robot_walk.mp4")
OUT_GIF = os.path.join(RESULTS, "robot_walk.gif")
FIXTURE  = os.path.join(ROOT, "benchmark", "fixtures", "minimal_scene.xml")
os.makedirs(RESULTS, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1.  Minimal quadruped XML  (Go2-inspired, no meshes)
# ─────────────────────────────────────────────────────────────

QUAD_XML = """
<mujoco model="go2_lite">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="50" solver="Newton"/>
  <default>
    <joint limited="true" damping="2.0" armature="0.01"/>
    <geom contype="1" conaffinity="1" condim="3" friction="0.8 0.02 0.001"/>
    <motor ctrllimited="true" ctrlrange="-40 40" gear="1"/>
  </default>
  <worldbody>
    <light name="top"  pos="0 0 5"   dir="0 0 -1"  diffuse="0.9 0.9 0.8" castshadow="true"/>
    <light name="fill" pos="3 -3 4"  dir="-1 1 -1" diffuse="0.4 0.5 0.6" castshadow="false"/>
    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.76 0.82 0.72 1"
          contype="1" conaffinity="1"/>

    <!-- TORSO: constrained to x-slide + z-slide + pitch (y-hinge).
         No roll / yaw DOF → lateral tipping is physically prevented. -->
    <body name="trunk" pos="0 0 0.361">
      <joint name="slide_x" type="slide" axis="1 0 0" limited="false" damping="0.5"/>
      <joint name="slide_z" type="slide" axis="0 0 1" limited="false" damping="2.0"/>
      <joint name="pitch"   type="hinge" axis="0 1 0" limited="true"
             range="-0.5 0.5" damping="3.0"/>
      <geom name="trunk_geom" type="box" size="0.1875 0.0495 0.057"
            rgba="0.2 0.3 0.8 1" mass="6.921"/>
      <inertial pos="0 0 0" mass="6.921"
                fullinertia="0.0158533 0.0377999 0.0456542 0 0 0"/>

      <!-- FL leg -->
      <body name="FL_hip" pos="0.1881 0.04675 0">
        <joint name="FL_hip_joint" type="hinge" axis="1 0 0" range="-1.047 1.047"/>
        <geom type="cylinder" size="0.046 0.030" euler="0 1.5708 0"
              rgba="0.15 0.15 0.15 1" mass="0.678"/>
        <body name="FL_thigh" pos="0 0.08 0">
          <joint name="FL_thigh_joint" type="hinge" axis="0 1 0" range="-1.5708 3.4907"/>
          <geom type="capsule" size="0.022" fromto="0 0 0  0 0 -0.213"
                rgba="0.2 0.3 0.8 1" mass="1.152"/>
          <body name="FL_calf" pos="0 0 -0.213">
            <joint name="FL_calf_joint" type="hinge" axis="0 1 0" range="-2.7227 -0.8378"/>
            <geom type="capsule" size="0.016" fromto="0 0 0  0 0 -0.213"
                  rgba="0.15 0.15 0.15 1" mass="0.154"/>
            <geom name="FL_foot" type="sphere" size="0.022" pos="0 0 -0.213"
                  rgba="0.1 0.1 0.1 1" mass="0.06"/>
          </body>
        </body>
      </body>

      <!-- FR leg -->
      <body name="FR_hip" pos="0.1881 -0.04675 0">
        <joint name="FR_hip_joint" type="hinge" axis="1 0 0" range="-1.047 1.047"/>
        <geom type="cylinder" size="0.046 0.030" euler="0 1.5708 0"
              rgba="0.15 0.15 0.15 1" mass="0.678"/>
        <body name="FR_thigh" pos="0 -0.08 0">
          <joint name="FR_thigh_joint" type="hinge" axis="0 1 0" range="-1.5708 3.4907"/>
          <geom type="capsule" size="0.022" fromto="0 0 0  0 0 -0.213"
                rgba="0.2 0.3 0.8 1" mass="1.152"/>
          <body name="FR_calf" pos="0 0 -0.213">
            <joint name="FR_calf_joint" type="hinge" axis="0 1 0" range="-2.7227 -0.8378"/>
            <geom type="capsule" size="0.016" fromto="0 0 0  0 0 -0.213"
                  rgba="0.15 0.15 0.15 1" mass="0.154"/>
            <geom name="FR_foot" type="sphere" size="0.022" pos="0 0 -0.213"
                  rgba="0.1 0.1 0.1 1" mass="0.06"/>
          </body>
        </body>
      </body>

      <!-- RL leg -->
      <body name="RL_hip" pos="-0.1881 0.04675 0">
        <joint name="RL_hip_joint" type="hinge" axis="1 0 0" range="-1.047 1.047"/>
        <geom type="cylinder" size="0.046 0.030" euler="0 1.5708 0"
              rgba="0.15 0.15 0.15 1" mass="0.678"/>
        <body name="RL_thigh" pos="0 0.08 0">
          <joint name="RL_thigh_joint" type="hinge" axis="0 1 0" range="-1.5708 3.4907"/>
          <geom type="capsule" size="0.022" fromto="0 0 0  0 0 -0.213"
                rgba="0.2 0.3 0.8 1" mass="1.152"/>
          <body name="RL_calf" pos="0 0 -0.213">
            <joint name="RL_calf_joint" type="hinge" axis="0 1 0" range="-2.7227 -0.8378"/>
            <geom type="capsule" size="0.016" fromto="0 0 0  0 0 -0.213"
                  rgba="0.15 0.15 0.15 1" mass="0.154"/>
            <geom name="RL_foot" type="sphere" size="0.022" pos="0 0 -0.213"
                  rgba="0.1 0.1 0.1 1" mass="0.06"/>
          </body>
        </body>
      </body>

      <!-- RR leg -->
      <body name="RR_hip" pos="-0.1881 -0.04675 0">
        <joint name="RR_hip_joint" type="hinge" axis="1 0 0" range="-1.047 1.047"/>
        <geom type="cylinder" size="0.046 0.030" euler="0 1.5708 0"
              rgba="0.15 0.15 0.15 1" mass="0.678"/>
        <body name="RR_thigh" pos="0 -0.08 0">
          <joint name="RR_thigh_joint" type="hinge" axis="0 1 0" range="-1.5708 3.4907"/>
          <geom type="capsule" size="0.022" fromto="0 0 0  0 0 -0.213"
                rgba="0.2 0.3 0.8 1" mass="1.152"/>
          <body name="RR_calf" pos="0 0 -0.213">
            <joint name="RR_calf_joint" type="hinge" axis="0 1 0" range="-2.7227 -0.8378"/>
            <geom type="capsule" size="0.016" fromto="0 0 0  0 0 -0.213"
                  rgba="0.15 0.15 0.15 1" mass="0.154"/>
            <geom name="RR_foot" type="sphere" size="0.022" pos="0 0 -0.213"
                  rgba="0.1 0.1 0.1 1" mass="0.06"/>
          </body>
        </body>
      </body>
    </body><!-- trunk -->
  </worldbody>

  <actuator>
    <motor name="FL_hip"   joint="FL_hip_joint"/>
    <motor name="FL_thigh" joint="FL_thigh_joint"/>
    <motor name="FL_calf"  joint="FL_calf_joint"/>
    <motor name="FR_hip"   joint="FR_hip_joint"/>
    <motor name="FR_thigh" joint="FR_thigh_joint"/>
    <motor name="FR_calf"  joint="FR_calf_joint"/>
    <motor name="RL_hip"   joint="RL_hip_joint"/>
    <motor name="RL_thigh" joint="RL_thigh_joint"/>
    <motor name="RL_calf"  joint="RL_calf_joint"/>
    <motor name="RR_hip"   joint="RR_hip_joint"/>
    <motor name="RR_thigh" joint="RR_thigh_joint"/>
    <motor name="RR_calf"  joint="RR_calf_joint"/>
  </actuator>
</mujoco>
"""

# ─────────────────────────────────────────────────────────────
# 2.  Scene builders (before / after)
# ─────────────────────────────────────────────────────────────

_TERRAIN_KWARGS = dict(
    init_pos=(0.5, -0.8, 0.0),
    nums=(8, 8),
    box_size=[0.18, 0.18, 0.06],
    box_euler=[0.0, 0.0, 0.0],
    separation=[0.22, 0.22],
    box_size_rand=[0.04, 0.04, 0.02],
    box_euler_rand=[0.08, 0.08, 0.08],
    separation_rand=[0.03, 0.03],
)


def _inject_boxes_into_xml(box_data_iter, is_before: bool) -> str:
    """Merge a list of box records into QUAD_XML and return a temp-file path."""
    root = ET.fromstring(QUAD_XML)
    wb   = root.find("worldbody")

    for bd in box_data_iter:
        if is_before:
            # before: list of dicts {"pos":…, "size":…, "quat":…}
            pos  = bd["pos"].tolist()
            size = bd["size"].tolist()   # already half-extent from before code
            quat = bd["quat"].tolist()
        else:
            # after: BoxData dataclass .pos .size .quat
            pos  = bd.pos.tolist()
            size = bd.size.tolist()
            quat = bd.quat.tolist()

        def v2s(v): return " ".join(f"{x:.6f}" for x in v)
        body = ET.SubElement(wb, "body")
        body.attrib["pos"]  = v2s(pos)
        body.attrib["quat"] = v2s(quat)
        geo = ET.SubElement(body, "geom")
        geo.attrib["type"] = "box"
        geo.attrib["size"] = v2s(size)
        geo.attrib["rgba"] = "0.60 0.50 0.38 1"
        geo.attrib["contype"]    = "1"
        geo.attrib["conaffinity"] = "1"

    vis = ET.SubElement(root, "visual")
    gb  = ET.SubElement(vis, "global")
    gb.attrib["offwidth"]  = "1280"
    gb.attrib["offheight"] = "720"

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    ET.ElementTree(root).write(tmp.name, encoding="unicode", xml_declaration=False)
    tmp.close()
    return tmp.name


def _euler_to_rot(roll, pitch, yaw):
    """ZYX Euler → 3×3 rotation matrix (mirrors before/terrain/generator.py)."""
    cx, sx = np.cos(roll),  np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx


def _euler_to_quat(roll, pitch, yaw):
    """ZYX Euler → quaternion [w,x,y,z] (mirrors before/terrain/generator.py)."""
    cx, sx = np.cos(roll/2),  np.sin(roll/2)
    cy, sy = np.cos(pitch/2), np.sin(pitch/2)
    cz, sz = np.cos(yaw/2),   np.sin(yaw/2)
    return np.array([
        cx*cy*cz + sx*sy*sz,
        sx*cy*cz - cx*sy*sz,
        cx*sy*cz + sx*cy*sz,
        cx*cy*sz - sx*sy*cz,
    ])


def _before_rough_ground(init_pos, nums, box_size, box_euler,
                         separation, box_size_rand, box_euler_rand,
                         separation_rand):
    """Exact copy of src/before/terrain/generator.py::AddRoughGround (nested loops)."""
    box_data = []
    local_pos = [0.0, 0.0, -0.5 * box_size[2]]
    new_sep = np.array(separation) + np.array(separation_rand) * np.random.uniform(-1, 1, 2)
    euler = [0.0, 0.0, 0.0]   # terrain euler (flat)
    for i in range(nums[0]):
        local_pos[0] += new_sep[0]
        local_pos[1] = 0.0
        for j in range(nums[1]):
            new_box_size  = np.array(box_size)  + np.array(box_size_rand)  * np.random.uniform(-1, 1, 3)
            new_box_euler = np.array(box_euler) + np.array(box_euler_rand) * np.random.uniform(-1, 1, 3)
            new_sep       = np.array(separation) + np.array(separation_rand) * np.random.uniform(-1, 1, 2)
            local_pos[1] += new_sep[1]
            pos  = _euler_to_rot(euler[0], euler[1], euler[2]) @ np.array(local_pos) + np.array(init_pos)
            quat = _euler_to_quat(new_box_euler[0], new_box_euler[1], new_box_euler[2])
            box_data.append({"pos": pos, "size": 0.5 * new_box_size, "quat": quat})
    return box_data


def build_before_scene(seed: int = 7) -> tuple[str, float]:
    """Build scene with before (original) AddRoughGround. Returns (xml_path, gen_ms)."""
    np.random.seed(seed)
    t0 = time.perf_counter()
    box_data = _before_rough_ground(**_TERRAIN_KWARGS)
    gen_ms = (time.perf_counter() - t0) * 1000
    return _inject_boxes_into_xml(box_data, is_before=True), gen_ms


def build_after_scene(seed: int = 7) -> tuple[str, float]:
    """Build scene with after (optimized) AddRoughGround. Returns (xml_path, gen_ms)."""
    from src.after.terrain.generator import TerrainConfig, TerrainGenerator

    np.random.seed(seed)
    cfg = TerrainConfig.from_args(width=0.3, step_height=0.08, num_stairs=5)
    tg  = TerrainGenerator(cfg)

    t0 = time.perf_counter()
    tg.AddRoughGround(**_TERRAIN_KWARGS)
    gen_ms = (time.perf_counter() - t0) * 1000

    return _inject_boxes_into_xml(tg.box_data, is_before=False), gen_ms


# ─────────────────────────────────────────────────────────────
# 3.  Gait CPG + PD controller
# ─────────────────────────────────────────────────────────────

NOMINAL_HIP   = 0.0
NOMINAL_THIGH = 0.67
NOMINAL_CALF  = -1.30
THIGH_SWING   = 0.15
CALF_LIFT     = 0.20


def standing_ctrl(model, data, Kp=120.0, Kd=5.0):
    targets = np.array([NOMINAL_HIP, NOMINAL_THIGH, NOMINAL_CALF] * 4)
    # qpos layout: [slide_x, slide_z, pitch, leg×12]
    q   = data.qpos[3:]
    dq  = data.qvel[3:]
    tau = Kp * (targets - q) - Kd * dq
    data.ctrl[:] = np.clip(tau, -40, 40)


def trot_ctrl(model, data, t, freq=1.2, Kp=80.0, Kd=3.0):
    phi = 2.0 * np.pi * freq * t

    def leg(offset):
        s  = np.sin(phi + offset)
        th = float(np.clip(NOMINAL_THIGH - THIGH_SWING * s, -1.5708, 3.4907))
        cf = float(np.clip(NOMINAL_CALF  - CALF_LIFT   * s, -2.7227, -0.8378))
        return NOMINAL_HIP, th, cf

    fl, fr, rl, rr = leg(0.0), leg(np.pi), leg(np.pi), leg(0.0)
    targets = np.array([*fl, *fr, *rl, *rr])
    q   = data.qpos[3:]
    dq  = data.qvel[3:]
    tau = Kp * (targets - q) - Kd * dq
    data.ctrl[:] = np.clip(tau, -40, 40)


# ─────────────────────────────────────────────────────────────
# 4.  Single-panel simulation
# ─────────────────────────────────────────────────────────────

_PW, _PH = 640, 360          # per-panel size


def _render_frame(renderer, model, data):
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    trunk_pos     = data.xpos[model.body("trunk").id]
    cam.lookat[:] = trunk_pos + np.array([0.2, 0.0, 0.05])
    cam.distance  = 2.0
    cam.azimuth   = 210.0
    cam.elevation = -18.0
    renderer.update_scene(data, camera=cam)
    px = renderer.render()
    return (px[:, :, :3] if px.shape[2] == 4 else px).astype(np.uint8)


def run_simulation(scene_xml: str,
                   total_seconds: float = 8.0,
                   fps: int = 30,
                   warmup_seconds: float = 1.0) -> list[np.ndarray]:
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data  = mujoco.MjData(model)

    # Initial pose: trunk joints = 0, legs = nominal
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.0
    for leg_start in (3, 6, 9, 12):
        data.qpos[leg_start]     = NOMINAL_HIP
        data.qpos[leg_start + 1] = NOMINAL_THIGH
        data.qpos[leg_start + 2] = NOMINAL_CALF
    mujoco.mj_forward(model, data)

    dt              = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    total_steps     = int(total_seconds / dt)
    warmup_steps    = int(warmup_seconds / dt)

    frames: list[np.ndarray] = []
    with mujoco.Renderer(model, height=_PH, width=_PW) as renderer:
        for step in range(total_steps):
            if step < warmup_steps:
                standing_ctrl(model, data)
            else:
                trot_ctrl(model, data, t=(step - warmup_steps) * dt)
            mujoco.mj_step(model, data)

            if step >= warmup_steps and (step - warmup_steps) % steps_per_frame == 0:
                frames.append(_render_frame(renderer, model, data))

    return frames


# ─────────────────────────────────────────────────────────────
# 5.  Side-by-side stitching with text overlay
# ─────────────────────────────────────────────────────────────

def _put_label(img: np.ndarray, text: str, color_bgr: tuple) -> np.ndarray:
    """Overlay a semi-transparent banner + text at the top of img (in-place copy)."""
    out = img.copy()
    h, w = out.shape[:2]
    # dark translucent banner
    banner = out[:44, :].astype(np.float32)
    banner *= 0.35
    out[:44, :] = banner.astype(np.uint8)
    cv2.putText(out, text,
                org=(12, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.85,
                color=color_bgr,
                thickness=2,
                lineType=cv2.LINE_AA)
    return out


def stitch(left_frames: list[np.ndarray],
           right_frames: list[np.ndarray],
           left_label: str,
           right_label: str) -> list[np.ndarray]:
    n = min(len(left_frames), len(right_frames))
    out = []
    for i in range(n):
        l = _put_label(left_frames[i],  left_label,  (80, 80, 255))   # red-ish (BGR)
        r = _put_label(right_frames[i], right_label, (80, 220, 80))   # green (BGR)
        # vertical divider line
        divider = np.full((_PH, 4, 3), 255, dtype=np.uint8)
        out.append(np.hstack([l, divider, r]))
    return out


# ─────────────────────────────────────────────────────────────
# 6.  Save MP4 / GIF
# ─────────────────────────────────────────────────────────────

def save_mp4(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  saved {path}  ({len(frames)} frames, {len(frames)/fps:.1f} s)")


def save_gif(frames: list[np.ndarray], path: str,
             fps: int = 20, max_seconds: float = 4.0, scale: float = 0.75) -> None:
    try:
        import imageio
    except ImportError:
        print("  imageio not installed — skipping GIF")
        return
    keep   = min(len(frames), int(max_seconds * fps))
    subset = [cv2.resize(f, (0, 0), fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
              for f in frames[:keep]]
    imageio.mimsave(path, subset, duration=1.0 / fps, loop=0)
    print(f"  saved {path}  ({keep} frames, {keep/fps:.1f} s, GIF)")


# ─────────────────────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEED = 7

    print("Building BEFORE scene (original AddRoughGround)…")
    before_xml, before_ms = build_before_scene(seed=SEED)
    print(f"  terrain gen time: {before_ms:.2f} ms")

    print("Building AFTER scene (optimized AddRoughGround)…")
    after_xml, after_ms = build_after_scene(seed=SEED)
    print(f"  terrain gen time: {after_ms:.2f} ms")
    speedup = before_ms / after_ms if after_ms > 0 else float("inf")
    print(f"  speedup: {speedup:.1f}×")

    print("Simulating BEFORE…")
    left_frames = run_simulation(before_xml, total_seconds=8.0, fps=30, warmup_seconds=1.0)
    os.unlink(before_xml)

    print("Simulating AFTER…")
    right_frames = run_simulation(after_xml, total_seconds=8.0, fps=30, warmup_seconds=1.0)
    os.unlink(after_xml)

    print("Stitching side-by-side…")
    left_label  = f"Before (original)  {before_ms:.1f} ms"
    right_label = f"After (optimized)  {after_ms:.1f} ms  [{speedup:.1f}x faster]"
    combined = stitch(left_frames, right_frames, left_label, right_label)
    print(f"  {len(combined)} combined frames")

    save_mp4(combined, OUT_MP4, fps=30)
    save_gif(combined, OUT_GIF, fps=20, max_seconds=4.0, scale=0.75)
    print("Done.")
