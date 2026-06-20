"""
Optimized terrain generator.

Changes from before/terrain/generator.py:
  A. Data structures / vectorization
     - AddRoughGround: all per-cell random draws were inside a nested Python
       loop (3 × np.random.uniform calls per cell).  Replaced with 3 batched
       calls that generate all values at once, then numpy cumsum for positions.
       Positions that depend on random separations are computed with cumsum
       instead of sequential accumulation.
     - box_data list now holds BoxData dataclass instances instead of plain
       dicts. Attribute access is ~2x faster than dict key lookup for hot paths.

  C. Class design / SRP
     - TerrainConfig (frozen dataclass): all numeric parameters extracted from
       TerrainGenerator.__init__ into an immutable config object.
     - BoxData (dataclass): typed container for a single terrain box.
     - RoughGroundBuilder: encapsulates the vectorized rough-ground box
       generation logic.  TerrainGenerator.AddRoughGround delegates to it;
       TerrainGenerator only handles coordinate transforms and accumulation.
     - TerrainGenerator no longer loads an XML file in __init__ unless render=True
       and a scene_path is provided.  The config-only path (render=False) works
       without any XML file on disk.

  D. Decorators
     - @timing on AddRoughGround and AddStairs to measure invocation cost.
     - @validate_grid on generate_14 / generate_discrete.
     - @log_call on TerrainGenerator.Save to surface IO errors clearly.
"""
from __future__ import annotations

import random
import string
import os
import sys
import xml.etree.ElementTree as xml_et
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..utils.decorators import log_call, timing, validate_grid


# ---------------------------------------------------------------------------
# C: Configuration and data dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerrainConfig:
    """Immutable terrain geometry parameters.

    Frozen so instances can be shared or reused without mutation risk.
    Exactly one of `num_stairs` or `length` must be None (computed from others).
    """
    width: float
    step_height: float
    num_stairs: int
    length: float
    block_height: float
    render: bool = False

    @staticmethod
    def from_args(
        width: Optional[float] = None,
        step_height: Optional[float] = None,
        num_stairs: Optional[int] = None,
        length: Optional[float] = None,
        render: bool = False,
    ) -> "TerrainConfig":
        args = [width, step_height, num_stairs, length]
        if args.count(None) != 1:
            raise ValueError("Exactly three of width/step_height/num_stairs/length must be provided.")
        if num_stairs is None:
            num_stairs = int(length / width)
        elif length is None:
            length = num_stairs * width
        block_height = num_stairs * step_height
        return TerrainConfig(
            width=width, step_height=step_height,
            num_stairs=num_stairs, length=length,
            block_height=block_height, render=render,
        )


@dataclass
class BoxData:
    """Typed container for a single terrain box.

    Attribute access is faster than dict key lookup, and fields are
    explicit — no silent typos in key names.
    """
    pos: np.ndarray
    size: np.ndarray
    quat: np.ndarray


# ---------------------------------------------------------------------------
# Math utilities (unchanged from before)
# ---------------------------------------------------------------------------

def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(roll / 2), np.sin(roll / 2)
    cy, sy = np.cos(pitch / 2), np.sin(pitch / 2)
    cz, sz = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], dtype=np.float64)


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    c, s = np.cos, np.sin
    Rx = np.array([[1, 0, 0], [0, c(roll), -s(roll)], [0, s(roll), c(roll)]])
    Ry = np.array([[c(pitch), 0, s(pitch)], [0, 1, 0], [-s(pitch), 0, c(pitch)]])
    Rz = np.array([[c(yaw), -s(yaw), 0], [s(yaw), c(yaw), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float64)


def rot2d(x: float, y: float, yaw: float) -> Tuple[float, float]:
    return x * np.cos(yaw) - y * np.sin(yaw), x * np.sin(yaw) + y * np.cos(yaw)


def rot3d(pos, euler) -> np.ndarray:
    return euler_to_rot(euler[0], euler[1], euler[2]) @ np.asarray(pos)


def list_to_str(vec) -> str:
    return " ".join(str(s) for s in vec)


def random_box_name() -> str:
    return "box_" + "".join(random.choices(string.ascii_letters + string.digits, k=5))


# ---------------------------------------------------------------------------
# C: RoughGroundBuilder — single responsibility: vectorized box generation
# A: all random draws batched; positions via cumsum instead of sequential loop
# ---------------------------------------------------------------------------

class RoughGroundBuilder:
    """Generates rough-ground box data using fully vectorized random sampling.

    Separates the generation logic from TerrainGenerator so each class has
    one clear responsibility.  The builder produces boxes in local space;
    the caller applies world-space transform.
    """

    def __init__(
        self,
        nums: List[int],
        box_size: List[float],
        box_euler: List[float],
        separation: List[float],
        box_size_rand: List[float],
        box_euler_rand: List[float],
        separation_rand: List[float],
    ) -> None:
        self.nums = nums
        self.box_size = np.asarray(box_size, dtype=float)
        self.box_euler = np.asarray(box_euler, dtype=float)
        self.separation = np.asarray(separation, dtype=float)
        self.box_size_rand = np.asarray(box_size_rand, dtype=float)
        self.box_euler_rand = np.asarray(box_euler_rand, dtype=float)
        self.separation_rand = np.asarray(separation_rand, dtype=float)

    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions, eulers, sizes) arrays — all boxes in local space.

        A: all random values generated in three batched calls instead of
        O(N) individual np.random.uniform calls inside the nested loop.
        Positions are derived via np.cumsum on random separations, eliminating
        sequential state accumulation from the inner loop.
        """
        n0, n1 = self.nums
        n_total = n0 * n1

        # A: batch all random generation — 3 calls instead of 3*N
        rng_sizes  = self.box_size  + self.box_size_rand  * np.random.uniform(-1, 1, (n_total, 3))
        rng_eulers = self.box_euler + self.box_euler_rand * np.random.uniform(-1, 1, (n_total, 3))
        # (n_total + n0) separations: n0 for x (one per row), n_total for y (one per cell)
        rng_seps   = self.separation + self.separation_rand * np.random.uniform(-1, 1, (n0 + n_total, 2))

        # x positions: one separation per row, cumsum gives row x-coordinates
        x_seps = rng_seps[:n0, 0]                          # shape (n0,)
        x_pos  = np.cumsum(x_seps)                         # shape (n0,)

        # y positions: reset per row, cumsum within each row
        y_seps = rng_seps[n0:, 1].reshape(n0, n1)          # shape (n0, n1)
        y_pos  = np.cumsum(y_seps, axis=1)                  # shape (n0, n1)

        base_z = -0.5 * self.box_size[2]
        X = np.repeat(x_pos, n1)                           # (n0*n1,)
        Y = y_pos.ravel()                                   # (n0*n1,)
        Z = np.full(n_total, base_z)

        local_positions = np.stack([X, Y, Z], axis=1)      # (n_total, 3)
        return local_positions, rng_eulers, rng_sizes


# ---------------------------------------------------------------------------
# C: TerrainGenerator — assembly only; delegates generation to builders
# ---------------------------------------------------------------------------

class TerrainGenerator:
    """Assembles terrain boxes from geometry builders.

    Single responsibility: apply world-space transforms and accumulate BoxData.
    Does not generate random values directly (delegated to builders).
    Does not load XML unless render=True and scene_path is provided.
    """

    def __init__(self, config: TerrainConfig, scene_path: Optional[str] = None) -> None:
        self.config = config
        self.count_boxes = 0
        self.box_data: List[BoxData] = []

        # Legacy attributes for compatibility with addElement / generate_14
        self.width = config.width
        self.step_height = config.step_height
        self.num_stairs = config.num_stairs
        self.length = config.length
        self.block_height = config.block_height
        self.render = config.render
        self.max_bodies: Optional[int] = None

        if config.render and scene_path:
            self._scene = xml_et.parse(scene_path)
            self._root = self._scene.getroot()
            self._worldbody = self._root.find("worldbody")
        else:
            self._scene = self._root = self._worldbody = None

    # --- box accumulation --------------------------------------------------

    def AddBox(
        self,
        position=(1.0, 0.0, 0.0),
        euler=(0.0, 0.0, 0.0),
        size=(2, 2, 2),
    ) -> None:
        if self.render and self._worldbody is not None:
            if self.max_bodies is not None and self.count_boxes >= self.max_bodies - 1:
                return
            body = xml_et.SubElement(self._worldbody, "body")
            body.attrib["pos"] = list_to_str(position)
            body.attrib["quat"] = list_to_str(euler_to_quat(*euler))
            body.attrib["name"] = random_box_name()
            geo = xml_et.SubElement(body, "geom")
            geo.attrib["type"] = "box"
            geo.attrib["size"] = list_to_str(0.5 * np.array(size))
            geo.attrib["contype"] = "2"
            geo.attrib["conaffinity"] = "1"
        else:
            self.box_data.append(BoxData(
                pos=np.asarray(position, dtype=float),
                size=0.5 * np.asarray(size, dtype=float),
                quat=euler_to_quat(*euler),
            ))
        self.count_boxes += 1

    # --- terrain builders --------------------------------------------------

    @timing(repeat=1)
    def AddStairs(self, init_pos=(0.0, 0.0, 0.0), yaw: float = 0.0) -> None:
        cfg = self.config
        local_pos = [-cfg.width / 2, cfg.length / 2, 0.0]
        for _ in range(cfg.num_stairs):
            local_pos[0] += cfg.width
            local_pos[2] += cfg.step_height
            x, y = rot2d(
                local_pos[0] - cfg.num_stairs * cfg.width / 2,
                local_pos[1] - cfg.num_stairs * cfg.width / 2,
                yaw,
            )
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2] / 2 + init_pos[2]],
                [0.0, 0.0, yaw],
                [cfg.width, cfg.length, local_pos[2]],
            )

    def AddFlat(self, init_pos=(0.0, 0.0, 0.0), height: float = 1.0, width: float = 0.1) -> None:
        if height > 0.0:
            self.AddBox([init_pos[0], init_pos[1], height / 2], [0, 0, 0],
                        [self.length, self.length, height])
        else:
            self.AddBox([init_pos[0], init_pos[1], height - width / 2], [0, 0, 0],
                        [self.length, self.length, width])

    def AddTurningStairsUp(self, init_pos=(0.0, 0.0, 0.0), yaw: float = np.pi / 2) -> None:
        cfg = self.config
        w, h, n = cfg.width, cfg.step_height, cfg.num_stairs
        lp = [-n * w / 2 - w / 2, n * w / 2, 0.0]
        for i in range(n):
            lp[0] += w; lp[1] -= w / 2; lp[2] += h
            x, y = rot2d(lp[0], lp[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], lp[2] / 2 + init_pos[2]],
                        [0, 0, yaw], [w, w + w * i, lp[2]])
        lp = [n * w / 2 - w / 2, n * w / 2, h]
        for i in range(n - 1):
            lp[0] -= w; lp[1] -= w / 2; lp[2] += h
            x, y = rot2d(lp[0], lp[1], yaw + np.pi / 2)
            self.AddBox([x + init_pos[0], y + init_pos[1], lp[2] / 2 + init_pos[2]],
                        [0, 0, yaw + np.pi / 2], [w, w + w * i, lp[2]])

    def AddTurningStairsDown(self, init_pos=(0.0, 0.0, 0.0), yaw: float = np.pi / 2) -> None:
        cfg = self.config
        w, h, n = cfg.width, cfg.step_height, cfg.num_stairs
        lp = [-n * w / 2 - w / 2, n * w / 2, n * h + h]
        for i in range(n):
            lp[0] += w; lp[1] -= w / 2; lp[2] -= h
            x, y = rot2d(lp[0], lp[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], lp[2] / 2 + init_pos[2]],
                        [0, 0, yaw], [w, w + w * i, lp[2]])
        lp = [n * w / 2 - w / 2, n * w / 2, (n - 1) * h + h]
        for i in range(n - 1):
            lp[0] -= w; lp[1] -= w / 2; lp[2] -= h
            x, y = rot2d(lp[0], lp[1], yaw + np.pi / 2)
            self.AddBox([x + init_pos[0], y + init_pos[1], lp[2] / 2 + init_pos[2]],
                        [0, 0, yaw + np.pi / 2], [w, w + w * i, lp[2]])

    @timing(repeat=1)
    def AddRoughGround(
        self,
        init_pos=(1.0, 0.0, 0.0),
        euler=(0.0, 0.0, 0.0),
        nums=(10, 10),
        box_size=(0.5, 0.5, 0.5),
        box_euler=(0.0, 0.0, 0.0),
        separation=(0.2, 0.2),
        box_size_rand=(0.05, 0.05, 0.05),
        box_euler_rand=(0.2, 0.2, 0.2),
        separation_rand=(0.05, 0.05),
    ) -> None:
        """C: Delegates generation to RoughGroundBuilder; applies transform here."""
        builder = RoughGroundBuilder(
            nums=list(nums),
            box_size=list(box_size),
            box_euler=list(box_euler),
            separation=list(separation),
            box_size_rand=list(box_size_rand),
            box_euler_rand=list(box_euler_rand),
            separation_rand=list(separation_rand),
        )
        local_positions, rng_eulers, rng_sizes = builder.build()

        # Apply rotation matrix to all positions at once (vectorized)
        R = euler_to_rot(*euler)
        world_positions = (R @ local_positions.T).T + np.asarray(init_pos)

        for pos, eul, sz in zip(world_positions, rng_eulers, rng_sizes):
            self.AddBox(pos.tolist(), eul.tolist(), sz.tolist())

    @log_call
    def Save(self, filename: Optional[str] = None) -> None:
        if self._scene is None:
            raise RuntimeError("No scene loaded. Use render=True with a valid scene_path.")
        self._scene.write(filename or "terrain_scene_mjx.xml")


# ---------------------------------------------------------------------------
# WFC-based terrain generation helpers (compatible with before/ API)
# ---------------------------------------------------------------------------

# Lazy import: WFCCore lives in after.wfc.wfc; getIndexes lives alongside this file.
def _load_wfc():
    from ..wfc.wfc import WFCConfig, WFCCore
    return WFCConfig, WFCCore

def _load_indexes():
    from .getIndexes import Stairs, StairsTurningUp, StairsTurningDown
    return Stairs, StairsTurningUp, StairsTurningDown


@validate_grid(min_size=2, max_size=200)
def generate_discrete(size: int) -> np.ndarray:
    """D: @validate_grid guards input before expensive WFC setup."""
    WFCConfig, WFCCore = _load_wfc()
    connections = {
        0: {(-1, 0): (0, 1), (1, 0): (0, 1), (0, -1): (0, 1), (0, 1): (0, 1)},
        1: {(-1, 0): (0, 1), (1, 0): (0, 1), (0, -1): (0, 1), (0, 1): (0, 1)},
    }
    config = WFCConfig(n_tiles=2, shape=(size, size), connections=connections)
    core = WFCCore(config)
    core.init_randomly()
    return core.run()


@validate_grid(min_size=2, max_size=200)
def generate_14(size: int, test: bool = False) -> np.ndarray:
    """D: @validate_grid guards input before expensive WFC setup."""
    WFCConfig, WFCCore = _load_wfc()
    Stairs, StairsTurningUp, StairsTurningDown = _load_indexes()

    left, right, up, down = (-1, 0), (1, 0), (0, 1), (0, -1)
    directions = [left, down, right, up]
    connections = {
        0: {left: (0, 4, 10, 11), down: (0, 5, 11, 12), right: (0, 2, 12, 13), up: (0, 3, 13, 10)},
        1: {left: (1, 2, 6, 7),   down: (1, 3, 7, 8),   right: (1, 4, 8, 9),   up: (1, 5, 9, 6)},
    }
    connections.update(Stairs(directions))
    connections.update(StairsTurningUp(directions))
    connections.update(StairsTurningDown(directions))

    config = WFCConfig(n_tiles=14, shape=(size, size), connections=connections)
    core = WFCCore(config)

    outer = 0
    for x in range(size):
        core.init((x, 0), outer)
        core.init((x, size - 1), outer)
    for y in range(1, size - 1):
        core.init((0, y), outer)
        core.init((size - 1, y), outer)
    core.init((size // 2, size // 2), 0 if (test or np.random.random() > 0.5) else 1)

    return core.run()


def create_centered_grid(N: int, d: float) -> np.ndarray:
    half = (N - 1) / 2
    x = (np.arange(N) - half) * d
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.stack([X, Y], axis=-1)


def addElement(tg: TerrainGenerator, index: int, pos) -> None:
    height = tg.block_height
    dispatch = {
        1:  lambda: tg.AddFlat(init_pos=[pos[0], pos[1], 0.], height=height),
        2:  lambda: tg.AddStairs(init_pos=[pos[0], pos[1], 0], yaw=0),
        3:  lambda: tg.AddStairs(init_pos=[pos[0], pos[1], 0], yaw=np.pi / 2),
        4:  lambda: tg.AddStairs(init_pos=[pos[0], pos[1], 0], yaw=np.pi),
        5:  lambda: tg.AddStairs(init_pos=[pos[0], pos[1], 0], yaw=-np.pi / 2),
        6:  lambda: tg.AddTurningStairsUp(init_pos=[pos[0], pos[1], 0.], yaw=0.),
        7:  lambda: tg.AddTurningStairsUp(init_pos=[pos[0], pos[1], 0.], yaw=np.pi / 2),
        8:  lambda: tg.AddTurningStairsUp(init_pos=[pos[0], pos[1], 0.], yaw=np.pi),
        9:  lambda: tg.AddTurningStairsUp(init_pos=[pos[0], pos[1], 0.], yaw=-np.pi / 2),
        10: lambda: tg.AddTurningStairsDown(init_pos=[pos[0], pos[1], 0], yaw=0),
        11: lambda: tg.AddTurningStairsDown(init_pos=[pos[0], pos[1], 0], yaw=np.pi / 2),
        12: lambda: tg.AddTurningStairsDown(init_pos=[pos[0], pos[1], 0], yaw=np.pi),
        13: lambda: tg.AddTurningStairsDown(init_pos=[pos[0], pos[1], 0], yaw=-np.pi / 2),
    }
    action = dispatch.get(index)
    if action is not None:
        action()
