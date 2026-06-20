"""
MuJoCo offscreen terrain renderer.

Generates MuJoCo XML via before/after TerrainGenerator and renders with
mujoco.Renderer (requires MUJOCO_GL=osmesa).

Outputs:
  results/plots/fig_render_roughground.png  — before/after side-by-side
  results/plots/fig_render_wfc_terrain.png  — WFC-based full terrain
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile

import numpy as np
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# paths for before generator's transitive imports (getIndexes, wfc)
for _p in (
    os.path.join(ROOT, "src", "before", "terrain"),
    os.path.join(ROOT, "src", "before"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Korean font
for _fp in (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/nanum/NanumGothic.otf",
):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

RENDER_SCENE = os.path.join(ROOT, "benchmark", "fixtures", "render_scene.xml")
PLOTS_DIR    = os.path.join(ROOT, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Silence helper
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silence():
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            yield


# ---------------------------------------------------------------------------
# Core render helper
# ---------------------------------------------------------------------------

def _mj_render(
    xml_path: str,
    lookat: list,
    distance: float = 6.0,
    azimuth: float = 45.0,
    elevation: float = -30.0,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """Load xml_path with MuJoCo and render to a numpy RGB array."""
    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance  = distance
    cam.azimuth   = azimuth
    cam.elevation = elevation

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

    # render() returns RGBA in some versions; strip alpha
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        pixels = pixels[:, :, :3]
    return pixels.astype(np.uint8)


# ---------------------------------------------------------------------------
# XML builders — before version
# ---------------------------------------------------------------------------

def _build_roughground_before(nums: tuple, seed: int, init_pos: tuple) -> str:
    gen_path = os.path.join(ROOT, "src", "before", "terrain", "generator.py")
    spec = importlib.util.spec_from_file_location("_bgen_render", gen_path)
    mod  = importlib.util.module_from_spec(spec)
    with _silence():
        spec.loader.exec_module(mod)
    mod.INPUT_SCENE_PATH = RENDER_SCENE

    np.random.seed(seed)
    with _silence():
        tg = mod.TerrainGenerator(width=0.3, step_height=0.1, num_stairs=5, render=True)
        tg.AddRoughGround(init_pos=list(init_pos), nums=list(nums))

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False).name
    tg.Save(tmp)
    return tmp


# ---------------------------------------------------------------------------
# XML builders — after version
# ---------------------------------------------------------------------------

def _build_roughground_after(nums: tuple, seed: int, init_pos: tuple) -> str:
    from src.after.terrain.generator import TerrainConfig, TerrainGenerator
    np.random.seed(seed)
    cfg = TerrainConfig.from_args(width=0.3, step_height=0.1, num_stairs=5, render=True)
    tg  = TerrainGenerator(cfg, scene_path=RENDER_SCENE)
    with _silence():
        tg.AddRoughGround(init_pos=init_pos, nums=nums)

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False).name
    tg.Save(tmp)
    return tmp


def _build_wfc_terrain(size: int, seed: int, tile_spacing: float) -> str:
    from src.after.terrain.generator import (
        generate_14, addElement, TerrainConfig, TerrainGenerator,
    )
    np.random.seed(seed)
    with _silence():
        wfc_grid = generate_14(size=size)

    cfg = TerrainConfig.from_args(width=0.3, step_height=0.1, num_stairs=5, render=True)
    tg  = TerrainGenerator(cfg, scene_path=RENDER_SCENE)

    H, W = wfc_grid.shape
    for i in range(H):
        for j in range(W):
            tile = int(wfc_grid[i, j])
            pos  = [(i - H // 2) * tile_spacing,
                    (j - W // 2) * tile_spacing,
                    0.0]
            with _silence():
                addElement(tg, tile, pos)

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False).name
    tg.Save(tmp)
    return tmp


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------

def render_roughground_comparison(nums: tuple = (10, 10), seed: int = 42) -> None:
    """Before vs. after AddRoughGround side-by-side MuJoCo render."""
    # rough terrain centred around (1.1, 1.0, 0) with separation~0.2, nums=10x10
    lookat = [1.1, 1.0, 0.1]
    render_kw = dict(lookat=lookat, distance=5.5, azimuth=45.0, elevation=-28.0)

    print("  rendering before rough ground…")
    tmp_b = _build_roughground_before(nums, seed, init_pos=(0, 0, 0))
    pix_b = _mj_render(tmp_b, **render_kw)
    os.unlink(tmp_b)

    print("  rendering after rough ground…")
    tmp_a = _build_roughground_after(nums, seed, init_pos=(0, 0, 0))
    pix_a = _mj_render(tmp_a, **render_kw)
    os.unlink(tmp_a)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(pix_b)
    axes[0].set_title(f"Before — AddRoughGround ({nums[0]}×{nums[1]})", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(pix_a)
    axes[1].set_title(f"After  — AddRoughGround ({nums[0]}×{nums[1]})", fontsize=12)
    axes[1].axis("off")
    fig.suptitle(
        f"AddRoughGround MuJoCo 렌더링 — Before vs After (seed={seed})",
        fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig_render_roughground.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def render_wfc_terrain_scene(size: int = 7, seed: int = 42) -> None:
    """WFC-generated full terrain (after code) rendered with MuJoCo."""
    tile_spacing = 1.5  # metres between tile centres
    lookat = [0.0, 0.0, 0.3]
    render_kw = dict(lookat=lookat, distance=13.0, azimuth=45.0, elevation=-28.0)

    print("  rendering WFC terrain…")
    tmp = _build_wfc_terrain(size=size, seed=seed, tile_spacing=tile_spacing)
    pix = _mj_render(tmp, **render_kw)
    os.unlink(tmp)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(pix)
    ax.set_title(
        f"WFC 지형 MuJoCo 렌더링 — After ({size}×{size} 타일, seed={seed})",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig_render_wfc_terrain.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("MuJoCo 오프스크린 렌더링 시작 (MUJOCO_GL=" + os.environ.get("MUJOCO_GL", "unset") + ")")
    render_roughground_comparison(nums=(10, 10), seed=42)
    render_wfc_terrain_scene(size=7, seed=42)
    print("완료. 렌더링 이미지 →", PLOTS_DIR)
