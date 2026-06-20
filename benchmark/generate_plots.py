"""
Generate all report figures:
  1. WFC grid visualizations — before and after (same algorithm output, different class structure)
  2. WFC benchmark bar chart — execution time before vs after
  3. WFC speedup line chart
  4. AddRoughGround benchmark bar chart
  5. AddRoughGround speedup line chart
  6. Memory usage comparison
  7. Terrain box scatter plot — before and after AddRoughGround

Saves all figures to results/plots/.

Run:
    python benchmark/generate_plots.py
"""
from __future__ import annotations

import contextlib
import io
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# Register NanumGothic for Korean labels (installed via fonts-nanum apt package)
_nanum_candidates = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/nanum/NanumGothic.otf",
]
for _fp in _nanum_candidates:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PLOTS_DIR  = os.path.join(ROOT, "results", "plots")
FIXTURE    = os.path.join(ROOT, "benchmark", "fixtures", "minimal_scene.xml")

BEFORE_COLOR = "#E07B54"
AFTER_COLOR  = "#4C9BE8"
SPEEDUP_COLOR = "#56B87E"

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
# Style helpers
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


# ---------------------------------------------------------------------------
# Figure 1 & 2: WFC grid visualizations (before / after classes, same output)
# ---------------------------------------------------------------------------

TILE_COLORS = {
    0:  "#A8D8A8",   # flat-empty (light green)
    1:  "#4CAF50",   # flat-filled (green)
    2:  "#2196F3",   # stair E
    3:  "#03A9F4",   # stair N
    4:  "#00BCD4",   # stair W
    5:  "#009688",   # stair S
    6:  "#FF9800",   # turn-up
    7:  "#FF5722",   # turn-up
    8:  "#F44336",   # turn-up
    9:  "#E91E63",   # turn-up
    10: "#9C27B0",   # turn-down
    11: "#673AB7",   # turn-down
    12: "#3F51B5",   # turn-down
    13: "#1A237E",   # turn-down
}

TILE_NAMES = {
    0: "Flat (empty)", 1: "Flat (solid)",
    2: "Stair E", 3: "Stair N", 4: "Stair W", 5: "Stair S",
    6: "TurnUp (0°)", 7: "TurnUp (90°)", 8: "TurnUp (180°)", 9: "TurnUp (270°)",
    10: "TurnDn (0°)", 11: "TurnDn (90°)", 12: "TurnDn (180°)", 13: "TurnDn (270°)",
}


def _run_wfc_before(size: int) -> np.ndarray:
    path = os.path.join(ROOT, "src", "before", "wfc", "wfc.py")
    spec = importlib.util.spec_from_file_location("_bwfc", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    connections = _wfc_14_connections()
    wfc = mod.WFCCore(14, connections, (size, size))
    _init_border(wfc, size)
    wfc.solve()
    return wfc.wave.wave


def _run_wfc_after(size: int) -> np.ndarray:
    from src.after.wfc.wfc import WFCConfig, WFCCore
    config = WFCConfig(n_tiles=14, shape=(size, size), connections=_wfc_14_connections())
    core = WFCCore(config)
    _init_border_after(core, size)
    return core.run()


def _wfc_14_connections():
    here = os.path.join(ROOT, "src", "after", "terrain")
    if here not in sys.path:
        sys.path.insert(0, here)
    from src.after.terrain.getIndexes import Stairs, StairsTurningUp, StairsTurningDown
    left, right, up, down = (-1, 0), (1, 0), (0, 1), (0, -1)
    dirs = [left, down, right, up]
    c = {
        0: {left: (0,4,10,11), down: (0,5,11,12), right: (0,2,12,13), up: (0,3,13,10)},
        1: {left: (1,2,6,7),   down: (1,3,7,8),   right: (1,4,8,9),  up: (1,5,9,6)},
    }
    c.update(Stairs(dirs))
    c.update(StairsTurningUp(dirs))
    c.update(StairsTurningDown(dirs))
    return c


def _init_border(wfc, size):
    for x in range(size):
        wfc.init((x, 0), 0); wfc.init((x, size-1), 0)
    for y in range(1, size-1):
        wfc.init((0, y), 0); wfc.init((size-1, y), 0)
    wfc.init((size//2, size//2), 1)


def _init_border_after(core, size):
    for x in range(size):
        core.init((x, 0), 0); core.init((x, size-1), 0)
    for y in range(1, size-1):
        core.init((0, y), 0); core.init((size-1, y), 0)
    core.init((size//2, size//2), 1)


def _render_wave(ax, wave: np.ndarray, title: str):
    size = wave.shape[0]
    color_img = np.zeros((size, size, 3))
    for tile_id, hex_color in TILE_COLORS.items():
        r, g, b = int(hex_color[1:3], 16)/255, int(hex_color[3:5], 16)/255, int(hex_color[5:7], 16)/255
        mask = wave == tile_id
        color_img[mask] = [r, g, b]

    ax.imshow(color_img, origin="upper", interpolation="nearest")
    ax.set_title(title, pad=8)
    ax.set_xlabel("col"); ax.set_ylabel("row")
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.tick_params(labelsize=7)

    for i in range(size):
        for j in range(size):
            ax.text(j, i, str(wave[i, j]), ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold")


def plot_wfc_grids(size: int = 9):
    _apply_style()
    np.random.seed(42)
    with _silence():
        wave_b = _run_wfc_before(size)
    np.random.seed(42)
    with _silence():
        wave_a = _run_wfc_after(size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _render_wave(axes[0], wave_b, f"WFC 출력 — Before  ({size}×{size})")
    _render_wave(axes[1], wave_a, f"WFC 출력 — After   ({size}×{size})")

    # shared legend
    patches = [mpatches.Patch(color=TILE_COLORS[i], label=f"{i}: {TILE_NAMES[i]}")
               for i in range(14)]
    fig.legend(handles=patches, loc="lower center", ncol=7,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("WFC 타일 배치 결과 (tile index별 색상)", fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig1_wfc_grids.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Terrain box scatter (before / after)
# ---------------------------------------------------------------------------

def _get_boxes_before(nums_side: int) -> list:
    before_paths = [
        os.path.join(ROOT, "src", "before", "terrain"),
        os.path.join(ROOT, "src", "before"),
    ]
    for p in before_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    path = os.path.join(ROOT, "src", "before", "terrain", "generator.py")
    spec = importlib.util.spec_from_file_location("_bgen", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.INPUT_SCENE_PATH = FIXTURE
    np.random.seed(7)
    tg = mod.TerrainGenerator(width=0.3, step_height=0.1, num_stairs=3, render=False)
    tg.AddRoughGround(nums=[nums_side, nums_side])
    return tg.box_data   # list of dicts


def _get_boxes_after(nums_side: int) -> list:
    from src.after.terrain.generator import TerrainConfig, TerrainGenerator
    np.random.seed(7)
    cfg = TerrainConfig.from_args(width=0.3, step_height=0.1, num_stairs=3)
    tg  = TerrainGenerator(cfg)
    tg.AddRoughGround(nums=(nums_side, nums_side))
    return tg.box_data   # list of BoxData


def plot_terrain_scatter(nums_side: int = 15):
    _apply_style()
    with _silence():
        boxes_b = _get_boxes_before(nums_side)
        boxes_a = _get_boxes_after(nums_side)

    def _extract(boxes, is_dict):
        if is_dict:
            xs = [b["pos"][0] for b in boxes]
            ys = [b["pos"][1] for b in boxes]
            zs = [b["size"][2] for b in boxes]
        else:
            xs = [b.pos[0] for b in boxes]
            ys = [b.pos[1] for b in boxes]
            zs = [b.size[2] for b in boxes]
        return xs, ys, zs

    xb, yb, zb = _extract(boxes_b, is_dict=True)
    xa, ya, za = _extract(boxes_a, is_dict=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, xs, ys, zs, title, color in [
        (axes[0], xb, yb, zb, f"Before — {len(boxes_b)} boxes", BEFORE_COLOR),
        (axes[1], xa, ya, za, f"After  — {len(boxes_a)} boxes", AFTER_COLOR),
    ]:
        sc = ax.scatter(xs, ys, c=zs, cmap="YlOrRd", s=40, edgecolors="white", linewidths=0.4)
        ax.set_title(title, pad=8)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="half-height (m)", fraction=0.04)

    fig.suptitle(f"AddRoughGround 박스 배치 ({nums_side}×{nums_side})", fontsize=14)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig2_terrain_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 & 5: Benchmark bar + speedup charts
# ---------------------------------------------------------------------------

def _load_csv() -> dict:
    import csv
    path = os.path.join(ROOT, "results", "benchmark_results.csv")
    data: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            exp  = row["experiment"]
            ver  = row["version"]
            pval = int(row["param_value"])
            mean = float(row["mean_ms"])
            std  = float(row["stdev_ms"])
            mem  = float(row["mem_kb"])
            data.setdefault(exp, {}).setdefault(ver, {})[pval] = (mean, std, mem)
    return data


def plot_benchmark_charts():
    _apply_style()
    data = _load_csv()

    for exp_key, xlabel, title_prefix in [
        ("wfc",          "Grid size (N×N)", "WFC solve()"),
        ("rough_ground", "Boxes per side N (N×N total)", "AddRoughGround"),
    ]:
        exp = data.get(exp_key, {})
        sizes = sorted(set(list(exp.get("before", {}).keys()) + list(exp.get("after", {}).keys())))

        b_means = [exp["before"][s][0] for s in sizes]
        b_stds  = [exp["before"][s][1] for s in sizes]
        a_means = [exp["after"][s][0]  for s in sizes]
        a_stds  = [exp["after"][s][1]  for s in sizes]
        speedups = [b / a for b, a in zip(b_means, a_means)]

        x = np.arange(len(sizes))
        w = 0.35

        # --- bar chart ---
        fig, ax = plt.subplots(figsize=(9, 5))
        bars_b = ax.bar(x - w/2, b_means, w, yerr=b_stds, capsize=4,
                        color=BEFORE_COLOR, label="Before", alpha=0.9)
        bars_a = ax.bar(x + w/2, a_means, w, yerr=a_stds, capsize=4,
                        color=AFTER_COLOR, label="After",  alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(sizes)
        ax.set_xlabel(xlabel); ax.set_ylabel("실행 시간 (ms)")
        ax.set_title(f"{title_prefix} — 실행 시간 비교 (n={7}회 평균 ± std)")
        ax.legend()
        # annotate speedup above each pair
        for xi, sp in zip(x, speedups):
            ax.annotate(f"{sp:.1f}×", xy=(xi, max(b_means[x.tolist().index(xi)],
                                                    a_means[x.tolist().index(xi)]) * 1.05),
                        ha="center", fontsize=9, color=SPEEDUP_COLOR, fontweight="bold")
        fig.tight_layout()
        slug = exp_key.replace("_", "")
        out = os.path.join(PLOTS_DIR, f"fig_bench_{slug}_bar.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

        # --- speedup line ---
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sizes, speedups, marker="o", color=SPEEDUP_COLOR, linewidth=2, markersize=8)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="1× (no gain)")
        ax.fill_between(sizes, 1, speedups, alpha=0.15, color=SPEEDUP_COLOR)
        for s, sp in zip(sizes, speedups):
            ax.annotate(f"{sp:.2f}×", (s, sp), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel("속도 향상 (before / after)")
        ax.set_title(f"{title_prefix} — 속도 향상 배율")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"fig_bench_{slug}_speedup.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 6: Memory comparison
# ---------------------------------------------------------------------------

def plot_memory_charts():
    _apply_style()
    data = _load_csv()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, exp_key, xlabel, subtitle in [
        (axes[0], "wfc",          "Grid size (N×N)",           "WFC — 메모리"),
        (axes[1], "rough_ground", "Boxes per side N (N×N)", "AddRoughGround — 메모리"),
    ]:
        exp   = data.get(exp_key, {})
        sizes = sorted(set(list(exp.get("before", {}).keys()) + list(exp.get("after", {}).keys())))
        b_mem = [exp["before"][s][2] for s in sizes]
        a_mem = [exp["after"][s][2]  for s in sizes]

        x = np.arange(len(sizes)); w = 0.35
        ax.bar(x - w/2, b_mem, w, color=BEFORE_COLOR, label="Before", alpha=0.9)
        ax.bar(x + w/2, a_mem, w, color=AFTER_COLOR,  label="After",  alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(sizes)
        ax.set_xlabel(xlabel); ax.set_ylabel("Peak 메모리 (KB)")
        ax.set_title(subtitle)
        ax.legend()

    fig.suptitle("Peak 메모리 사용량 비교", fontsize=14)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig_memory.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating plots...")
    plot_wfc_grids(size=9)
    plot_terrain_scatter(nums_side=15)
    plot_benchmark_charts()
    plot_memory_charts()
    print("Done. All figures saved to", PLOTS_DIR)
