"""
Benchmark: before vs. after optimization.

Usage:
    python benchmark/run_benchmark.py --mode before   # only before
    python benchmark/run_benchmark.py --mode after    # only after
    python benchmark/run_benchmark.py --mode compare  # side-by-side (default)

Docker:
    docker compose -f docker/docker-compose.yml run before
    docker compose -f docker/docker-compose.yml run after
    docker compose -f docker/docker-compose.yml run compare

Measures:
  - WFCCore.solve() : grid sizes 5, 8, 10, 12, 15
  - TerrainGenerator.AddRoughGround() : nums N×N, N in [5, 10, 15, 20, 30]

Each configuration is repeated N_REPEAT times; mean and std-dev are reported.
Results saved to results/benchmark_results.csv.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import os
import statistics
import sys
import time
import tracemalloc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIXTURE_SCENE = os.path.join(ROOT, "benchmark", "fixtures", "minimal_scene.xml")
RESULTS_DIR   = os.path.join(ROOT, "results")
RESULTS_CSV   = os.path.join(RESULTS_DIR, "benchmark_results.csv")

N_REPEAT   = 7
WFC_SIZES  = [5, 8, 10, 12, 15]
ROUGH_NUMS = [5, 10, 15, 20, 30]

COLUMNS = [
    "experiment", "version", "param", "param_value",
    "mean_ms", "stdev_ms", "mem_kb",
]


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silence():
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            yield


def measure(fn, *, repeat: int = N_REPEAT):
    """Return (mean_ms, stdev_ms, mean_peak_kb)."""
    times: list[float] = []
    peaks: list[float] = []
    for _ in range(repeat):
        tracemalloc.start()
        t0 = time.perf_counter()
        with _silence():
            fn()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed * 1_000)
        peaks.append(peak / 1_024)
    return (
        statistics.mean(times),
        statistics.stdev(times) if len(times) > 1 else 0.0,
        statistics.mean(peaks),
    )


# ---------------------------------------------------------------------------
# WFC benchmarks
# ---------------------------------------------------------------------------

def _wfc_connections_2tile():
    return {
        0: {(-1, 0): (0, 1), (1, 0): (0, 1), (0, -1): (0, 1), (0, 1): (0, 1)},
        1: {(-1, 0): (0, 1), (1, 0): (0, 1), (0, -1): (0, 1), (0, 1): (0, 1)},
    }


def bench_wfc_before(size: int):
    path = os.path.join(ROOT, "src", "before", "wfc", "wfc.py")
    spec = importlib.util.spec_from_file_location("_before_wfc", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    wfc = mod.WFCCore(2, _wfc_connections_2tile(), (size, size))
    wfc.init_randomly()
    wfc.solve()


def bench_wfc_after(size: int):
    from src.after.wfc.wfc import WFCConfig, WFCCore
    config = WFCConfig(n_tiles=2, shape=(size, size), connections=_wfc_connections_2tile())
    core = WFCCore(config)
    core.init_randomly()
    core.run()


# ---------------------------------------------------------------------------
# Terrain / RoughGround benchmarks
# ---------------------------------------------------------------------------

def _inject_before_paths():
    for p in (
        os.path.join(ROOT, "src", "before", "terrain"),
        os.path.join(ROOT, "src", "before"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def bench_terrain_before(nums_side: int):
    _inject_before_paths()
    gen_path = os.path.join(ROOT, "src", "before", "terrain", "generator.py")
    spec = importlib.util.spec_from_file_location("_before_gen", gen_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Patch AFTER exec so module-level assignments don't overwrite it
    mod.INPUT_SCENE_PATH = FIXTURE_SCENE

    tg = mod.TerrainGenerator(width=0.3, step_height=0.1, num_stairs=3, render=False)
    tg.AddRoughGround(nums=[nums_side, nums_side])


def bench_terrain_after(nums_side: int):
    from src.after.terrain.generator import TerrainConfig, TerrainGenerator
    cfg = TerrainConfig.from_args(width=0.3, step_height=0.1, num_stairs=3)
    tg  = TerrainGenerator(cfg)
    tg.AddRoughGround(nums=(nums_side, nums_side))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _row(writer, exp, version, param, val, mean, stdev, mem):
    writer.writerow({
        "experiment": exp, "version": version,
        "param": param, "param_value": val,
        "mean_ms": f"{mean:.3f}", "stdev_ms": f"{stdev:.3f}",
        "mem_kb": f"{mem:.1f}",
    })


def _header(title: str, col1: str):
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(f"  {col1:>6}  {'mean_ms':>10}  {'stdev_ms':>9}  {'mem_kb':>9}")
    print("  " + "-" * 38)


def run_single(version: str, writer) -> list:
    """Run only one version; return list of result rows."""
    rows = []

    if version == "before":
        bench_wfc   = bench_wfc_before
        bench_rough = bench_terrain_before
    else:
        bench_wfc   = bench_wfc_after
        bench_rough = bench_terrain_after

    _header(f"WFC  [{version}]", "size")
    for size in WFC_SIZES:
        m, s, k = measure(lambda sz=size: bench_wfc(sz))
        print(f"  {size:>6}  {m:>10.2f}  {s:>9.2f}  {k:>9.1f}")
        rows.append(("wfc", version, "grid_size", size, m, s, k))
        if writer:
            _row(writer, "wfc", version, "grid_size", size, m, s, k)

    print()
    _header(f"AddRoughGround  [{version}]", "nums")
    for n in ROUGH_NUMS:
        m, s, k = measure(lambda nn=n: bench_rough(nn))
        print(f"  {n:>6}  {m:>10.2f}  {s:>9.2f}  {k:>9.1f}")
        rows.append(("rough_ground", version, "nums_side", n, m, s, k))
        if writer:
            _row(writer, "rough_ground", version, "nums_side", n, m, s, k)

    print()
    return rows


def run_compare(writer):
    """Run both versions and print side-by-side speedup table."""
    _header("WFC  [before vs after]", "size")
    print(f"  {'size':>6}  {'before ms':>10}  {'after ms':>10}  {'speedup':>8}")
    print("  " + "-" * 44)

    rows = []
    for size in WFC_SIZES:
        bm, bs, bk = measure(lambda sz=size: bench_wfc_before(sz))
        am, as_, ak = measure(lambda sz=size: bench_wfc_after(sz))
        sp = bm / am if am > 0 else float("inf")
        print(f"  {size:>6}  {bm:>9.2f}±{bs:<4.2f}  {am:>9.2f}±{as_:<4.2f}  {sp:>7.2f}×")
        for v, m, s, k in [("before", bm, bs, bk), ("after", am, as_, ak)]:
            rows.append(("wfc", v, "grid_size", size, m, s, k))
            if writer:
                _row(writer, "wfc", v, "grid_size", size, m, s, k)

    print()
    _header("AddRoughGround  [before vs after]", "nums")
    print(f"  {'nums':>6}  {'before ms':>10}  {'after ms':>10}  {'speedup':>8}")
    print("  " + "-" * 44)

    for n in ROUGH_NUMS:
        bm, bs, bk = measure(lambda nn=n: bench_terrain_before(nn))
        am, as_, ak = measure(lambda nn=n: bench_terrain_after(nn))
        sp = bm / am if am > 0 else float("inf")
        print(f"  {n:>6}  {bm:>9.2f}±{bs:<4.2f}  {am:>9.2f}±{as_:<4.2f}  {sp:>7.2f}×")
        for v, m, s, k in [("before", bm, bs, bk), ("after", am, as_, ak)]:
            rows.append(("rough_ground", v, "nums_side", n, m, s, k))
            if writer:
                _row(writer, "rough_ground", v, "nums_side", n, m, s, k)

    print()
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WFC & terrain benchmark")
    parser.add_argument(
        "--mode",
        choices=["before", "after", "compare"],
        default="compare",
        help="Which version(s) to benchmark (default: compare)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if os.path.getsize(RESULTS_CSV) == 0:
            writer.writeheader()

        if args.mode in ("before", "after"):
            run_single(args.mode, writer)
        else:
            run_compare(writer)

    print(f"Results appended → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
