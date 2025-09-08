#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
from ipfn import ipfn

# Optional Polars import to benchmark polars backend when available
try:
    import polars as pl  # type: ignore
except Exception:
    pl = None


def time_run(name, builder_fn, repeats=1000, algorithms=("legacy", "optimized")):
    print(f"\n=== {name} ===")
    times = {algo: [] for algo in algorithms}
    results = {}
    for algo in algorithms:
        for _ in range(repeats):
            original, aggregates, dimensions, weight_col = builder_fn()
            # If benchmarking polars and original is pandas, convert to Polars to begin with
            if (pl is not None) and (algo == 'polars'):
                if isinstance(original, pd.DataFrame):
                    try:
                        original = pl.from_pandas(original)
                    except Exception:
                        original = pl.DataFrame(original)
            IPF = ipfn(original, aggregates, dimensions, convergence_rate=1e-6, algorithm=algo)
            t0 = time.perf_counter()
            out = IPF.iteration()
            t1 = time.perf_counter()
            times[algo].append(t1 - t0)
            results[algo] = out
        print(f"{algo:9s} avg time over {repeats}: {np.mean(times[algo]):.6f}s (min {np.min(times[algo]):.6f}s, max {np.max(times[algo]):.6f}s)")
    return results


def wikipedia_2d_example():
    def build():
        m = np.array([[40, 30, 20, 10], [35, 50, 100, 75], [30, 80, 70, 120], [20, 30, 40, 50]], dtype=float)
        xip = np.array([150, 300, 400, 150], dtype=float)
        xpj = np.array([200, 300, 400, 100], dtype=float)
        aggregates = [xip, xpj]
        dimensions = [[0], [1]]
        return m.copy(), aggregates, dimensions, 'total'
    return build


def three_d_example():
    def build():
        m = np.zeros((2, 4, 3), dtype=float)
        m[0,0,0] = 1; m[0,0,1] = 2; m[0,0,2] = 1
        m[0,1,0] = 3; m[0,1,1] = 5; m[0,1,2] = 5
        m[0,2,0] = 6; m[0,2,1] = 2; m[0,2,2] = 2
        m[0,3,0] = 1; m[0,3,1] = 7; m[0,3,2] = 2
        m[1,0,0] = 5; m[1,0,1] = 4; m[1,0,2] = 2
        m[1,1,0] = 5; m[1,1,1] = 5; m[1,1,2] = 5
        m[1,2,0] = 3; m[1,2,1] = 8; m[1,2,2] = 7
        m[1,3,0] = 2; m[1,3,1] = 7; m[1,3,2] = 6
        xipp = np.array([52, 48], dtype=float)
        xpjp = np.array([20, 30, 35, 15], dtype=float)
        xppk = np.array([35, 40, 25], dtype=float)
        xijp = np.array([[9, 17, 19, 7], [11, 13, 16, 8]], dtype=float)
        xpjk = np.array([[7, 9, 4], [8, 12, 10], [15, 12, 8], [5, 7, 3]], dtype=float)
        aggregates = [xipp, xpjp, xppk, xijp, xpjk]
        dimensions = [[0], [1], [2], [0, 1], [1, 2]]
        return m.copy(), aggregates, dimensions, 'total'
    return build


def pandas_example():
    def build():
        age = [30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50]
        distance = [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]
        m = [8., 4., 6., 7., 3., 6., 5., 2., 9., 11., 3., 1.]
        df = pd.DataFrame({'age': age, 'distance': distance, 'total': m})
        xip = df.groupby('age')['total'].sum()
        xip.loc[30] = 20
        xip.loc[40] = 18
        xip.loc[50] = 22
        xpj = df.groupby('distance')['total'].sum()
        xpj.loc[10] = 18
        xpj.loc[20] = 16
        xpj.loc[30] = 12
        xpj.loc[40] = 14
        dimensions = [['age'], ['distance']]
        aggregates = [xip, xpj]
        return df.copy(), aggregates, dimensions, 'total'
    return build


def main():
    print("Benchmarking IPFN: legacy vs optimized (+ polars if available)")
    time_run('Wikipedia 2D (NumPy)', wikipedia_2d_example(), repeats=10)
    time_run('3D example (NumPy)', three_d_example(), repeats=10)
    algos_pd = ("legacy", "optimized") + (("polars",) if pl is not None else tuple())
    time_run('Pandas example', pandas_example(), repeats=10, algorithms=algos_pd)
    # New required 6D/1500-rows benchmark with 100 repetitions and time/CPU/RAM metrics
    benchmark_6d_1500(repeats=100)


if __name__ == '__main__':
    main()


# ---- 6D 1500-rows benchmark with time/CPU/RAM measurements ----

def _build_6d_dataset_1500(seed: int = 42):
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(seed)
    n_rows = 1500
    # Define 6 dimensions with varying number of categories
    dims = {
        'd1': [f'a{i}' for i in range(5)],
        'd2': [f'b{i}' for i in range(5)],
        'd3': [f'c{i}' for i in range(5)],
        'd4': [f'd{i}' for i in range(5)],
        'd5': [f'e{i}' for i in range(3)],
        'd6': [f'f{i}' for i in range(4)],
    }
    data = {}
    for k, cats in dims.items():
        data[k] = rng.choice(cats, size=n_rows, replace=True)
    # Positive weights
    data['total'] = rng.gamma(shape=2.0, scale=5.0, size=n_rows).astype(float)
    df = pd.DataFrame(data)

    # Build single-dimension marginals with a small random perturbation and rescale to preserve total
    total_sum = df['total'].sum()
    aggregates = []
    dimensions = []
    for k in dims.keys():
        grp = df.groupby(k)['total'].sum()
        # Perturb each group by up to ±10%
        noise = rng.uniform(0.9, 1.1, size=len(grp))
        target = grp.values * noise
        # Rescale to match original total sum for stability
        target *= (total_sum / target.sum())
        agg = pd.Series(target, index=grp.index, name='total')
        aggregates.append(agg)
        dimensions.append([k])

    return df, aggregates, dimensions, 'total'


def _measure_ipfn_run(algo: str, original, aggregates, dimensions, weight_col='total'):
    import time
    import copy as _copy
    import tracemalloc

    # Optional psutil import
    try:
        import psutil  # type: ignore
        proc = psutil.Process()
    except Exception:
        psutil = None  # type: ignore
        proc = None

    # Prepare input per algorithm (convert to Polars if requested and available)
    local_original = original.copy(deep=True) if hasattr(original, 'copy') else _copy.deepcopy(original)
    if (pl is not None) and (algo == 'polars'):
        try:
            import polars as _pl  # type: ignore
            if not isinstance(local_original, _pl.DataFrame):
                try:
                    local_original = _pl.from_pandas(local_original)
                except Exception:
                    local_original = _pl.DataFrame(local_original)
        except Exception:
            pass

    # Measurements
    cpu_t0 = time.process_time()
    wall_t0 = time.perf_counter()

    tracemalloc.start()
    rss_before = proc.memory_info().rss if proc else None

    IPF = ipfn(local_original, _copy.deepcopy(aggregates), _copy.deepcopy(dimensions), convergence_rate=1e-6, algorithm=algo)
    _ = IPF.iteration()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    rss_after = proc.memory_info().rss if proc else None

    return {
        'wall_time_s': wall_t1 - wall_t0,
        'cpu_time_s': cpu_t1 - cpu_t0,
        'peak_tracemalloc_bytes': int(peak),
        'rss_before_bytes': int(rss_before) if rss_before is not None else None,
        'rss_after_bytes': int(rss_after) if rss_after is not None else None,
        'rss_delta_bytes': (int(rss_after) - int(rss_before)) if (rss_before is not None and rss_after is not None) else None,
    }


def benchmark_6d_1500(repeats: int = 100):
    print("\nBenchmark: 6D dataset with 1500 rows, 100 runs per algorithm")
    # Determine algorithms: always legacy and optimized; include polars when available
    algos = ["legacy", "optimized"] + (["polars"] if pl is not None else [])

    original, aggregates, dimensions, weight_col = _build_6d_dataset_1500()

    import numpy as _np
    results_summary = {}
    for algo in algos:
        wall_times = []
        cpu_times = []
        peaks = []
        rss_deltas = []
        rss_after = []
        for _ in range(repeats):
            m = _measure_ipfn_run(algo, original, aggregates, dimensions, weight_col)
            wall_times.append(m['wall_time_s'])
            cpu_times.append(m['cpu_time_s'])
            peaks.append(m['peak_tracemalloc_bytes'])
            if m['rss_delta_bytes'] is not None:
                rss_deltas.append(m['rss_delta_bytes'])
            if m['rss_after_bytes'] is not None:
                rss_after.append(m['rss_after_bytes'])
        summary = {
            'wall_time_avg_s': float(_np.mean(wall_times)),
            'wall_time_min_s': float(_np.min(wall_times)),
            'wall_time_max_s': float(_np.max(wall_times)),
            'cpu_time_avg_s': float(_np.mean(cpu_times)),
            'peak_tracemalloc_avg_bytes': int(_np.mean(peaks)),
        }
        if rss_deltas:
            summary['rss_delta_avg_bytes'] = float(_np.mean(rss_deltas))
        if rss_after:
            summary['rss_after_avg_bytes'] = float(_np.mean(rss_after))
        results_summary[algo] = summary

        # Pretty print summary
        print(f"{algo:9s} | wall avg {summary['wall_time_avg_s']:.6f}s (min {summary['wall_time_min_s']:.6f}, max {summary['wall_time_max_s']:.6f}) | "
              f"cpu avg {summary['cpu_time_avg_s']:.6f}s | peak_py_mem avg {summary['peak_tracemalloc_avg_bytes']/1_048_576:.3f} MiB" +
              (f" | rss_delta avg {summary['rss_delta_avg_bytes']/1_048_576:.3f} MiB | rss_after avg {summary['rss_after_avg_bytes']/1_048_576:.3f} MiB" if 'rss_delta_avg_bytes' in summary else ""))

    return results_summary
