"""Run with: python -m benchmarks.sampling --device cuda --output benchmarks/results.json."""

import argparse
import importlib
import json
import platform
import time
from pathlib import Path

import torch

from annadca.rbm.sampling import sample


def inputs(kind, batch, length, hidden, states, labels, device, dtype=torch.float32):
    shape = (length, states, hidden) if kind == "categorical" else (length, hidden)
    params = {
        "weight_matrix": torch.randn(shape, device=device, dtype=dtype) * 0.05,
        "label_matrix": torch.randn(labels, hidden, device=device, dtype=dtype) * 0.05,
        "hbias": torch.randn(hidden, device=device, dtype=dtype) * 0.1,
        "vbias": torch.randn(shape[:-1], device=device, dtype=dtype) * 0.1,
        "lbias": torch.randn(labels, device=device, dtype=dtype) * 0.1,
    }
    v = torch.randint(2, (batch, length), device=device).to(dtype)
    if kind == "categorical":
        v = torch.nn.functional.one_hot(
            torch.randint(states, (batch, length), device=device), states
        ).to(dtype)
    l = torch.nn.functional.one_hot(torch.randint(labels, (batch,), device=device), labels).to(
        dtype
    )
    return v, l, params


def measure_all(funcs, device, repeats):
    # Interleave backends to reduce drift from GPU clocks and desktop activity.
    for fn in funcs.values():
        for _ in range(10):
            fn()
    if device == "cuda":
        torch.cuda.synchronize()
    timings = {name: [] for name in funcs}
    names = list(funcs)
    for repeat in range(repeats):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            start = time.perf_counter()
            funcs[name]()
            if device == "cuda":
                torch.cuda.synchronize()
            timings[name].append(1000 * (time.perf_counter() - start))
    results = {}
    for name, values in timings.items():
        t = torch.tensor(values)
        results[name] = {
            "median_ms": t.median().item(),
            "p20_ms": t.quantile(0.2).item(),
            "p80_ms": t.quantile(0.8).item(),
        }
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", default="benchmarks/results.json")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run outside the sandbox or use --device cpu")
    result = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "threads": args.threads,
        "repeats": args.repeats,
        "dtype": "float32",
        "beta": 1.0,
        "cuda": torch.version.cuda,
        "timing": "interleaved synchronized wall time; 10 warmups per backend",
        "results": [],
    }
    with torch.no_grad():
        for kind in ("binary", "categorical"):
            baseline = importlib.import_module(f"benchmarks._baseline_{kind}")
            for batch, length, hidden in [(64, 128, 128), (256, 256, 256), (1024, 512, 256)]:
                v, l, p = inputs(kind, batch, length, hidden, 21, 8, args.device)
                for conditioned in (False, True):
                    for steps in (1, 10):
                        ref = baseline._sample_conditioned if conditioned else baseline._sample
                        funcs = {
                            "baseline": lambda: ref(steps, v, l, p),
                            "torch": lambda: sample(steps, v, l, p, conditioned=conditioned),
                        }
                        if args.device == "cuda":
                            funcs["triton"] = lambda: sample(
                                steps, v, l, p, conditioned=conditioned, backend="triton"
                            )
                        row = dict(
                            kind=kind,
                            batch=batch,
                            length=length,
                            hidden=hidden,
                            states=21 if kind == "categorical" else 2,
                            labels=8,
                            conditioned=conditioned,
                            steps=steps,
                        )
                        row.update(measure_all(funcs, args.device, args.repeats))
                        for name in funcs.keys() - {"baseline"}:
                            row[name]["speedup"] = (
                                row["baseline"]["median_ms"] / row[name]["median_ms"]
                            )
                        result["results"].append(row)
                        print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
