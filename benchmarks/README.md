# Sampling benchmark

This benchmark compares the sampling code present in the working tree before
this task (`_baseline_binary.py` and `_baseline_categorical.py`) with the shared
PyTorch Gibbs driver and the opt-in Triton backend. Baselines are snapshots of
the user's current implementation, not the repository's older HEAD revision.

Run from the repository root:

```bash
python -m benchmarks.sampling --device cuda --repeats 60 --output benchmarks/results_cuda.json
```

The committed JSON records the environment, dimensions, and median/p20/p80
latencies. Measurements use float32, beta=1, eight labels, and 21 visible states
for categorical models. Each case gets ten warmups per backend, excluding JIT
compilation. Sixty timed calls per backend are interleaved with rotating order.
CUDA synchronization after each call measures wall latency including Python,
launches, random-number generation, allocation, and device work. Input/model
initialization is outside the timer. No CUDA graphs or reduced matmul precision
are enabled by the benchmark. This is a Gibbs-loop benchmark; the public API's
avoided chain initialization is an additional benefit not measured here.

The RTX A5000 also drives the desktop; timings are not from an isolated GPU and
will vary with other activity and clocks. Speedup is baseline median divided by
backend median. Values below 1 mean a slowdown. Neither backend is universally
faster, so Triton remains opt-in.

The Triton backend keeps matrix products in PyTorch and fuses bias, beta,
sigmoid/softmax, and sample output. It processes multiple short categorical rows
per program. Random uniforms come from PyTorch's device RNG; matching seeds
reproduce a backend's own draws, not draws across backends. Probabilities are
checked against the original implementations and empirical frequencies against
the target distributions. See `tests/test_sampling.py`.

The default PyTorch driver removes unused zero tensors, caches the label
projection during conditioned chains, and scatters label samples directly into
the final output dtype. It retains the original scripted categorical visible
operation after benchmarks showed regressions when splitting it. Public sampling
also skips chain initialization when both input states are supplied. Zero or
negative Gibbs steps now raise a clear ValueError (the original unconditioned
sampler accessed an uninitialized `label_mag` for zero steps).

Kernel reference: [Triton's fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).

## Results (RTX A5000, 2026-09-11)

PyTorch 2.14.0+cu130, Triton 3.8.0, Python 3.12.14. All times are milliseconds.

| Model | Batch × length × hidden | Conditioned | Steps | Original | PyTorch | Triton | PyTorch speedup | Triton speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|
| binary | 64 × 128 × 128 | False | 1 | 0.385 | 0.299 | 0.252 | 1.29× | 1.53× |
| binary | 64 × 128 × 128 | False | 10 | 3.394 | 2.828 | 2.290 | 1.20× | 1.48× |
| binary | 64 × 128 × 128 | True | 1 | 0.166 | 0.129 | 0.168 | 1.28× | 0.99× |
| binary | 64 × 128 × 128 | True | 10 | 1.203 | 0.993 | 1.375 | 1.21× | 0.87× |
| binary | 256 × 256 × 256 | False | 1 | 0.391 | 0.305 | 0.259 | 1.28× | 1.51× |
| binary | 256 × 256 × 256 | False | 10 | 3.423 | 2.784 | 2.267 | 1.23× | 1.51× |
| binary | 256 × 256 × 256 | True | 1 | 0.179 | 0.140 | 0.178 | 1.27× | 1.00× |
| binary | 256 × 256 × 256 | True | 10 | 1.273 | 1.055 | 1.419 | 1.21× | 0.90× |
| binary | 1024 × 512 × 256 | False | 1 | 0.430 | 0.334 | 0.262 | 1.29× | 1.64× |
| binary | 1024 × 512 × 256 | False | 10 | 3.917 | 3.194 | 2.407 | 1.23× | 1.63× |
| binary | 1024 × 512 × 256 | True | 1 | 0.200 | 0.162 | 0.180 | 1.23× | 1.11× |
| binary | 1024 × 512 × 256 | True | 10 | 1.567 | 1.349 | 1.511 | 1.16× | 1.04× |
| categorical | 64 × 128 × 128 | False | 1 | 0.580 | 0.506 | 0.276 | 1.14× | 2.10× |
| categorical | 64 × 128 × 128 | False | 10 | 5.622 | 5.105 | 2.534 | 1.10× | 2.22× |
| categorical | 64 × 128 × 128 | True | 1 | 0.376 | 0.345 | 0.196 | 1.09× | 1.92× |
| categorical | 64 × 128 × 128 | True | 10 | 3.451 | 3.295 | 1.658 | 1.05× | 2.08× |
| categorical | 256 × 256 × 256 | False | 1 | 0.645 | 0.569 | 0.306 | 1.13× | 2.11× |
| categorical | 256 × 256 × 256 | False | 10 | 5.683 | 5.164 | 2.622 | 1.10× | 2.17× |
| categorical | 256 × 256 × 256 | True | 1 | 0.441 | 0.409 | 0.228 | 1.08× | 1.94× |
| categorical | 256 × 256 × 256 | True | 10 | 4.303 | 4.041 | 2.006 | 1.06× | 2.14× |
| categorical | 1024 × 512 × 256 | False | 1 | 2.356 | 2.131 | 1.045 | 1.11× | 2.25× |
| categorical | 1024 × 512 × 256 | False | 10 | 26.056 | 24.900 | 12.062 | 1.05× | 2.16× |
| categorical | 1024 × 512 × 256 | True | 1 | 2.166 | 2.234 | 1.185 | 0.97× | 1.83× |
| categorical | 1024 × 512 × 256 | True | 10 | 24.935 | 24.502 | 12.310 | 1.02× | 2.03× |
