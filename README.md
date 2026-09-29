# NN bounds benchmarking — raw heat-network reproduction

[**Read the matched-grid benchmark report**](docs/benchmarking.md) · [Reproduction details](docs/reproduction.md) · [Third-party attribution](THIRD_PARTY_NOTICES.md)

This checkout reproduces the report's **55 bound computations and 5 independent reference computations** for the saved `2 → 128 → 128 → 1` tanh network on `[0,1]²`:

- Direct second-order NetBounds supremum bounds for `Fxx` and `Ftt`.
- NetBounds affine Q1 supremum and L2 constructions, using raw network derivatives **through fourth order**.
- Released **∂-CROWN** second-derivative bounds and native-affine-pair L2 integration by the benchmark harness.
- NetBounds grids `n=8,16,32,64,128`; published ∂-CROWN grids `n=8,16,32,64`.

**Pure neural-network computations only:** no training, PDE residuals, boundary masks, or solver calls. The heat label identifies the checkpoint's origin, not a PDE computation performed here. The earlier model files already present on `main` are retained, but this reproduction selects only the named heat checkpoint.

The computations use ordinary CPU float64 and are **not outward-rounded machine certificates**. References and point checks are diagnostics. CPU times in the report are historical measurements; fresh runs measure fresh times, which depend on hardware and load.

## Quick start (Linux CPU, Python 3.11)

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduction.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

# Audit every published row, source pin, geometry, reduction and table value.
.venv/bin/python -m reproducers.heat_reproduce verify

# Recompute all methods/targets at n=8, plus the five references.
.venv/bin/python -m reproducers.heat_reproduce smoke --output runs/smoke
.venv/bin/python -m reproducers.heat_reproduce check-run runs/smoke

# Run the regression suite (includes selected native cell replays).
.venv/bin/python -m pytest -q -p no:cacheprovider tests
```

This is an intentionally **checkout-rooted/source-archive reproduction**, not an installable wheel. Do not run `pip install .` or the provenance-only upstream `setup.py`. No separate NetBounds or ∂-CROWN checkout, Git history, user cache, Gurobi, ONNX, auto_LiRPA, or GPU is needed. Network access is needed only to install the pinned Python packages.

## Full table recomputation

```bash
.venv/bin/python -m reproducers.heat_reproduce list
.venv/bin/python -m reproducers.heat_reproduce run \
  --output runs/full --acknowledge-expensive
.venv/bin/python -m reproducers.heat_reproduce check-run runs/full
```

The full run includes expensive ∂-CROWN grids and may take **tens of minutes or longer**. Run it in a durable terminal/session. Every output directory must be new. Failures retain their logs; there is no automatic retry or silent reuse of a completed row. Each row gets a fresh interpreter, checkpoint load and frozen source snapshot.

For a faster NetBounds-only matrix with all references:

```bash
.venv/bin/python -m reproducers.heat_reproduce run \
  --netbounds-and-references-only --output runs/netbounds
.venv/bin/python -m reproducers.heat_reproduce check-run runs/netbounds
```

The published evidence in `results/` is immutable verification data, never an input to a numerical worker. A separate `check-run` compares full-precision numerical outputs and every saved array, **not CPU-time equality**. Pending ∂-CROWN `n=128` values are not fabricated or included in this release matrix.

## Attribution and licenses

∂-CROWN is the authors' implementation from [fgirbal/partial_crown](https://github.com/fgirbal/partial_crown), associated with Eiras et al., **Efficient Error Certification for Physics-Informed Neural Networks**, ICML 2024. Please use the full citation and read the license-provenance disclosure in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Its pinned `setup.py` declares MIT, but that revision contains no standalone license file.

The bundled NetBounds source retains its original MIT license. The repository's own contributions remain **CC BY-NC 4.0**, with explicit third-party exclusions in [NOTICE](NOTICE).
