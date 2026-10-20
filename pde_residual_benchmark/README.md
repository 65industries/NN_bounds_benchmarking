# Heat1 PDE-residual uniform comparison

This separate extension uses the **PDEbenchmarking Heat1 trained checkpoint** and the plain-network heat residual

$$r(x,t)=F_t(x,t)-0.1F_{xx}(x,t),\qquad (x,t)\in[0,1]^2.$$

The original [`docs/benchmarking.md`](../docs/benchmarking.md) is a pure-derivative experiment on a **different trained checkpoint**. Its 60 computations, registry, kernels and saved artifacts remain unchanged.

## Run

From the checkout root, using Python 3.11:

```bash
uv pip install --python /path/to/python -r pde_residual_benchmark/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

# Source/checkpoint changes must be committed; outputs must not already exist.
/path/to/python -B -m pde_residual_benchmark smoke --output runs/heat1-pde-smoke
/path/to/python -B -m pde_residual_benchmark.report runs/heat1-pde-smoke

# Full authorized uniform n=8,16,32,64, plus independent Gauss32/64 reference.
/path/to/python -B -m pde_residual_benchmark run \
  --output pde_residual_benchmark/measurements/heat1-uniform-v1
/path/to/python -B -m pde_residual_benchmark.report \
  pde_residual_benchmark/measurements/heat1-uniform-v1 \
  --output-prefix pde_residual_benchmark/heat1_uniform_comparison
```

The complete matrix contains **four fresh bound rows (5,440 native cells)** and **one separately computed informal reference row**. Each row has a fresh interpreter, exact selected checkpoint load, frozen source closure, read-path audit and exact-child CPU/wall receipt. No GPU or adaptive runs. A failure preserves its native call/box/stage journal; there is no automatic retry, endpoint repair, or substitution from a different row.

∂-CROWN's original value/first/second derivative propagation and relaxation defaults are unchanged. Its native input-affine Ft/Fxx bounds are composed with the signed PDE coefficient before integrating the piecewise-quadratic square sandwich. This operator composition/integration is benchmark-harness code, not a new authors' bound engine. Effective Torch/BLAS thread counts are checked at one. Stored, execution and accumulation precision are float64.

Method CPU/wall include backend setup, geometry, full native derivative chains, affine readout/composition, exact moments and global reduction. Model loading, serialization, independent references and post-freeze diagnostics are outside these clocks. Full child clocks retain those costs separately.

NetBounds Q0/Q1 and AdaQuad dense IBP/RecurJac comparisons use **retained original PDEbenchmarking numbers and wall times**, with original producing-source metadata. They are not rerun or retimed. The fresh ∂-CROWN software stack differs and the clocks have documented preparation boundaries; do not interpret the report as repeated timing statistics or a pure engine-only speed ratio. Retained input hashes and source are in [`inputs/provenance.json`](inputs/provenance.json).

All-cell affine reconstruction, geometry, original-call journals, global reductions and alternate integration are audited. Nine points/cell, own-row Gauss64 and selected primitive-envelope/native replays are post-freeze observations, never repairs. Ordinary float64 results are **not outward-rounded machine certificates**.

## Tests

```bash
/path/to/python -B -m pytest -q -p no:cacheprovider tests pde_residual_benchmark/test_heat1.py pde_residual_benchmark/test_review_fixes.py
/path/to/python -B -m reproducers.heat_reproduce verify
```

Third-party provenance and licenses remain in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The copied NetBounds checkpoint retains its original provenance. No training or randomly initialized network is benchmarked here.
