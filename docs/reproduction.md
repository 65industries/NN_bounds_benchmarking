# Reproduction contract

## Numerical authority

`docs/benchmarking.md` is the user-selected report. Its measured values and CPU times are preserved. `reproducers/cases.json` enumerates exactly **55 bound rows + 5 references**, with every original result/diagnostic/array file and SHA-256. Evidence JSON and NPZ files under `results/` are byte-preserved; historical absolute paths inside them are **provenance only**, not runtime dependencies.

The checkpoint is `model_weights/heat_1d_separable_L2_W128_netbounds.pt`, SHA-256 `fb1f6f1d8dc628c84f561a05b0f5c9b2ef1537d93be4fac8c427f13c722efb68`. Stored float32 values are copied exactly to float64; coordinates are `(x,t)`, and both coordinates vary over `[0,1]`.

`n` means cell centres **per coordinate**: `n²` equal boxes, centres `((j+1/2)/n,(k+1/2)/n)`, both radii `1/(2n)`. No adaptive refinement, clipping, intersection, sample repair, CPU ceiling or new relaxation is used.

## Mathematical implementation map

| Report construction | Reproduction path |
|---|---|
| Direct E2, `abs(Fii(center)) + variation` | `reproducers/heat_second_order_e_comparison.py`: `netbounds_call`; `vendor/netbounds/derivative_bounds/second_order.py` |
| Q1 supremum, affine `Fii` plus fourth-order-controlled quadratic remainder | `independent_benchmark/second_derivative.py`: `run_netbounds`, `grid_reduction`; `netbounds_benchmark/uniform_q1.py`: `RawQ1Integrator` |
| Q1 squared norm, exact affine moments and cross/remainder terms | `netbounds_benchmark/uniform_q1.py`: `q1_moment_cell`; `reproducers/heat_l2_matched_comparison.py`: `reduce_q1` |
| Raw ∂-CROWN supremum | `reproducers/heat_second_order_e_comparison.py`: `partial_call`; upstream solution → first partial → second partial chain |
| ∂-CROWN affine L2 | `independent_benchmark/crown_affine.py`: `AffineCrownCell`; `minimal_affine_l2_native.py`: `cell_integrals` |
| Independent supremum observation | Nested autograd on a 257×257 endpoint grid |
| Independent L2 observation | Tensor Gauss–Legendre degree 200, with degree 100 self-check |

For Q1 the lower square is `max(0, sum(Q) - sum(J))`, with **one global truncation**, and the upper square is `sum(Q + J + K)`. For affine ∂-CROWN the harness integrates the **pointwise** square image of the native affine interval using exact-real polygon moments. It does not move a maximum outside an integral or substitute a scalar interval reduction. The retained degeneracy-safe helper counts the `lower + upper ≡ 0` case only once.

## Source closure and changes

- `vendor/sources.json`: upstream commits, original hashes, bundled hashes and explicit scope omissions.
- `reproducers/extraction.json`: frozen measured-harness origins, source hashes and retained definitions.
- NetBounds includes seven raw-network modules, minimal initializers and its unchanged MIT license. PDE/wave and spatial-mask wrapper definitions are omitted; retained numerical expressions are unchanged.
- ∂-CROWN includes three byte-identical numerical modules, the upstream README, setup metadata and requirements as provenance. See [attribution](../THIRD_PARTY_NOTICES.md).
- Existing import/dimension/float64/noninteractive patches remain in the separate ∂-CROWN loader and are logged. No repair variant is applied.
- External-Git/HOME checks are replaced by bundled-source hash verification. A vendor directory is not falsely reported as a clean upstream Git checkout.
- The fixed-grid and integration modules retain only reached numerical functions; unrelated CLI, auto_LiRPA, adaptive-campaign and GPU-adapter imports are excluded.
- The portable coordinator is new orchestration, **not a new bound algorithm**. It freezes the closure, launches one process per selected row, records failures, and never reads baseline numerical evidence in workers.

The report differs from the local publication only in fixing references to files/functions not part of this minimal release. Numerical table rows and mathematical formulas are unchanged. The older adaptive/auto_LiRPA report is intentionally not included.

## Commands and outputs

See the [README](../README.md) for installation and full/smoke matrices. One row can be run with:

```bash
.venv/bin/python -m reproducers.heat_reproduce row \
  netbounds-q1-expansion-F00-n128 --output runs/one-row
.venv/bin/python -m reproducers.heat_reproduce check-run runs/one-row
```

`run --netbounds-and-references-only` selects all 35 NetBounds rows and five references. `smoke` selects 12 n=8 bound rows and five references. `run --acknowledge-expensive` selects all 60 published computations, not the pending n=128 ∂-CROWN rows.

Each run contains a source/checkpoint snapshot, manifest, fresh row logs, full-precision JSON, per-cell NPZ arrays, diagnostics, execution exit codes and a completion record. A failure has no successful completion record. Reusing or overwriting an output directory is rejected.

`verify` checks all published artifact hashes, model identity, exact complete-cover geometry, reduction identities, independently expanded ordered Q1 moments, every numerical table cell and relative link. `check-run` is a separate post-computation comparison: it checks native arrays and numerical reductions with predeclared `rtol=2e-12`, `atol=5e-13`, reports bitwise equality separately, and does not compare elapsed/CPU times.

## Timing and qualification

Preserved workers retain their original timing boundaries. Numerical process CPU includes geometry, bounding, requested integration and global reduction. Checkpoint loading, setup, output and independent diagnostics are excluded; the existing numerical-clock ledger excludes measured inline diagnostics. Wall time can include those diagnostics. Necessary structural checks and native bookkeeping remain charged. No fresh timing is relabelled as a historical measurement.

Passing tests and reference containment are not machine certificates. Native float64 tensor/root-finding arithmetic has no propagated outward rounding; all conclusions retain the report's qualification.
