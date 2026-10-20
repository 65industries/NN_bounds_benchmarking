# Heat1 PDE residual: matched uniform-grid comparison

Informal original-forward Gauss64 reference: **J = 4.001960432102522**, **L2 = 2.00049004799**; Gauss32/64 difference 0. This is not a bound.

Target: $r=F_t-0.1F_{xx}$ for the **exact PDEbenchmarking Heat1 checkpoint**, coordinates $(x,t)$, on $[0,1]^2$. No boundary mask, training, or random model initialization is used. The original raw-NN report uses a different trained checkpoint and is unchanged.

Checkpoint SHA256: `514531c1d38d138b053de85842e4c4a92de0bc82a629d448437ad1c8b5028ba6`. Architecture `2 → 128 → 128 → 1`, tanh, storage/execution/accumulation float64, CPU, one effective numerical thread. Fresh numerical producer: `41c0ef1f4582357602236c02b51721d008826931`.

## Measured uniform bounds

Each n means n cells per coordinate (n² cells). Norm endpoints are square roots AFTER globally summing squared-integral endpoints. Widths use full-precision values, not rounded table cells.

| n | Cells | Method | L2 lower | L2 upper | L2 width | Wall (s) |
|---:|---:|---|---:|---:|---:|---:|
| 8 | 64 | NetBounds Q0 (retained) | 0.478706515 | 38.5065941 | 38.0279 | 0.039544 |
| 8 | 64 | NetBounds Q1 (retained) | 0.445430014 | 57.4373674 | 56.9919 | 0.065022 |
| 8 | 64 | AdaQuad dense IBP (retained) | 0 | 7.13242764 | 7.13243 | 1.446259 |
| 8 | 64 | AdaQuad dense RecurJac (retained) | 0 | 5.24191522 | 5.24192 | 1.435580 |
| 8 | 64 | ∂-CROWN affine residual (fresh) | 1.7937134 | 2.21452311 | 0.42081 | 10.617555 |
| 16 | 256 | NetBounds Q0 (retained) | 1.67594626 | 2.442159 | 0.766213 | 0.105988 |
| 16 | 256 | NetBounds Q1 (retained) | 1.68720884 | 2.52403966 | 0.836831 | 0.185255 |
| 16 | 256 | AdaQuad dense IBP (retained) | 0.0171528779 | 4.6639745 | 4.64682 | 5.609040 |
| 16 | 256 | AdaQuad dense RecurJac (retained) | 0.616011548 | 3.39478975 | 2.77878 | 5.689774 |
| 16 | 256 | ∂-CROWN affine residual (fresh) | 1.94787204 | 2.05335345 | 0.105481 | 38.774024 |
| 32 | 1024 | NetBounds Q0 (retained) | 1.96372321 | 2.0369321 | 0.0732089 | 0.426925 |
| 32 | 1024 | NetBounds Q1 (retained) | 1.99167474 | 2.00943721 | 0.0177625 | 0.773548 |
| 32 | 1024 | AdaQuad dense IBP (retained) | 0.704021816 | 3.34497616 | 2.64095 | 22.622892 |
| 32 | 1024 | AdaQuad dense RecurJac (retained) | 1.35706473 | 2.64474219 | 1.28768 | 22.817776 |
| 32 | 1024 | ∂-CROWN affine residual (fresh) | 1.98726721 | 2.01363013 | 0.0263629 | 155.825930 |
| 64 | 4096 | NetBounds Q0 (retained) | 1.98769377 | 2.01322641 | 0.0255326 | 1.816627 |
| 64 | 4096 | NetBounds Q1 (retained) | 2.00016266 | 2.00083574 | 0.000673079 | 2.743958 |
| 64 | 4096 | AdaQuad dense IBP (retained) | 1.33469625 | 2.67337371 | 1.33868 | 88.977386 |
| 64 | 4096 | AdaQuad dense RecurJac (retained) | 1.69335169 | 2.30810279 | 0.614751 | 89.026701 |
| 64 | 4096 | ∂-CROWN affine residual (fresh) | 1.99716196 | 2.00378454 | 0.00662258 | 608.010933 |

## Construction and timing

The unmodified pinned ∂-CROWN chain computes a fresh value solution plus Fx/Fxx and Ft on every cell. Native input-affine derivative pairs are combined **before concretization**: lower = Ft_lower − 0.1 Fxx_upper; upper = Ft_upper − 0.1 Fxx_lower. The harness integrates the exact piecewise-quadratic square sandwich by polygon clipping and closed triangle moments; it does not numerically quadrature across switching kinks. This is a harness reduction of authors-native derivative bounds, not a released end-to-end heat-verifier mode.

Fresh method clocks include native setup, grid construction, full propagation, coefficient readout, signed PDE composition, affine-square moments and final sum/square roots. They exclude checkpoint loading, serialization and post-freeze diagnostics. Both process CPU and elapsed wall are saved. Exact-child wait4 CPU and parent elapsed wall include startup/loading/diagnostics/I/O/exit. Minimal progress/call bookkeeping is charged.

NetBounds/AdaQuad numbers and wall clocks are retained byte-for-byte from their original PDEbenchmarking run, not newly timed or converted from CPU. Original metadata and per-method producing URLs are in inputs/retained_heat1. They use Torch 2.12.0 / NumPy 2.4.6; fresh ∂-CROWN uses its validated Torch 2.7.1 / NumPy 2.2.6 / SciPy 1.15.3 stack. These are single measurements with different preparation/software boundaries, not repeated trials or a controlled engine-only speed ratio.

## Validation and evidence

- Integrity audit: PASS; 5440 full native cells, 5 distinct independently loaded workers (including reference).
- Post-freeze numerical diagnostics: PASS. Checks cover nine points per cell for Ft, Fxx and the combined residual; selected native affine/primitive replays; full-cell alternate polygon integration; own-row Gauss64 containment.
- Scalar-lift values in diagnostics reuse these same native outputs and have no independent propagation timing; they are ablations, not extra benchmark rows.
- Ordinary float64 bounds are **not outward-rounded machine certificates**. Point checks and Gauss agreement are observations, not proofs.
- Only uniform n=8,16,32,64 are authorized here. No n=128, adaptive rows, GPU, new AdaQuad/NetBounds sweep, or legacy-row replacement.

Raw new campaign: [manifest.json](measurements/heat1-uniform-v1/manifest.json); numerical arrays, native call journals, exact child receipts, post-freeze diagnostics and frozen numerical source are under that directory.
Retained-input byte hashes and source revision: [inputs/provenance.json](inputs/provenance.json).
