# Third-party source and attribution

The root `LICENSE` applies to this project's own contributions. It **does not relicense** either vendored upstream below. Scientific comparisons do not imply endorsement by either upstream author.

## ∂-CROWN / `fgirbal/partial_crown`

- Source: <https://github.com/fgirbal/partial_crown>
- Pinned revision: [`161377048ee92b4c09faf5ce41628168c7e01556`](https://github.com/fgirbal/partial_crown/tree/161377048ee92b4c09faf5ce41628168c7e01556).
- Package author, as declared by upstream: **Francisco Girbal Eiras**.
- License declaration: **MIT**, in the preserved [`setup.py`](vendor/partial_crown/setup.py). The pinned revision contains **no standalone license text or copyright notice**. We do not invent a copyright year or represent a license file as having been supplied upstream. The original author/license declaration and README are retained verbatim.
- The three original numerical files under [`vendor/partial_crown/pinn_verifier`](vendor/partial_crown/pinn_verifier) are **byte-for-byte upstream copies**; original SHA-256 values are in [`vendor/sources.json`](vendor/sources.json).
- `setup.py` and `requirements.txt` are provenance records, **not** this reproduction's installation recipe. Use `requirements-reproduction.txt`.

Please cite the paper requested in the authors' [original README](vendor/partial_crown/README.md):

> Francisco Eiras, Adel Bibi, Rudy R Bunel, Krishnamurthy Dj Dvijotham, Philip Torr, and M. Pawan Kumar. **Efficient Error Certification for Physics-Informed Neural Networks.** Forty-first International Conference on Machine Learning (ICML), 2024. <https://openreview.net/forum?id=5t4V7Q6lmz>.

```bibtex
@inproceedings{eiras2024efficient,
  title={Efficient Error Certification for Physics-Informed Neural Networks},
  author={Francisco Eiras and Adel Bibi and Rudy R Bunel and Krishnamurthy Dj Dvijotham and Philip Torr and M. Pawan Kumar},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=5t4V7Q6lmz}
}
```

**Separately maintained harness adaptations:** `crown_benchmark/partial.py` applies the measured benchmark's import shims, actual-input-dimension fix, NumPy `Inf` compatibility, explicit float64 coefficient factories, postponed optional-solver annotations, and noninteractive debugger traps. It records original and executed-source hashes and every patch. It does not replace relaxation formulas, tune slopes/tolerances, clip outputs, or retry failures. Optional Gurobi/ONNX APIs raise explicitly if reached; they are not fake solvers. No GPU/device port is used here.

The original supremum call chain and `AffineCrownCell` readout are retained. Piecewise squared-affine integration, uniform-grid drivers, CPU accounting and independent diagnostics are **benchmark-harness contributions**, not an authors-released ∂-CROWN L2 API. The root project license applies to those contributions, not to the original upstream code.

## NetBounds

- Source: <https://github.com/65industries/NetBounds-dev>
- Pinned revision: `7835f105a6731bb6c133fd3d8ddcc243c1e9326b`.
- License: **MIT, Copyright (c) 2026 emilhaugen**; the complete original [license](vendor/netbounds/LICENSE) is included unchanged.
- Scope: raw neural-network derivative bounds through network order four, activation bounds, and raw center derivatives. No training runner, boundary mask, PDE residual, wave-operator combination, or spatial-wrapper API is included.
- The seven numerical modules retain the upstream implementation. Unused spatial-wrapper definitions in `first_order.py` and the wave-operator tail in `fourth_order.py` are deleted; package initializers are reduced to avoid importing PDE/training modules. The exact omissions, original hashes and packaged hashes are recorded in [`vendor/sources.json`](vendor/sources.json). No retained numerical formula is rewritten.

Both upstreams are loaded from bundled, hash-verified sources, not a user cache or external Git checkout. This packaging change is distinct from the historically measured source provenance and does not turn ordinary float64 computation into outward-rounded certification.
