# Release verification

This records actual checks of the portable source-archive candidate; it is not a claim that every expensive CROWN grid was rerun for packaging.

## Clean-room execution

A Git-tree source archive (no `.git` directory) was tested in a fresh Python 3.11 environment installed from `requirements-reproduction.txt`. Bubblewrap hid the original research checkouts, the external upstream-source cache and the old benchmark virtualenv, with an isolated network namespace and an empty HOME. The fresh environment had no auto_LiRPA installation.

Results:

| Check | Result |
|---|---|
| Published evidence/table audit | 60 computations: 55 bounds, 5 references; 175 artifacts verified |
| Complete saved-cell geometry/reductions | 179,968 bound cells checked |
| Regression suite | **94 passed**, including fresh-run authority/provenance mutation tests |
| Fresh-process smoke | **17 computations; 140 saved arrays bitwise identical** to measured evidence |
| Full NetBounds matrix plus references | **40 computations; 369 saved arrays bitwise identical** to measured evidence |
| All published affine integrals | 16,320 saved cells replayed through the existing corrected integrator; compared with the older independent polygon implementation |
| ∂-CROWN selected native-cell replays | All 20 published CROWN rows covered, including each published grid through n=64 |
| Source-extraction audit | 89 retained harness definitions unchanged; only loader-origin/integrity functions adapted |
| Report preservation | All 54 Markdown table lines unchanged |
| Diff hygiene | Passed; upstream whitespace deliberately preserved via vendor attributes |

The repeated smoke/reference cases are not additional independent scientific measurements for the report. Recomputed CPU times were **not** substituted for the published ones.

## Verification-gate hardening

Independent review found that an earlier `check-run` trusted run-supplied baseline paths and omitted fresh provenance checks. Mutation tests first reproduced these failures. The gate now resolves baselines exclusively from the canonical registry; rejects empty, duplicate, unknown or incomplete matrices; verifies frozen source hashes, successful worker execution, checkpoint/dtype/reference policies and sealed fresh artifacts; and refuses to call diagnostic failures a validated reproduction. The numerical workers and published measurements are unchanged.

## Limits

- The full n=16/32/64 CROWN matrices were not rerun solely for this packaging check. Their original saved arrays/results are retained, and selected native cells and all affine integration reductions were replayed.
- The independently running n=128 CROWN campaign is outside this release snapshot.
- Finite diagnostics and bitwise replay do not provide outward-rounded machine certification.
- GitHub CI runs the evidence audit, regression suite, smoke and complete NetBounds/reference matrix on a separate runner; its status must be read from the PR, not inferred from these local results.
