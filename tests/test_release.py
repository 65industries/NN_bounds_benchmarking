"""Release-specific fail-closed contracts and bounded native differential replays."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from reproducers.heat_reproduce import ROOT, CHECKPOINT, cases, compare_run, launch
from reproducers.verify_release import verify, verify_table_cells
from reproducers import source_integrity
from reproducers.heat_second_order_e_comparison import partial_setup, partial_call
from crown_benchmark.models import load_checkpoint
from independent_benchmark.crown_affine import AffineCrownCell
from independent_benchmark.minimal_affine_l2_native import cell_integrals


def test_complete_publication():
    answer = verify()
    assert answer['published_computations'] == 60
    assert answer['bound_rows'] == 55 and answer['reference_rows'] == 5
    assert answer['artifacts_verified'] == 175
    assert answer['bound_cells_checked'] == 179968


def test_registry_is_exact_and_cost_gate_precedes_execution(tmp_path):
    rows = cases()
    assert len([r for r in rows if r['n'] == 8 or 'reference' in r['method']]) == 17
    assert len([r for r in rows if not r['method'].startswith('partial-crown')]) == 40
    with pytest.raises(ValueError, match='acknowledge-expensive'):
        launch(tmp_path / 'expensive', [r for r in rows if r['n'] == 64 and r['method'].startswith('partial-crown')])
    assert not (tmp_path / 'expensive').exists()


def test_vendor_tampering_rejected(tmp_path, monkeypatch):
    shutil.copytree(ROOT / 'vendor', tmp_path / 'vendor')
    monkeypatch.setattr(source_integrity, 'ROOT', tmp_path)
    source_integrity.verify_vendor('partial_crown')
    path = tmp_path / 'vendor/partial_crown/pinn_verifier/crown.py'
    path.write_bytes(path.read_bytes() + b'\n# mutation\n')
    with pytest.raises(ValueError, match='source changed'):
        source_integrity.verify_vendor('partial_crown')


def test_extra_vendor_module_rejected(tmp_path, monkeypatch):
    shutil.copytree(ROOT / 'vendor', tmp_path / 'vendor')
    monkeypatch.setattr(source_integrity, 'ROOT', tmp_path)
    (tmp_path / 'vendor/netbounds/pde.py').write_text('raise RuntimeError("must not import")\n')
    with pytest.raises(ValueError, match='Unexpected/missing'):
        source_integrity.verify_vendor('netbounds')


def test_table_mutation_rejected():
    records = {r['result_path']: (r, json.loads((ROOT / r['result_path']).read_text())) for r in cases()}
    text = (ROOT / 'docs/benchmarking.md').read_text()
    with pytest.raises(AssertionError):
        verify_table_cells(text.replace('444.512232', '444.512233'), records)


def test_no_git_or_cache_needed(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A runtime source loader must not call Git')
    monkeypatch.setattr(subprocess, 'check_output', forbidden)
    monkeypatch.setenv('NETBOUNDS_ROOT', '/nonexistent')
    monkeypatch.setenv('PARTIAL_CROWN_PATH', '/nonexistent')
    from netbounds_benchmark.adapter import load_backend
    from crown_benchmark.partial import load_authors
    root, _ = load_backend()
    authors = load_authors()
    assert root == ROOT / 'vendor/netbounds'
    assert authors.metadata['upstream_tracked_clean'] is None
    assert authors.metadata['source_verification'] == 'vendored SHA-256 allowlist'
    assert len(authors.metadata['patches']) == 70


@pytest.mark.parametrize('case', [r for r in cases() if r['method'].startswith('partial-crown')], ids=lambda r: r['case_id'])
def test_all_crown_grids_selected_cells_against_measurement(case):
    torch.set_num_threads(1)
    model = load_checkpoint(ROOT / CHECKPOINT).model
    coordinate = None if case['target'] == 'F' else int(case['target'][-1])
    row = ROOT / case['result_path']
    result = json.loads(row.read_text())
    arrays = np.load(row.parent / 'cells.npz', allow_pickle=False)
    indices = sorted({0, case['n'] ** 2 - 1, result.get('max_cell', case['n'] ** 2 // 2)})
    if case['method'] == 'partial-crown':
        authors, relaxations = partial_setup(model)
        for i in indices:
            l, u = partial_call(model, coordinate, authors, relaxations,
                               torch.from_numpy(arrays['lower'][i]), torch.from_numpy(arrays['upper'][i]))
            np.testing.assert_allclose([l, u], [arrays['lower_bound'][i], arrays['upper_bound'][i]], rtol=2e-12, atol=5e-13)
    else:
        kernel = AffineCrownCell(model, 'value' if coordinate is None else 'second-diagonal', coordinate)
        for i in indices:
            lo, hi = arrays['lower'][i].tolist(), arrays['upper'][i].tolist()
            AL, cL, AU, cU, L, U = kernel.affine_bound(lo, hi)
            raw = dict(A_lower=AL, c_lower=cL, A_upper=AU, c_upper=cU, lower_bound=L, upper_bound=U)
            for key, value in raw.items():
                np.testing.assert_allclose(value, arrays[key][i], rtol=2e-12, atol=5e-13)
            integrals = cell_integrals(raw, lo, hi)
            for key, value in integrals.items():
                np.testing.assert_allclose(value, arrays[key][i], rtol=2e-12, atol=5e-13)
            assert not kernel.drain_audit()['failed']
    arrays.close()


def test_all_published_affine_integrals_against_both_retained_integrators():
    from independent_benchmark.affine_l2 import cell_square_integral
    count = 0
    for case in cases():
        if case['method'] != 'partial-crown-affine-l2':
            continue
        with np.load((ROOT / case['result_path']).parent / 'cells.npz', allow_pickle=False) as a:
            for i in range(case['n'] ** 2):
                raw = {k: a[k][i].tolist() for k in ('A_lower','A_upper','c_lower','c_upper','lower_bound','upper_bound')}
                new = cell_integrals(raw, a['lower'][i], a['upper'][i])
                old = cell_square_integral(raw['A_lower'], raw['c_lower'], raw['A_upper'], raw['c_upper'],
                                           a['lower'][i], a['upper'][i])
                np.testing.assert_allclose([new['lower_square'], new['upper_square']],
                                           [a['lower_square'][i], a['upper_square'][i]], rtol=0, atol=0)
                np.testing.assert_allclose(old, [new['lower_square'], new['upper_square']], rtol=2e-10, atol=1e-13)
                count += 1
    assert count == 16320
