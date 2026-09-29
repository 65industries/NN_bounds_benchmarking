"""Read-only release audit: source closure, every table cell and saved reduction."""
from pathlib import Path
import json
import math
import re
import numpy as np
from .heat_reproduce import ROOT, CHECKPOINT, CHECKPOINT_SHA, cases, digest
from .source_integrity import verify_vendor


def verify_table_cells(text, records):
    reached = set()
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        links = re.findall(r'\]\(([^)]+/result\.json)\)', line)
        if not links:
            continue
        cells = [x.strip() for x in line.strip('|').split('|')]
        for link in links:
            rel = str((ROOT / 'docs' / link).resolve().relative_to(ROOT))
            case, result = records[rel]
            reached.add(case['case_id'])
            method = case['method']
            if method == 'sup-reference':
                assert cells[1] == f"{result['observed_maximum']:.12g}"
                assert cells[3].replace(',', '') == str(result['point_evaluations'])
                assert cells[5] == f"{result['reference_cpu_seconds']:.6f}"
            elif method == 'gauss-reference':
                assert cells[1] == f"{result['reference']['value']:.12g}"
                assert cells[2] == f"{result['reference']['self_check_abs_difference']:.2e}"
            else:
                assert cells[0] == str(case['n']) and int(cells[1].replace(',', '')) == case['n'] ** 2
                if method in ('netbounds-e2', 'netbounds-q1-expansion', 'partial-crown'):
                    index = {'netbounds-e2': 2, 'netbounds-q1-expansion': 4, 'partial-crown': 6}[method]
                    assert cells[index] == f"{result['bound']:.12g}", (case['case_id'], cells)
                    assert cells[index + 1] == f"{result['bounding_cpu_seconds']:.6f}"
                else:
                    index = 2 if method == 'netbounds-q1-l2' else 5
                    assert cells[index] == f"{result['lower']:.12g}"
                    assert cells[index + 1] == f"{result['upper']:.12g}"
                    assert cells[index + 2] == f"{result['bounding_cpu_seconds']:.6f}"
    assert reached == {case['case_id'] for case, _ in records.values()}


def verify_reduction(case, result):
    directory = (ROOT / case['result_path']).parent
    if case['method'] == 'gauss-reference':
        ref = result['reference']
        assert ref['degree'] == 200 and ref['half_degree'] == 100
        assert ref['self_check_abs_difference'] == abs(ref['value'] - ref['half_value'])
        return 0
    if case['method'] == 'sup-reference':
        with np.load(directory / 'reference.npz', allow_pickle=False) as a:
            assert a['points'].shape == (257 * 257, 2)
            axis = np.linspace(0., 1., 257)
            mesh = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
            np.testing.assert_array_equal(mesh, a['points'])
            v = a['values'].reshape(-1)
            assert float(abs(v).max()) == result['observed_maximum']
            assert a['points'][abs(v).argmax()].tolist() == result['witness']
        return 0
    n = case['n']
    assert result['cells'] == n * n and result['epsilon'] == [.5 / n, .5 / n]
    assert result['cpu_ceiling'] is None and result['stop_reason'] == 'complete_uniform_grid'
    with np.load(directory / 'cells.npz', allow_pickle=False) as a:
        ij = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing='ij'), axis=-1).reshape(-1, 2)
        for key, expected in [('centers', (ij + .5) / n), ('lower', ij / n), ('upper', (ij + 1) / n)]:
            np.testing.assert_array_equal(a[key], expected)
        for key in a.files:
            assert a[key].dtype == np.float64 and np.isfinite(a[key]).all()
            assert len(a[key]) == n * n
        method = case['method']
        if method in ('netbounds-e2', 'partial-crown', 'netbounds-q1-expansion'):
            assert float(a['absolute_bound'].max()) == result['bound']
            if method == 'netbounds-e2':
                np.testing.assert_array_equal(a['absolute_bound'], abs(a['base']) + a['variation'])
            elif method == 'partial-crown':
                np.testing.assert_array_equal(a['absolute_bound'], np.maximum(-a['lower_bound'], a['upper_bound']))
            else:
                eps = np.array(result['epsilon'])
                rad = (abs(a['gradient']) * eps).sum(1) + .5 * np.einsum('i,nij,j->n', eps, a['hessian_sup'], eps)
                np.testing.assert_array_equal(a['absolute_bound'], abs(a['base']) + rad)
            assert float(a['lower_bound'].min()) == result['signed_lower']
            assert float(a['upper_bound'].max()) == result['signed_upper']
        elif method == 'netbounds-q1-l2':
            # Independently expand the ordered double and quadruple sums.
            h, volume = .5 / n, 1 / (n * n)
            def moment(indices):
                return volume * math.prod(h ** indices.count(k) / (indices.count(k) + 1) for k in (0, 1))
            H, b, base = a['hessian_sup'], abs(a['gradient']), abs(a['base'])
            Q = volume * (base ** 2 + (b ** 2).sum(1) * h * h / 3)
            J = np.zeros(n * n); K = np.zeros(n * n)
            for i in (0, 1):
                for j in (0, 1):
                    J += base * H[:, i, j] * moment([i, j])
                    for k in (0, 1):
                        J += b[:, k] * H[:, i, j] * moment([i, j, k])
                        for l in (0, 1):
                            K += .25 * H[:, i, j] * H[:, k, l] * moment([i, j, k, l])
            for key, values in [('affine_square', Q), ('cross_error', J), ('remainder_square', K)]:
                np.testing.assert_allclose(a[key], values, rtol=2e-13, atol=1e-15)
            sums = {k: math.fsum(a[k].tolist()) for k in result['moment_sums']}
            assert sums == result['moment_sums']
            assert result['lower_square'] == max(0., sums['affine_square'] - sums['cross_error'])
            assert result['upper_square'] == sums['upper_square']
        else:
            assert method == 'partial-crown-affine-l2'
            assert result['lower_square'] == math.fsum(a['lower_square'].tolist())
            assert result['upper_square'] == math.fsum(a['upper_square'].tolist())
        if method.endswith('-l2'):
            assert result['lower'] == math.sqrt(result['lower_square'])
            assert result['upper'] == math.sqrt(result['upper_square'])
        diag = json.loads((directory / 'diagnostics.json').read_text())
        assert diag['point_evaluations'] == 9 * n * n
        assert diag['tolerance_violating_points'] == 0
    return n * n


def verify():
    definitions = json.loads((ROOT / 'reproducers/definition-provenance.json').read_text())
    for item in definitions:
        assert digest(ROOT / item['path']) == item['packaged_sha256'], item['path']
    for component in ('netbounds', 'partial_crown'):
        verify_vendor(component)
    assert digest(ROOT / CHECKPOINT) == CHECKPOINT_SHA
    rows = cases()
    records = {}
    artifact_count = bound_cells = 0
    for case in rows:
        for path, expected in case['artifacts'].items():
            assert digest(ROOT / path) == expected, path
            artifact_count += 1
        result = json.loads((ROOT / case['result_path']).read_text())
        actual_sha = result.get('checkpoint_sha256', result.get('checkpoint', {}).get('sha256'))
        assert actual_sha == CHECKPOINT_SHA
        records[case['result_path']] = case, result
        bound_cells += verify_reduction(case, result)
    doc = ROOT / 'docs/benchmarking.md'
    text = doc.read_text()
    verify_table_cells(text, records)
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if link.startswith('#'):
            assert f'id="{link[1:]}"' in text
        elif not link.startswith(('https:', 'http:')):
            assert (doc.parent / link.split('#')[0]).exists(), link
    # Every reference is diagnostic, but also inside each corresponding bound.
    for case, result in records.values():
        if 'lower' in result:
            reference = next(r['reference']['value'] for c, r in records.values()
                             if c['method'] == 'gauss-reference' and c['target'] == case['target'])
            assert result['lower'] <= reference <= result['upper']
    return dict(published_computations=len(rows), bound_rows=55, reference_rows=5,
                artifacts_verified=artifact_count, bound_cells_checked=bound_cells,
                numerical_tables_verified=True, local_links_verified=True,
                machine_certified=False)
