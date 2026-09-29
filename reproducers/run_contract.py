"""Fail-closed fresh-run contracts, independent of run-supplied baseline paths."""
from pathlib import Path
import json
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selection_for(selected, registry):
    """Recognize only the documented, complete matrices or one canonical row."""
    lookup = {r['case_id']: r for r in registry}
    ids = [r['case_id'] for r in selected]
    require(bool(ids) and len(ids) == len(set(ids)), 'Empty or duplicate case selection')
    require(all(r == lookup.get(r['case_id']) for r in selected), 'Altered or unknown registry entry')
    if len(ids) == 1:
        return {'kind': 'single', 'case_id': ids[0]}
    for kind, rows in (
        ('all', registry),
        ('smoke', [r for r in registry if r['n'] == 8 or 'reference' in r['method']]),
        ('netbounds-and-references', [r for r in registry if not r['method'].startswith('partial-crown')]),
    ):
        if ids == [r['case_id'] for r in rows]:
            return {'kind': kind}
    raise ValueError('Selection is not a complete documented matrix')


def selected_from_contract(selection, registry):
    kind = selection.get('kind')
    if kind == 'single':
        rows = [r for r in registry if r['case_id'] == selection.get('case_id')]
    elif kind == 'all':
        rows = registry
    elif kind == 'smoke':
        rows = [r for r in registry if r['n'] == 8 or 'reference' in r['method']]
    elif kind == 'netbounds-and-references':
        rows = [r for r in registry if not r['method'].startswith('partial-crown')]
    else:
        raise ValueError('Unknown matrix contract')
    require(rows and selection_for(rows, registry) == selection, 'Invalid matrix contract')
    return rows


def artifact_paths(selected):
    names = ['manifest.json']
    for case in selected:
        case_id = case['case_id']
        names += [f'execution/{case_id}.json', f'{case_id}.log', f'rows/{case_id}/result.json']
        if case['method'] == 'sup-reference':
            names.append(f'rows/{case_id}/reference.npz')
        else:
            names.append(f'rows/{case_id}/diagnostics.json')
            if case['method'] != 'gauss-reference':
                names.append(f'rows/{case_id}/cells.npz')
    return names


def validate_run(output):
    from .heat_reproduce import ROOT, CHECKPOINT_SHA, cases, digest, sources
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    complete = json.loads((output / 'complete.json').read_text())
    registry = cases()
    selected = selected_from_contract(manifest.get('selection', {}), registry)
    # All comparison paths come from the release registry, never from the run.
    require(manifest.get('cases') == selected, 'Run cases differ from canonical matrix/registry')
    require(type(complete.get('case_count')) is int and complete['case_count'] == len(selected), 'Incomplete run')
    require(complete.get('sources_unchanged') is True, 'Source integrity did not pass')
    require(manifest.get('checkpoint_sha256') == CHECKPOINT_SHA, 'Wrong checkpoint provenance')
    require(manifest.get('versions') == dict(torch='2.7.1+cpu', numpy='2.2.6', scipy='1.15.3'), 'Wrong numerical runtime')
    require(manifest.get('threads') == 1 and manifest.get('independent_rows') is True, 'Wrong execution contract')
    inventory = sources()
    require(manifest.get('source_sha256') == inventory, 'Source inventory differs from this release')
    actual_sources = {str(p.relative_to(output / 'sources')) for p in (output / 'sources').rglob('*') if p.is_file()}
    require(actual_sources == set(inventory), 'Missing or unexpected frozen source files')
    for relative, sha in inventory.items():
        require(digest(output / 'sources' / relative) == sha, f'Changed frozen source: {relative}')
    expected_artifacts = artifact_paths(selected)
    hashes = complete.get('artifact_sha256', {})
    require(set(hashes) == set(expected_artifacts), 'Missing or unexpected sealed run artifacts')
    for relative in expected_artifacts:
        require(digest(output / relative) == hashes[relative], f'Fresh artifact hash differs: {relative}')
    require({p.name for p in (output / 'rows').iterdir()} == {r['case_id'] for r in selected}, 'Incomplete/unexpected row directories')
    pids = set()
    summary = dict(point_evaluations=0, strict_violating_points=0, tolerance_violating_points=0,
                   native_audit_failures=0, own_reference_outside=0)
    for case in selected:
        case_id = case['case_id']
        row = output / 'rows' / case_id
        execution = json.loads((output / 'execution' / (case_id + '.json')).read_text())
        require(execution.get('case_id') == case_id and execution.get('returncode') == 0, 'Missing/failed worker execution')
        command = execution.get('command', [])
        require(len(command) == 10 and command[1:8] == ['-B', '-W', 'ignore', '-m', 'reproducers.heat_reproduce', '_worker', case_id]
                and command[8] == '--campaign', 'Wrong recorded worker command')
        result = json.loads((row / 'result.json').read_text())
        pid = result.get('pid')
        require(type(pid) is int and pid > 0 and pid not in pids, 'Workers must have distinct recorded process identities')
        pids.add(pid)
        sha = result.get('checkpoint_sha256', result.get('checkpoint', {}).get('sha256'))
        require(sha == CHECKPOINT_SHA, 'Wrong row checkpoint identity')
        if 'checkpoint' in result:
            meta = result['checkpoint']
            require(meta.get('storage_dtype') == 'torch.float32' and meta.get('execution_dtype') == 'torch.float64'
                    and meta.get('device') == 'cpu' and meta.get('frozen') is True, 'Wrong checkpoint execution policy')
        coord = None if case['target'] == 'F' else int(case['target'][-1])
        require(result.get('coordinate') == coord, 'Wrong row coordinate')
        # Baseline integrity is verified independently, not against run assertions.
        for rel, sha in case['artifacts'].items():
            require(digest(ROOT / rel) == sha, f'Published evidence changed: {rel}')
        if case['method'] == 'sup-reference':
            require(result.get('grid_nodes_per_axis') == 257 and result.get('point_evaluations') == 257 ** 2
                    and result.get('is_upper_bound') is False, 'Wrong supremum reference contract')
            continue
        require(result.get('case_id') == case_id and result.get('n') == case['n']
                and result.get('method') == case['method'] and result.get('target', case['target']) == case['target'], 'Wrong row case contract')
        diag = json.loads((row / 'diagnostics.json').read_text())
        require(diag.get('affects_bound') is False, 'Diagnostics must not modify bounds')
        if case['method'] == 'gauss-reference':
            ref = result['reference']
            require(result.get('is_bound') is False and ref.get('degree') == case['n']
                    and ref.get('half_degree') == case['n'] // 2, 'Wrong Gauss reference contract')
            require(ref.get('self_check_abs_difference') == abs(ref['value'] - ref['half_value']), 'Wrong Gauss self-check')
            continue
        require(result.get('cells') == case['n'] ** 2 and result.get('epsilon') == [.5 / case['n']] * 2
                and result.get('cpu_ceiling') is None and result.get('stop_reason') == 'complete_uniform_grid', 'Wrong bound geometry/policy')
        for field in ('bounding_cpu_seconds', 'bounding_wall_seconds'):
            require(math.isfinite(result[field]) and result[field] >= 0, 'Invalid fresh timing')
        require(diag.get('numerical_artifacts_unchanged') is True, 'Worker numerical-artifact integrity failed')
        expected = {name: digest(row / name) for name in ('result.json', 'cells.npz')}
        require(diag.get('numerical_artifact_sha256') == expected, 'Worker artifact hashes differ')
        require(diag.get('cells_checked') == case['n'] ** 2 and diag.get('points_per_cell') == 9
                and diag.get('point_evaluations') == 9 * case['n'] ** 2, 'Incomplete point diagnostics')
        for field in ('point_evaluations', 'strict_violating_points', 'tolerance_violating_points'):
            require(type(diag.get(field)) is int and diag[field] >= 0, 'Invalid diagnostic count')
            summary[field] += diag[field]
        if case['method'] == 'partial-crown':
            replay = diag.get('native_replay', {})
            require(type(replay.get('failed_lines')) is int and replay['failed_lines'] >= 0
                    and replay.get('scope') == 'selected cells only', 'Missing native replay diagnostics')
            summary['native_audit_failures'] += replay['failed_lines']
        if case['method'] == 'partial-crown-affine-l2':
            audit = diag.get('native_envelope_audits', {})
            require(audit.get('calls') == case['n'] ** 2 and type(audit.get('failed_calls')) is int
                    and audit['failed_calls'] >= 0, 'Incomplete native envelope diagnostics')
            summary['native_audit_failures'] += audit['failed_calls']
            consistency = diag.get('affine_concretization', {})
            require(consistency.get('checked') == case['n'] ** 2
                    and 0 <= consistency.get('max_gap', math.inf) <= 1e-9, 'Incomplete affine reconstruction checks')
        if case['method'].endswith('-l2'):
            own = diag.get('own_quadrature', {})
            require(own.get('degree') == 64 and own.get('half_degree') == 32
                    and math.isfinite(own.get('value', math.inf)), 'Missing own-row quadrature check')
            inside = result['lower'] <= own['value'] <= result['upper']
            require(diag.get('own_quadrature_inside') is inside, 'Incorrect own-reference diagnostic')
            summary['own_reference_outside'] += not inside
    # Observations never repair or abort native computation. This post-run gate
    # refuses to label a diagnostic failure a validated numerical reproduction.
    require(not (summary['tolerance_violating_points'] or summary['native_audit_failures'] or summary['own_reference_outside']),
            f'Fresh diagnostic failures retained; reproduction is not validated: {summary}')
    return selected, summary
