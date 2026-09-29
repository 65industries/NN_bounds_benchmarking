"""Portable reproduction of the measured rows in docs/benchmarking.md.

Workers use the retained numerical functions, one interpreter/checkpoint load
per row. Published measurements are verification data, never numerical inputs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'model_weights/heat_1d_separable_L2_W128_netbounds.pt'
CHECKPOINT_SHA = 'fb1f6f1d8dc628c84f561a05b0f5c9b2ef1537d93be4fac8c427f13c722efb68'
# Fixed before replay; allow cross-machine float64 roundoff, never CPU equality.
REPLAY_RTOL = 2e-12
REPLAY_ATOL = 5e-13


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def cases():
    rows = json.loads((ROOT / 'reproducers/cases.json').read_text())
    expected = set()
    for method, targets, grids in (
        ('netbounds-e2', ('F00', 'F11'), (8, 16, 32, 64, 128)),
        ('netbounds-q1-expansion', ('F00', 'F11'), (8, 16, 32, 64, 128)),
        ('partial-crown', ('F00', 'F11'), (8, 16, 32, 64)),
        ('netbounds-q1-l2', ('F', 'F00', 'F11'), (8, 16, 32, 64, 128)),
        ('partial-crown-affine-l2', ('F', 'F00', 'F11'), (8, 16, 32, 64)),
        ('sup-reference', ('F00', 'F11'), (257,)),
        ('gauss-reference', ('F', 'F00', 'F11'), (200,)),
    ):
        expected.update((method, target, n) for target in targets for n in grids)
    actual = {(r['method'], r['target'], r['n']) for r in rows}
    if actual != expected or len(rows) != len(actual) or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('The registry must contain exactly the 60 published computations')
    return rows


def sources():
    paths = []
    for directory in ('crown_benchmark', 'netbounds_benchmark', 'independent_benchmark', 'reproducers'):
        paths.extend((ROOT / directory).glob('*.py'))
    paths.extend(p for p in (ROOT / 'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(ROOT / p for p in ('reproducers/cases.json', 'model_weights/manifest.json', CHECKPOINT))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def runtime():
    import numpy
    import scipy
    import torch
    versions = dict(torch=torch.__version__, numpy=numpy.__version__, scipy=scipy.__version__)
    expected = dict(torch='2.7.1+cpu', numpy='2.2.6', scipy='1.15.3')
    if versions != expected or sys.version_info[:2] != (3, 11):
        raise RuntimeError(f'Use Python 3.11 and requirements-reproduction.txt; found {versions}, Python {sys.version}')
    return versions


def worker(case_id, campaign):
    campaign = Path(campaign).resolve()
    case = next(r for r in cases() if r['case_id'] == case_id)
    row = campaign / 'rows' / case_id
    # A child may read its own output and frozen source, not another row.
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode, flags = args[1], args[2]
        reading = (isinstance(mode, str) and ('r' in mode or '+' in mode)) or (
            mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
        if reading:
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(campaign / 'rows') and not path.is_relative_to(row):
                raise PermissionError(f'Independent worker cannot read another row: {path}')
    sys.addaudithook(guard)
    runtime()
    if digest(ROOT / CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Wrong checkpoint')
    args = argparse.Namespace(output=row, campaign=campaign, n=case['n'],
                              method=case['method'], target=case['target'],
                              coordinate=None if case['target'] == 'F' else int(case['target'][-1]))
    from . import heat_second_order_e_comparison as e2
    from . import heat_second_order_q1_comparison as q1
    from . import heat_l2_matched_comparison as l2
    if case['method'] == 'sup-reference':
        e2.run_reference(args)
    elif case['method'] in ('netbounds-e2', 'partial-crown'):
        e2.run_row(args)
    elif case['method'] == 'netbounds-q1-expansion':
        q1.run_row(args)
    else:
        l2.run_row(args)


def launch(output, selected, acknowledge_expensive=False):
    versions = runtime()
    from .source_integrity import verify_vendor
    for component in ('netbounds', 'partial_crown'):
        verify_vendor(component)
    if any(r['method'].startswith('partial-crown') and r['n'] >= 64 for r in selected) and not acknowledge_expensive:
        raise ValueError('Fine partial-CROWN grids can take tens of minutes; add --acknowledge-expensive')
    output = Path(output).resolve()
    if output.is_relative_to(ROOT / 'results') or output == ROOT:
        raise ValueError('Use a new runs/ or external directory; published evidence is immutable')
    output.mkdir(parents=True, exist_ok=False)
    inventory = sources()
    snapshot = output / 'sources'
    for rel, expected in inventory.items():
        destination = snapshot / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
        if digest(destination) != expected:
            raise RuntimeError(f'Source changed while freezing: {rel}')
    write_json(output / 'manifest.json', dict(
        cases=selected, source_sha256=inventory, checkpoint_sha256=CHECKPOINT_SHA,
        versions=versions, python=sys.version, started_utc=datetime.now(timezone.utc).isoformat(),
        timing='Measured fresh; archived CPU measurements are not recomputed or reused.',
        independent_rows=True, threads=1, numerical_inputs='Frozen code and selected checkpoint only'))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(snapshot),
               HEAT_E_ORIGINAL_ROOT=str(ROOT))
    for key in ('NNBB_CHECKPOINT', 'NETBOUNDS_ROOT', 'PARTIAL_CROWN_PATH', 'CROWN_EXTERNAL_ROOT'):
        env.pop(key, None)
    for case in selected:
        command = [sys.executable, '-B', '-W', 'ignore', '-m', 'reproducers.heat_reproduce',
                   '_worker', case['case_id'], '--campaign', str(output)]
        with (output / (case['case_id'] + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / 'execution' / (case['case_id'] + '.json'),
                   dict(case_id=case['case_id'], returncode=result.returncode, command=command))
        if result.returncode:
            raise RuntimeError(f"Worker failed: {output / (case['case_id'] + '.log')}; no automatic retry")
        print(case['case_id'] + ': complete', flush=True)
    if any(digest(snapshot / rel) != sha for rel, sha in inventory.items()):
        raise RuntimeError('Frozen source changed during computation')
    write_json(output / 'complete.json', dict(case_count=len(selected),
        completed_utc=datetime.now(timezone.utc).isoformat(), sources_unchanged=True))


def compare_run(output):
    """Post-computation comparison only; no baseline is read by numerical workers."""
    import numpy as np
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    complete = json.loads((output / 'complete.json').read_text())
    if complete['case_count'] != len(manifest['cases']):
        raise ValueError('Incomplete run')
    array_count = 0
    all_bitwise = True
    for case in manifest['cases']:
        baseline = ROOT / case['result_path']
        new = output / 'rows' / case['case_id']
        a, b = json.loads(baseline.read_text()), json.loads((new / 'result.json').read_text())
        for key in ('n', 'cells', 'coordinate', 'method', 'epsilon', 'native_calls', 'stop_reason'):
            if key in a:
                if a[key] != b[key]:
                    raise ValueError(f"{case['case_id']}: changed {key}")
        for key in ('bound', 'signed_lower', 'signed_upper', 'lower', 'upper', 'lower_square', 'upper_square', 'observed_maximum'):
            if key in a:
                np.testing.assert_allclose(b[key], a[key], rtol=REPLAY_RTOL, atol=REPLAY_ATOL,
                                           err_msg=case['case_id'] + ':' + key)
        if 'reference' in a:
            for key in ('value', 'half_value'):
                np.testing.assert_allclose(b['reference'][key], a['reference'][key], rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        for name in ('cells.npz', 'reference.npz'):
            if (baseline.parent / name).exists():
                with np.load(baseline.parent / name, allow_pickle=False) as expected, np.load(new / name, allow_pickle=False) as actual:
                    if set(expected.files) != set(actual.files):
                        raise ValueError('Array schema differs')
                    for key in expected.files:
                        np.testing.assert_allclose(actual[key], expected[key], rtol=REPLAY_RTOL, atol=REPLAY_ATOL,
                                                   err_msg=case['case_id'] + ':' + key)
                        all_bitwise = all_bitwise and np.array_equal(actual[key], expected[key])
                        array_count += 1
    answer = dict(compared_rows=len(manifest['cases']), compared_arrays=array_count,
                  all_arrays_bitwise_equal=all_bitwise, rtol=REPLAY_RTOL, atol=REPLAY_ATOL,
                  cpu_times_compared=False)
    print(json.dumps(answer, indent=2))
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    commands.add_parser('verify')
    for command in ('smoke', 'run', 'row'):
        p = commands.add_parser(command)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--acknowledge-expensive', action='store_true')
        if command == 'row':
            p.add_argument('case_id')
        if command == 'run':
            p.add_argument('--netbounds-and-references-only', action='store_true')
    p = commands.add_parser('check-run'); p.add_argument('output', type=Path)
    p = commands.add_parser('_worker'); p.add_argument('case_id'); p.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    if args.command == '_worker':
        worker(args.case_id, args.campaign)
    elif args.command == 'list':
        print('\n'.join(r['case_id'] for r in cases()))
    elif args.command == 'verify':
        from .verify_release import verify
        print(json.dumps(verify(), indent=2))
    elif args.command == 'check-run':
        compare_run(args.output)
    else:
        selected = cases()
        if args.command == 'smoke':
            selected = [r for r in selected if r['n'] == 8 or r['method'] in ('sup-reference', 'gauss-reference')]
        elif args.command == 'row':
            selected = [r for r in selected if r['case_id'] == args.case_id]
            if not selected:
                parser.error('Unknown case ID; see list')
        elif args.netbounds_and_references_only:
            selected = [r for r in selected if not r['method'].startswith('partial-crown')]
        launch(args.output, selected, args.acknowledge_expensive)


if __name__ == '__main__':
    main()
