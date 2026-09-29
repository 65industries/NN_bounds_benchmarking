"""Mutation tests: a reproduction must not choose its own verification authority."""
import json
from pathlib import Path
import shutil
import pytest
from reproducers.heat_reproduce import cases, launch, compare_run


@pytest.fixture(scope='module')
def completed_reference(tmp_path_factory):
    output = tmp_path_factory.mktemp('fresh-reference') / 'run'
    launch(output, [r for r in cases() if r['case_id'] == 'gauss-reference-F'])
    return output


@pytest.fixture
def candidate(completed_reference, tmp_path):
    output = tmp_path / 'run'
    shutil.copytree(completed_reference, output)
    return output


def update(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_valid_fresh_reference(completed_reference):
    assert compare_run(completed_reference)['compared_rows'] == 1


def test_run_cannot_redirect_baseline_to_itself(candidate):
    result = candidate / 'rows/gauss-reference-F/result.json'
    update(result, lambda d: d['reference'].update(value=0., half_value=0.))
    update(candidate / 'manifest.json', lambda d: d['cases'][0].update(result_path=str(result)))
    with pytest.raises((ValueError, AssertionError)):
        compare_run(candidate)


def test_empty_matrix_cannot_pass(candidate):
    update(candidate / 'manifest.json', lambda d: d.update(cases=[]))
    update(candidate / 'complete.json', lambda d: d.update(case_count=0))
    with pytest.raises((ValueError, AssertionError)):
        compare_run(candidate)


def test_duplicate_matrix_cannot_pass(candidate):
    update(candidate / 'manifest.json', lambda d: d['cases'].append(dict(d['cases'][0])))
    update(candidate / 'complete.json', lambda d: d.update(case_count=2))
    with pytest.raises((ValueError, AssertionError)):
        compare_run(candidate)


def test_failed_source_integrity_cannot_pass(candidate):
    update(candidate / 'complete.json', lambda d: d.update(sources_unchanged=False))
    with pytest.raises((ValueError, AssertionError)):
        compare_run(candidate)


def test_wrong_checkpoint_cannot_pass(candidate):
    update(candidate / 'manifest.json', lambda d: d.update(checkpoint_sha256='0' * 64))
    with pytest.raises((ValueError, AssertionError)):
        compare_run(candidate)


def test_changed_reference_contract_cannot_pass(candidate):
    update(candidate / 'rows/gauss-reference-F/result.json',
           lambda d: d['reference'].update(degree=4, half_degree=2, self_check_abs_difference=100.))
    with pytest.raises((ValueError, AssertionError)):
        compare_run(candidate)


@pytest.mark.parametrize('relative', ['execution/gauss-reference-F.json',
                                     'rows/gauss-reference-F/diagnostics.json',
                                     'sources/vendor/partial_crown/pinn_verifier/crown.py'])
def test_missing_run_evidence_cannot_pass(candidate, relative):
    (candidate / relative).unlink()
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        compare_run(candidate)


def test_altered_frozen_source_cannot_pass(candidate):
    path = candidate / 'sources/reproducers/heat_l2_matched_comparison.py'
    path.write_bytes(path.read_bytes() + b'\n# changed\n')
    with pytest.raises((ValueError, AssertionError)):
        compare_run(candidate)


def test_declared_matrix_requires_all_selected_rows(candidate):
    update(candidate / 'manifest.json', lambda d: d.update(selection={'kind':'all'}))
    with pytest.raises((ValueError, AssertionError)):
        compare_run(candidate)


def reseal(candidate):
    from reproducers.heat_reproduce import digest
    from reproducers.run_contract import artifact_paths
    manifest = json.loads((candidate / 'manifest.json').read_text())
    hashes = {p: digest(candidate / p) for p in artifact_paths(manifest['cases'])}
    update(candidate / 'complete.json', lambda d: d.update(artifact_sha256=hashes))


def test_resealed_reference_metadata_still_cannot_pass(candidate):
    update(candidate / 'rows/gauss-reference-F/result.json',
           lambda d: d['reference'].update(degree=4, half_degree=2))
    reseal(candidate)
    with pytest.raises(ValueError, match='Gauss reference contract'):
        compare_run(candidate)


def test_resealed_checkpoint_policy_still_cannot_pass(candidate):
    update(candidate / 'rows/gauss-reference-F/result.json',
           lambda d: d['checkpoint'].update(execution_dtype='torch.float32'))
    reseal(candidate)
    with pytest.raises(ValueError, match='execution policy'):
        compare_run(candidate)


def test_postrun_diagnostic_failure_is_not_silently_validated(tmp_path):
    candidate = tmp_path / 'bound'
    launch(candidate, [r for r in cases() if r['case_id'] == 'netbounds-q1-l2-F-n8'])
    compare_run(candidate)
    update(candidate / 'rows/netbounds-q1-l2-F-n8/diagnostics.json',
           lambda d: d.update(tolerance_violating_points=1))
    reseal(candidate)
    with pytest.raises(ValueError, match='diagnostic failures retained'):
        compare_run(candidate)
