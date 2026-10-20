"""Read-only actual-data replay and explicit scratch mutation/relocation tests."""
import csv
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import pytest
from pde_residual_benchmark.campaign import ROOT,sha
from pde_residual_benchmark.report import audit,comparison,recorded_locations

CAMPAIGN=ROOT/'pde_residual_benchmark/measurements/heat1-uniform-v1'


@pytest.fixture(scope='module')
def publication():
    return comparison(CAMPAIGN)


def test_complete_actual_matrix_and_all_diagnostics(publication):
    md,table,result=publication
    rows=list(csv.DictReader(io.StringIO(table)))
    assert len(rows)==20 and len({(r['method'],r['n']) for r in rows})==20
    assert result['cells_checked']==5440 and result['distinct_worker_pids']==5
    assert result['all_numerical_diagnostics_pass']
    assert result['producing_commit']=='41c0ef1f4582357602236c02b51721d008826931'
    assert result['reference_J']==4.001960432102522
    assert result['reporter_sha256']==sha(ROOT/'pde_residual_benchmark/report.py')
    for row in rows:
        if row['method']=='partial_crown_affine_residual':
            assert row['provenance']=='fresh independent NN_bounds worker'
            assert float(row['cpu_s'])>0
        else:
            assert row['provenance']=='retained PDEbenchmarking measurement' and row['cpu_s']==''


def test_report_csv_and_audit_reproduce_exactly(publication):
    md,table,result=publication
    prefix=ROOT/'pde_residual_benchmark/heat1_uniform_comparison'
    assert Path(str(prefix)+'.md').read_text()==md
    assert Path(str(prefix)+'.csv').read_text()==table
    assert json.loads(Path(str(prefix)+'.audit.json').read_text())==result


def test_all_numeric_source_and_artifact_hashes_stay_unchanged(publication):
    m=json.loads((CAMPAIGN/'manifest.json').read_text())
    d=json.loads((CAMPAIGN/'complete.json').read_text())
    assert all(sha(CAMPAIGN/'sources'/p)==s for p,s in m['source_sha256'].items())
    assert all(sha(CAMPAIGN/p)==s for p,s in d['artifact_sha256'].items())
    provenance=json.loads((ROOT/'pde_residual_benchmark/inputs/provenance.json').read_text())
    assert all(sha(ROOT/'pde_residual_benchmark/inputs/retained_heat1'/p)==s for p,s in provenance['retained_sha256'].items())


def test_relocated_actual_campaign_without_rewriting_receipts(tmp_path,publication):
    moved=tmp_path/'moved-campaign';shutil.copytree(CAMPAIGN,moved)
    assert audit(moved)==publication[2]


def test_unrelated_worker_command_is_not_owned():
    e=json.loads((CAMPAIGN/'execution/uniform-n8.json').read_text())
    r=json.loads((CAMPAIGN/'rows/uniform-n8/receipt.json').read_text())
    recorded_locations(e,r,'uniform-n8')
    fake=copy.deepcopy(e);fake['command'][7]='uniform-n16'
    with pytest.raises(ValueError,match='owned numerical worker'):recorded_locations(fake,r,'uniform-n8')


def test_jsonl_read_cannot_hide_behind_refreshed_hashes(tmp_path):
    moved=tmp_path/'mutated-fixture';shutil.copytree(CAMPAIGN,moved)
    receipt=moved/'rows/reference-g32-g64/receipt.json'
    r=json.loads(receipt.read_text());r['read_paths'].append('/unrelated-campaign/native_calls.jsonl')
    receipt.write_text(json.dumps(r,indent=2)+'\n')
    done=moved/'complete.json';d=json.loads(done.read_text())
    d['artifact_sha256'][str(receipt.relative_to(moved))]=sha(receipt)
    done.write_text(json.dumps(d,indent=2)+'\n')
    with pytest.raises(ValueError,match='Unadmitted external data/source'):audit(moved)
