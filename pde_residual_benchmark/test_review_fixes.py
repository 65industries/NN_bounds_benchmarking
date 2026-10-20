"""Regression cases for independent-review findings (explicit synthetic clocks)."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from pde_residual_benchmark.campaign import failure_clocks,runtime_read_allowed


def test_postfreeze_failure_does_not_charge_diagnostics():
    d=failure_clocks((10.,20.),(46.,56.),(46.,56.),(146.,156.))
    assert d==dict(elapsed_method_cpu_s=36.,elapsed_method_wall_s=36.,method_completed=True,
        failure_phase_cpu_s=100.,failure_phase_wall_s=100.)


def test_inflight_failed_native_work_is_still_charged():
    d=failure_clocks((10.,20.),None,(10.,20.),(46.,56.))
    assert d['elapsed_method_cpu_s']==d['elapsed_method_wall_s']==36.
    assert not d['method_completed']


def test_setup_failure_has_no_invented_numerical_time():
    d=failure_clocks(None,None,(10.,20.),(46.,56.))
    assert d['elapsed_method_cpu_s'] is None and d['elapsed_method_wall_s'] is None
    assert d['failure_phase_cpu_s']==36.


@pytest.mark.parametrize('name',['native_calls.jsonl','native_calls.log','other.json','cells.npz','reference.npy','values.csv','fake.py'])
def test_external_data_and_code_are_not_runtime_metadata(name,tmp_path):
    assert not runtime_read_allowed(tmp_path/name)


def test_only_named_installed_metadata_is_admitted():
    base=Path(sys.prefix)/'lib/python3.11/site-packages'
    assert runtime_read_allowed(base/'example/module.py')
    assert runtime_read_allowed(base/'example-1.dist-info/entry_points.txt')
    assert runtime_read_allowed(base/'example-1.dist-info/direct_url.json')
    assert not runtime_read_allowed(base/'example/native_calls.jsonl')
    assert not runtime_read_allowed(base/'example/other.json')


def test_real_audit_hook_denies_outside_campaign_journals(tmp_path):
    campaign=tmp_path/'campaign';own=campaign/'rows/one';own.mkdir(parents=True)
    other=tmp_path/'unrelated-campaign';other.mkdir()
    for name in ('native_calls.jsonl','native_calls.log'):(other/name).write_text('123')
    script='''
import json,sys
from pathlib import Path
from pde_residual_benchmark.campaign import install_read_guard
c,o,r=map(Path,sys.argv[1:]);install_read_guard(c,o,r)
out=[]
for name in ['native_calls.jsonl','native_calls.log']:
 try:(r/name).read_text()
 except PermissionError:out.append(True)
 else:out.append(False)
print(json.dumps(out))
'''
    r=subprocess.run([sys.executable,'-B','-c',script,str(campaign),str(own),str(other)],
        check=True,capture_output=True,text=True)
    assert json.loads(r.stdout)==[True,True]
