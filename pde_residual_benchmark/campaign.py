"""Fresh independent rows, frozen source closure, exact child accounting.

Checkpoint loading is outside method clocks, matching the retained PDE convention.
Native setup, geometry, propagation, affine composition, moments and global reduction
are charged. References, post-freeze diagnostics, serialization and exit are not.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT='pde_residual_benchmark/inputs/heat1-l2-w128.pt'
CHECKPOINT_SHA='514531c1d38d138b053de85842e4c4a92de0bc82a629d448437ad1c8b5028ba6'
GRIDS=(8,16,32,64)
_THREAD_LIMITER=None


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def failure_clocks(method_start,method_end,phase_start,observed):
    """Completed numerical work stays frozen even when later diagnostics fail."""
    end=observed if method_end is None else method_end
    return dict(elapsed_method_cpu_s=None if method_start is None else end[0]-method_start[0],
        elapsed_method_wall_s=None if method_start is None else end[1]-method_start[1],
        method_completed=method_end is not None,
        failure_phase_cpu_s=observed[0]-phase_start[0],failure_phase_wall_s=observed[1]-phase_start[1])


def runtime_read_allowed(path,prefixes=None):
    """Admit installed executable sources, named package metadata and devices only.

    Never admit arbitrary site-packages data or external .jsonl/.log journals.
    This evidence/read policy is not a security sandbox for malicious code.
    """
    path=Path(path)
    if str(path) in ('/dev/null','/dev/urandom'):return True
    if len(path.parts)==4 and path.parts[1]=='proc' and path.parts[2].isdigit() and path.name=='maps':return True
    prefixes=prefixes or (sys.prefix,sys.base_prefix)
    if not any(path.is_relative_to(Path(p).resolve()) for p in prefixes):return False
    if path.suffix in ('.py','.pyc','.so','.pyd','.zip') or '.so.' in path.name:return True
    metadata={'METADATA','PKG-INFO','entry_points.txt','top_level.txt','namespace_packages.txt',
              'direct_url.json','WHEEL','RECORD','INSTALLER','REQUESTED'}
    return path.name in metadata and any(p.endswith(('.dist-info','.egg-info')) for p in path.parts)


def runtime():
    global _THREAD_LIMITER
    import numpy,scipy,torch,threadpoolctl
    versions=dict(torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,threadpoolctl=threadpoolctl.__version__)
    if versions!=dict(torch='2.7.1+cpu',numpy='2.2.6',scipy='1.15.3',threadpoolctl='3.6.0') or sys.version_info[:2]!=(3,11):
        raise RuntimeError(f'Use pinned reproduction requirements: {versions}, Python {sys.version}')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    _THREAD_LIMITER=threadpoolctl.threadpool_limits(limits=1)
    pools=threadpoolctl.threadpool_info()
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1 or any(p['num_threads']!=1 for p in pools):
        raise RuntimeError('Effective numerical thread count must be one')
    return dict(versions=versions,python=sys.version,executable=sys.executable,runtime_prefixes=[sys.prefix,sys.base_prefix],
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),threadpools=pools,
        platform=list(os.uname()),
        host=socket.gethostname(),cpu_affinity=sorted(os.sched_getaffinity(0)),load_average=list(os.getloadavg()))


def inventory():
    paths=[]
    for d in ('crown_benchmark','netbounds_benchmark','independent_benchmark','reproducers','pde_residual_benchmark'):
        paths.extend((ROOT/d).glob('*.py'))
    paths.extend(p for p in (ROOT/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(ROOT/p for p in (CHECKPOINT,'pde_residual_benchmark/inputs/manifest.json',
        'requirements-reproduction.txt','pde_residual_benchmark/requirements.txt'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def case_matrix(smoke=False) -> list[dict[str, Any]]:
    return [dict(id='reference-g32-g64',kind='reference',degrees=[32,64])]+[
        dict(id=f'uniform-n{n}',kind='affine-residual-l2',n=n,cells=n*n) for n in ((8,) if smoke else GRIDS)]


def install_read_guard(campaign,row,original_root):
    reads=set();snapshot=(campaign/'sources').resolve()
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        mode,flags=args[1:3]
        reading=(isinstance(mode,str) and ('r' in mode or '+' in mode)) or (mode is None and flags&os.O_ACCMODE!=os.O_WRONLY)
        if not reading:return
        path=Path(os.fsdecode(args[0])).resolve();reads.add(str(path))
        if path.suffix=='.pt' and path!=snapshot/CHECKPOINT:
            raise PermissionError(f'Unselected checkpoint read: {path}')
        if path.is_relative_to(snapshot) or path.is_relative_to(row):return
        if not runtime_read_allowed(path):
            raise PermissionError(f'Independent numerical worker cannot read unadmitted external data/source: {path}')
    sys.addaudithook(guard)
    return reads


def row_diagnostics(model,arrays,kernel):
    """Observations only; no returned endpoint is modified or selected."""
    import numpy as np
    import torch
    from .core import residual_components,gauss_reference,HeatResidualCrown,scalar_square
    from independent_benchmark.minimal_affine_l2_native import cell_integrals
    count=len(arrays['lower']);excess={'Ft':[],'Fxx':[],'residual':[]};point_count=0
    for start in range(0,count,64):
        lo,hi=arrays['lower'][start:start+64],arrays['upper'][start:start+64]
        points=np.stack([lo+(hi-lo)*np.array([a,b]) for a in (0.,.5,1.) for b in (0.,.5,1.)],axis=1)
        t,xx=residual_components(model,torch.from_numpy(points.reshape(-1,2)))
        values={'Ft':t.numpy().reshape(-1,9),'Fxx':xx.numpy().reshape(-1,9)}
        values['residual']=values['Ft']-.1*values['Fxx']
        for key,y in values.items():
            A=arrays[key+'_A_lower'][start:start+64];B=arrays[key+'_A_upper'][start:start+64]
            lower=np.einsum('ij,ikj->ik',A,points)+arrays[key+'_c_lower'][start:start+64,None]
            upper=np.einsum('ij,ikj->ik',B,points)+arrays[key+'_c_upper'][start:start+64,None]
            excess[key].append(np.maximum(lower-y,y-upper))
        point_count+=points.shape[0]*9
    observations={}
    for key,parts in excess.items():
        e=np.concatenate(parts);observations[key]=dict(strict_excess_count=int((e>0).sum()),
            tolerance_excess_count=int((e>1e-9).sum()),max_signed_excess=float(e.max()),tolerance=1e-9)
    selected=sorted({0,count//2,count-1});replays=[]
    # Replay with observational primitive wrappers outside method clocks.
    checker=HeatResidualCrown(model,audit=True)
    for i in selected:
        raw={};entry={'index':i,'native_replay_pass':False}
        try:
            checker.bound(arrays['lower'][i].tolist(),arrays['upper'][i].tolist(),raw)
            max_error=0.
            for kind in ('Ft','Fxx','residual'):
                for key,value in raw[kind].items():
                    expected=arrays[kind+'_'+key][i]
                    np.testing.assert_allclose(value,expected,rtol=2e-12,atol=5e-13)
                    max_error=max(max_error,float(np.max(np.abs(np.asarray(value)-expected))))
            entry.update(native_replay_pass=True,max_absolute_difference=max_error)
            primitive_count=sum(r.flush_audit() for r in checker.relaxations)
            entry.update(primitive_audit_pass=True,primitive_records=primitive_count)
        except Exception:
            entry.update(error=traceback.format_exc(),primitive_audit_pass=False)
            # New object avoids carrying a failed diagnostic's pending audit state.
            checker=HeatResidualCrown(model,audit=True)
        replays.append(entry)
    scalar_lo=[];scalar_hi=[]
    for i in range(count):
        l=arrays['Ft_lower_bound'][i]-.1*arrays['Fxx_upper_bound'][i]
        u=arrays['Ft_upper_bound'][i]-.1*arrays['Fxx_lower_bound'][i]
        v=float(np.prod(arrays['upper'][i]-arrays['lower'][i]))
        l,u=scalar_square(l,u,v);scalar_lo.append(l);scalar_hi.append(u)
    J=gauss_reference(model,64)
    return dict(policy='Post-freeze observations; never repair or replace native output',
        points_checked=point_count,observations=observations,native_replays=replays,
        own_row_gauss64_J=J,shared_output_scalar_ablation_J=[math.fsum(scalar_lo),math.fsum(scalar_hi)],
        scalar_ablation='Same native derivative outputs; not an independent benchmark row or timer')


def worker(case_id,campaign):
    campaign=Path(campaign).resolve();manifest=load(campaign/'manifest.json')
    case=next(c for c in manifest['cases'] if c['id']==case_id)
    row=campaign/'rows'/case_id;row.mkdir(parents=True,exist_ok=False)
    reads=install_read_guard(campaign,row,Path(manifest['original_root']))
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Owned worker received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    started=now();journals=[];arrays={};method_start=None;method_end=None
    phase='loading_and_setup';phase_start=(time.process_time(),time.perf_counter())
    try:
        effective=runtime()
        import numpy as np
        import torch
        from .core import load_heat1,HeatResidualCrown,grid_boxes,gauss_reference
        from independent_benchmark.minimal_affine_l2_native import cell_integrals
        if inventory()!=manifest['source_sha256']:raise RuntimeError('Frozen source inventory mismatch')
        loaded=load_heat1();model=loaded.model
        if sha(ROOT/CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('Wrong selected checkpoint')
        method_start=(time.process_time(),time.perf_counter())
        phase='numerical';phase_start=method_start
        if case['kind']=='reference':
            values={str(d):gauss_reference(model,d) for d in case['degrees']}
            method_end=(time.process_time(),time.perf_counter())
            cpu=method_end[0]-method_start[0];wall=method_end[1]-method_start[1]
            phase='serialization';phase_start=method_end
            numerical=dict(case=case,checkpoint_sha256=CHECKPOINT_SHA,producer_commit=manifest['producer_commit'],
                degrees=values,J=values['64'],L2=math.sqrt(values['64']),degree_difference=abs(values['32']-values['64']),
                cpu_s=cpu,wall_s=wall,status='complete',reference_kind='Informal original-forward nested-input-autograd Gauss-Legendre')
            write(row/'numerical.json',numerical)
        else:
            kernel=HeatResidualCrown(model)
            lower,upper=grid_boxes(case['n']);integrals=[]
            for i,(lo,hi) in enumerate(zip(lower,upper)):
                raw={'index':i};journals.append(raw)
                result=kernel.bound(lo.tolist(),hi.tolist(),raw)
                values=cell_integrals(result,lo,hi);integrals.append(values)
                if (i+1)%256==0:print(f'{case_id}: {i+1}/{case["cells"]} cells',flush=True)
            J=[math.fsum(x[k] for x in integrals) for k in ('lower_square','upper_square')]
            L2=[math.sqrt(x) for x in J]
            method_end=(time.process_time(),time.perf_counter())
            cpu=method_end[0]-method_start[0];wall=method_end[1]-method_start[1]
            phase='serialization';phase_start=method_end
            numerical=dict(case=case,checkpoint_sha256=CHECKPOINT_SHA,producer_commit=manifest['producer_commit'],
                J=J,L2=L2,L2_width=L2[1]-L2[0],method_cpu_s=cpu,method_wall_s=wall,
                native_cells_attempted=kernel.calls,native_subsolves=sum(len(j['completed_stages']) for j in journals),
                status='complete',stop_reason='complete_uniform_grid',backend=kernel.metadata)
            arrays=dict(lower=lower,upper=upper)
            for kind in ('Ft','Fxx','residual'):
                for key in journals[0][kind]:arrays[kind+'_'+key]=np.asarray([r[kind][key] for r in journals],dtype=np.float64)
            for key in integrals[0]:arrays[key]=np.asarray([r[key] for r in integrals],dtype=np.float64)
            # Freeze original numerical outputs BEFORE any independent checks.
            write(row/'numerical.json',numerical)
            with (row/'cells.npz').open('xb') as stream:
                np.savez_compressed(stream,**arrays)
            with (row/'native_calls.jsonl').open('x') as stream:
                for entry in journals:stream.write(json.dumps(entry,allow_nan=False,separators=(',',':'))+'\n')
            frozen={p.name:sha(p) for p in row.iterdir() if p.is_file()}
            phase='postfreeze_diagnostics';phase_start=(time.process_time(),time.perf_counter())
            diagnostics=row_diagnostics(model,arrays,kernel)
            diagnostics['own_reference_contained']=J[0]<=diagnostics['own_row_gauss64_J']<=J[1]
            if any(sha(row/name)!=digest for name,digest in frozen.items()):raise RuntimeError('Diagnostics mutated native evidence')
            write(row/'diagnostics.json',diagnostics)
        phase='receipt';phase_start=(time.process_time(),time.perf_counter())
        write(row/'receipt.json',dict(case_id=case_id,pid=os.getpid(),started_utc=started,ended_utc=now(),
            checkpoint=loaded.metadata,runtime=effective,read_paths=sorted(reads),sources_verified=True,
            read_policy='No other row, retained comparator, historical result, or nonselected checkpoint reads',
            artifact_sha256={p.name:sha(p) for p in row.iterdir() if p.is_file()}))
        print(json.dumps(numerical,allow_nan=False),flush=True)
    except BaseException:
        error=traceback.format_exc()
        observed=(time.process_time(),time.perf_counter())
        # A failing native call retains its attempted box and completed stages.
        def safe(v):
            if isinstance(v,float) and not math.isfinite(v):return {'nonfinite_float':repr(v)}
            if isinstance(v,dict):return {k:safe(w) for k,w in v.items()}
            if isinstance(v,(list,tuple)):return [safe(w) for w in v]
            return v
        write(row/'failure.json',dict(case=case,error=error,attempted_calls=safe(journals),
            started_utc=started,failed_utc=now(),failure_phase=phase,
            **failure_clocks(method_start,method_end,phase_start,observed)))
        print(error,file=sys.stderr,flush=True)
        raise


def launch(output,smoke=False):
    effective=runtime();selected=case_matrix(smoke)
    # Source must be committed/clean before a producer identity is claimed.
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True)
    if dirty:raise RuntimeError('Commit/freeze intended source before launching; worktree must be clean')
    producer=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    output=Path(output).resolve()
    if output==ROOT or output.is_relative_to(ROOT/'results'):
        raise ValueError('Never overwrite the historical published results')
    output.mkdir(parents=True,exist_ok=False);sources=inventory();snapshot=output/'sources'
    for rel,digest in sources.items():
        dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        if sha(dest)!=digest:raise RuntimeError(f'Drift while freezing {rel}')
    manifest=dict(cases=selected,approved_uniform_grids=list(GRIDS),selection='qualification-n8' if smoke else 'complete-uniform-matrix',
        producer_commit=producer,source_sha256=sources,checkpoint_sha256=CHECKPOINT_SHA,
        original_root=str(ROOT),started_utc=now(),runtime=effective,
        timing='Method CPU and wall include native setup/grid/propagation/affine composition/exact moments/reduction; exclude checkpoint loading, post-freeze checks, serialization. Exact wait4 child CPU and parent wall also retained.',
        target='plain F_t - 0.1 F_xx on [0,1]^2, x=0,t=1; no boundary mask',
        arithmetic='ordinary CPU float64, not outward-rounded machine certificates',independent_rows=True)
    write(output/'manifest.json',manifest)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(snapshot))
    for key in ('PYTHONHOME','NNBB_CHECKPOINT','NETBOUNDS_ROOT','PARTIAL_CROWN_PATH','CROWN_EXTERNAL_ROOT'):env.pop(key,None)
    current=None;start=time.perf_counter()
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Coordinator received signal {signum}')
    previous_sigterm=signal.signal(signal.SIGTERM,interrupted)
    try:
        for case in selected:
            rowid=case['id'];command=[sys.executable,'-B','-W','ignore','-m','pde_residual_benchmark','_worker',rowid,'--campaign',str(output)]
            launch_utc=now();wall0=time.perf_counter()
            with (output/(rowid+'.log')).open('x') as log:
                current=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                pid,status,usage=os.wait4(current.pid,0);current.returncode=os.waitstatus_to_exitcode(status)
            code=current.returncode;current=None
            receipt=dict(case_id=rowid,pid=pid,returncode=code,command=command,started_utc=launch_utc,ended_utc=now(),
                child_cpu_s=usage.ru_utime+usage.ru_stime,child_wall_s=time.perf_counter()-wall0,max_rss_kib=usage.ru_maxrss,
                exact_child_wait4=True)
            write(output/'execution'/(rowid+'.json'),receipt)
            print(f'{rowid}: exit={code}, child CPU={receipt["child_cpu_s"]:.3f}s, wall={receipt["child_wall_s"]:.3f}s',flush=True)
            if code:raise RuntimeError(f'Failed row {rowid}; preserved evidence; no retry or earlier-frontier substitution')
        if inventory()!=sources:raise RuntimeError('Live source drifted during campaign')
        if any(sha(snapshot/rel)!=digest for rel,digest in sources.items()):raise RuntimeError('Frozen source drifted')
        artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and not p.is_relative_to(snapshot)}
        write(output/'complete.json',dict(case_count=len(selected),bound_rows=len(selected)-1,reference_rows=1,
            source_unchanged=True,completed_utc=now(),campaign_wall_s=time.perf_counter()-start,artifact_sha256=artifacts))
    except BaseException:
        if current is not None and current.returncode is None:
            os.killpg(current.pid,signal.SIGTERM)
            pid,status,usage=os.wait4(current.pid,0);current.returncode=os.waitstatus_to_exitcode(status)
            # Retain the owned interrupted child's exit and actual costs as well.
            path=output/'execution'/(case['id']+'.json')
            if not path.exists():write(path,dict(case_id=case['id'],pid=pid,returncode=current.returncode,
                child_cpu_s=usage.ru_utime+usage.ru_stime,child_wall_s=time.perf_counter()-wall0,
                max_rss_kib=usage.ru_maxrss,command=command,exact_child_wait4=True,interrupted=True))
        write(output/'failed.json',dict(failed_utc=now(),error=traceback.format_exc()))
        raise
    finally:
        signal.signal(signal.SIGTERM,previous_sigterm)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('run','smoke'):
        p=sub.add_parser(name);p.add_argument('--output',required=True)
    p=sub.add_parser('_worker');p.add_argument('case_id');p.add_argument('--campaign',required=True)
    args=parser.parse_args()
    if args.action=='_worker':worker(args.case_id,args.campaign)
    else:launch(args.output,smoke=args.action=='smoke')

if __name__=='__main__':main()
