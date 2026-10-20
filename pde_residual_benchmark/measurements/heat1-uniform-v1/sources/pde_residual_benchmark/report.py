"""Read-only all-cell audits and deterministic same-checkpoint comparison report."""
import argparse
import csv
import io
import itertools
import json
import math
import os
from pathlib import Path
import numpy as np
from .campaign import ROOT,sha,load,case_matrix,runtime_read_allowed
from .core import CHECKPOINT_SHA
from independent_benchmark.minimal_affine_l2_native import cell_integrals
from independent_benchmark.affine_l2 import cell_square_integral


def audit(output):
    output=Path(output).resolve();m=load(output/'manifest.json');done=load(output/'complete.json')
    expected=case_matrix(m['selection']=='qualification-n8')
    if m['cases']!=expected or done['case_count']!=len(expected):raise ValueError('Incomplete or changed case matrix')
    for rel,digest in m['source_sha256'].items():
        if sha(output/'sources'/rel)!=digest:raise ValueError(f'Frozen source drift: {rel}')
    for rel,digest in done['artifact_sha256'].items():
        if sha(output/rel)!=digest:raise ValueError(f'Numerical artifact drift: {rel}')
    if m['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Wrong model')
    count=0;reconstruction=0.;worst_alternate=0.;rows=[];pids=set();previous_end=None
    reference=load(output/'rows/reference-g32-g64/numerical.json')
    for case in expected:
        folder=output/'rows'/case['id'];r=load(folder/'numerical.json');receipt=load(folder/'receipt.json')
        execution=load(output/'execution'/(case['id']+'.json'))
        if r['case']!=case or r['checkpoint_sha256']!=CHECKPOINT_SHA or r['producer_commit']!=m['producer_commit']:
            raise ValueError('Changed row/checkpoint/source identity')
        if execution['returncode']!=0 or not execution['exact_child_wait4'] or execution['pid']!=receipt['pid']:
            raise ValueError('No matching successful owned child exit')
        if receipt['pid'] in pids:raise ValueError('Rows must have distinct workers')
        pids.add(receipt['pid'])
        if previous_end and previous_end>execution['started_utc']:raise ValueError('Timed workers overlapped')
        previous_end=execution['ended_utc']
        for name,digest in receipt['artifact_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Row mutated after freeze')
        if any(p['num_threads']!=1 for p in receipt['runtime']['threadpools']):raise ValueError('Thread mismatch')
        # Screen EVERY recorded external read, including .jsonl/.log journals.
        prefixes=receipt['runtime'].get('runtime_prefixes')
        if prefixes!=m['runtime'].get('runtime_prefixes'):raise ValueError('Runtime-prefix admission mismatch')
        for text in receipt['read_paths']:
            p=Path(text)
            if p.is_relative_to(output/'sources') or p.is_relative_to(folder):continue
            if runtime_read_allowed(p,prefixes):continue
            raise ValueError(f'Unadmitted external data/source read: {p}')
        if case['kind']=='reference':
            if r['degree_difference']>1e-10:raise ValueError('Reference degree self-check did not converge')
            continue
        n=case['n'];size=n*n
        with np.load(folder/'cells.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        index=np.indices((n,n)).reshape(2,-1).T
        np.testing.assert_array_equal(a['lower'],index/n)
        np.testing.assert_array_equal(a['upper'],(index+1)/n)
        if r['native_cells_attempted']!=size or r['native_subsolves']!=4*size:raise ValueError('Wrong native work count')
        records=[json.loads(s) for s in (folder/'native_calls.jsonl').read_text().splitlines()]
        if len(records)!=size:raise ValueError('Incomplete native journal')
        for i,raw in enumerate(records):
            if raw['index']!=i or raw['completed_stages']!=['F','Fx','Fxx','Ft']:raise ValueError('Incomplete native cell')
            np.testing.assert_array_equal(raw['lower'],a['lower'][i]);np.testing.assert_array_equal(raw['upper'],a['upper'][i])
            for kind in ('Ft','Fxx','residual'):
                v=raw[kind]
                for key,value in v.items():np.testing.assert_array_equal(value,a[kind+'_'+key][i])
                corners=list(itertools.product(*zip(a['lower'][i],a['upper'][i])))
                lo=min(math.fsum((v['c_lower'],*(x*y for x,y in zip(v['A_lower'],p)))) for p in corners)
                hi=max(math.fsum((v['c_upper'],*(x*y for x,y in zip(v['A_upper'],p)))) for p in corners)
                error=max(abs(lo-v['lower_bound']),abs(hi-v['upper_bound']));reconstruction=max(reconstruction,error)
                if error>64*np.finfo(float).eps*max(1.,abs(lo),abs(hi)):raise ValueError('Native affine readout does not concretize')
            ft,xx,p=raw['Ft'],raw['Fxx'],raw['residual']
            np.testing.assert_array_equal(p['A_lower'],np.asarray(ft['A_lower'])-.1*np.asarray(xx['A_upper']))
            np.testing.assert_array_equal(p['A_upper'],np.asarray(ft['A_upper'])-.1*np.asarray(xx['A_lower']))
            if p['c_lower']!=ft['c_lower']-.1*xx['c_upper'] or p['c_upper']!=ft['c_upper']-.1*xx['c_lower']:
                raise ValueError('Wrong PDE coefficient/sign')
            values=cell_integrals(p,a['lower'][i],a['upper'][i])
            for key,value in values.items():
                if value!=a[key][i]:raise ValueError('Cell integral not reproducible')
            alt=cell_square_integral(p['A_lower'],p['c_lower'],p['A_upper'],p['c_upper'],a['lower'][i],a['upper'][i])
            np.testing.assert_allclose(alt,[values['lower_square'],values['upper_square']],rtol=2e-10,atol=1e-13)
            worst_alternate=max(worst_alternate,max(abs(alt[j]-values[k]) for j,k in enumerate(('lower_square','upper_square'))))
        J=[math.fsum(a[k]) for k in ('lower_square','upper_square')]
        if r['J']!=J or r['L2']!=[math.sqrt(v) for v in J] or r['L2_width']!=r['L2'][1]-r['L2'][0]:raise ValueError('Wrong global reduction')
        if not (0<r['method_cpu_s']<=execution['child_cpu_s']+.05 and 0<r['method_wall_s']<=execution['child_wall_s']+.05):
            raise ValueError('Method clock exceeds enclosing owned child')
        d=load(folder/'diagnostics.json')
        diagnostic_ok=(d['points_checked']==9*size and d['own_reference_contained']
            and all(v['tolerance_excess_count']==0 for v in d['observations'].values())
            and all(v['native_replay_pass'] and v['primitive_audit_pass'] for v in d['native_replays']))
        rows.append(dict(n=n,cells=size,J=J,L2=r['L2'],L2_width=r['L2_width'],wall_s=r['method_wall_s'],
            cpu_s=r['method_cpu_s'],child_wall_s=execution['child_wall_s'],child_cpu_s=execution['child_cpu_s'],
            diagnostic_ok=diagnostic_ok,reference_contained=J[0]<=reference['J']<=J[1],
            scalar_ablation_J=d['shared_output_scalar_ablation_J']))
        count+=size
    return dict(integrity_pass=True,rows=rows,cells_checked=count,distinct_worker_pids=len(pids),
        max_affine_reconstruction_difference=reconstruction,max_alternate_integrator_difference=worst_alternate,
        reference_J=reference['J'],reference_L2=reference['L2'],reference_degree_difference=reference['degree_difference'],
        all_numerical_diagnostics_pass=all(r['diagnostic_ok'] and r['reference_contained'] for r in rows),
        producing_commit=m['producer_commit'])


def comparison(output):
    output=Path(output).resolve();checked=audit(output)
    inputs=ROOT/'pde_residual_benchmark/inputs';provenance=load(inputs/'provenance.json')
    for name,digest in provenance['retained_sha256'].items():
        if sha(inputs/'retained_heat1'/name)!=digest:raise ValueError('Retained comparator changed')
    metadata=load(inputs/'retained_heat1/metadata.json')
    if abs(metadata['J_empirical']-checked['reference_J'])>1e-10:raise ValueError('Checkpoint/residual baseline mismatch')
    rows=[]
    for method in ('netbounds_q0','netbounds_q1','adaquad_dense_ibp','adaquad_dense_recurjac'):
        data=load(inputs/'retained_heat1'/(method+'.json'))
        for pc in checked['rows']:
            selected=[r for r in data if r['setup']==f'uniform-n{pc["n"]}']
            if len(selected)!=1 or selected[0]['status']!='complete':raise ValueError('Missing retained setup')
            r=selected[0];L2=[math.sqrt(x) for x in r['J']]
            rows.append(dict(method=method,n=pc['n'],cells=pc['cells'],J_lower=r['J'][0],J_upper=r['J'][1],
                L2_lower=L2[0],L2_upper=L2[1],L2_width=L2[1]-L2[0],wall_s=r['wall_s'],cpu_s='',
                provenance='retained PDEbenchmarking measurement',source=metadata.get('method_sources',{}).get(method,metadata['source'])))
    for pc in checked['rows']:
        rows.append(dict(method='partial_crown_affine_residual',n=pc['n'],cells=pc['cells'],
            J_lower=pc['J'][0],J_upper=pc['J'][1],L2_lower=pc['L2'][0],L2_upper=pc['L2'][1],L2_width=pc['L2_width'],
            wall_s=pc['wall_s'],cpu_s=pc['cpu_s'],provenance='fresh independent NN_bounds worker',source=checked['producing_commit']))
    if len(rows)!=len(checked['rows'])*5 or len({(r['method'],r['n']) for r in rows})!=len(rows):raise ValueError('Comparison matrix incomplete')
    data_path=os.path.relpath(output,ROOT/'pde_residual_benchmark')
    text=['# Heat1 PDE residual: matched uniform-grid comparison','',
        f'Informal original-forward Gauss64 reference: **J = {checked["reference_J"]:.16g}**, **L2 = {checked["reference_L2"]:.12g}**; Gauss32/64 difference {checked["reference_degree_difference"]:.3g}. This is not a bound.','',
        'Target: $r=F_t-0.1F_{xx}$ for the **exact PDEbenchmarking Heat1 checkpoint**, coordinates $(x,t)$, on $[0,1]^2$. No boundary mask, training, or random model initialization is used. The original raw-NN report uses a different trained checkpoint and is unchanged.','',
        f'Checkpoint SHA256: `{CHECKPOINT_SHA}`. Architecture `2 → 128 → 128 → 1`, tanh, storage/execution/accumulation float64, CPU, one effective numerical thread. Fresh numerical producer: `{checked["producing_commit"]}`.','',
        '## Measured uniform bounds','',
        'Each n means n cells per coordinate (n² cells). Norm endpoints are square roots AFTER globally summing squared-integral endpoints. Widths use full-precision values, not rounded table cells.','',
        '| n | Cells | Method | L2 lower | L2 upper | L2 width | Wall (s) |','|---:|---:|---|---:|---:|---:|---:|']
    labels=dict(netbounds_q0='NetBounds Q0 (retained)',netbounds_q1='NetBounds Q1 (retained)',
        adaquad_dense_ibp='AdaQuad dense IBP (retained)',adaquad_dense_recurjac='AdaQuad dense RecurJac (retained)',
        partial_crown_affine_residual='∂-CROWN affine residual (fresh)')
    for n in sorted({r['n'] for r in rows}):
        for r in [v for v in rows if v['n']==n]:
            text.append(f'| {n} | {r["cells"]} | {labels[r["method"]]} | {r["L2_lower"]:.9g} | {r["L2_upper"]:.9g} | {r["L2_width"]:.6g} | {r["wall_s"]:.6f} |')
    text+=['','## Construction and timing','',
        'The unmodified pinned ∂-CROWN chain computes a fresh value solution plus Fx/Fxx and Ft on every cell. Native input-affine derivative pairs are combined **before concretization**: lower = Ft_lower − 0.1 Fxx_upper; upper = Ft_upper − 0.1 Fxx_lower. The harness integrates the exact piecewise-quadratic square sandwich by polygon clipping and closed triangle moments; it does not numerically quadrature across switching kinks. This is a harness reduction of authors-native derivative bounds, not a released end-to-end heat-verifier mode.','',
        'Fresh method clocks include native setup, grid construction, full propagation, coefficient readout, signed PDE composition, affine-square moments and final sum/square roots. They exclude checkpoint loading, serialization and post-freeze diagnostics. Both process CPU and elapsed wall are saved. Exact-child wait4 CPU and parent elapsed wall include startup/loading/diagnostics/I/O/exit. Minimal progress/call bookkeeping is charged.','',
        'NetBounds/AdaQuad numbers and wall clocks are retained byte-for-byte from their original PDEbenchmarking run, not newly timed or converted from CPU. Original metadata and per-method producing URLs are in inputs/retained_heat1. They use Torch 2.12.0 / NumPy 2.4.6; fresh ∂-CROWN uses its validated Torch 2.7.1 / NumPy 2.2.6 / SciPy 1.15.3 stack. These are single measurements with different preparation/software boundaries, not repeated trials or a controlled engine-only speed ratio.','',
        '## Validation and evidence','',
        f'- Integrity audit: PASS; {checked["cells_checked"]} full native cells, {checked["distinct_worker_pids"]} distinct independently loaded workers (including reference).',
        f'- Post-freeze numerical diagnostics: {"PASS" if checked["all_numerical_diagnostics_pass"] else "FAIL/OBSERVED ISSUE — inspect diagnostics; native outputs are retained unchanged"}. Checks cover nine points per cell for Ft, Fxx and the combined residual; selected native affine/primitive replays; full-cell alternate polygon integration; own-row Gauss64 containment.',
        '- Scalar-lift values in diagnostics reuse these same native outputs and have no independent propagation timing; they are ablations, not extra benchmark rows.',
        '- Ordinary float64 bounds are **not outward-rounded machine certificates**. Point checks and Gauss agreement are observations, not proofs.',
        '- Only uniform n=8,16,32,64 are authorized here. No n=128, adaptive rows, GPU, new AdaQuad/NetBounds sweep, or legacy-row replacement.',
        '',f'Raw new campaign: [manifest.json]({data_path}/manifest.json); numerical arrays, native call journals, exact child receipts, post-freeze diagnostics and frozen numerical source are under that directory.',
        'Retained-input byte hashes and source revision: [inputs/provenance.json](inputs/provenance.json).','']
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return '\n'.join(text),stream.getvalue(),checked


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign');p.add_argument('--output-prefix');args=p.parse_args()
    md,csv_text,checked=comparison(args.campaign)
    if args.output_prefix:
        prefix=Path(args.output_prefix)
        for suffix,content in (('.md',md),('.csv',csv_text),('.audit.json',json.dumps(checked,indent=2)+'\n')):
            path=Path(str(prefix)+suffix)
            with path.open('x') as f:f.write(content)
    print(json.dumps(checked,indent=2))

if __name__=='__main__':main()
