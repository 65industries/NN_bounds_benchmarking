# Supremum and L2 norm bounds: NetBounds and ∂-CROWN

## Scope and main distinction

CPU-only comparison on the raw scalar network $`F:[0,1]^2\to\mathbb R`$ stored in
`model_weights/heat_1d_separable_L2_W128_netbounds.pt`.
The targets are $`F_{00}=\partial_x^2 F`$ and $`F_{11}=\partial_t^2 F`$, with zero-based input coordinates $`(x_0,x_1)=(x,t)`$.
A separate $`L^2`$ section below also bounds the integral norms of $`F`$, $`F_{00}`$ and $`F_{11}`$ on the same cells.
The model is $`2\to128\to128\to1`$, with two tanh hidden layers and an affine output.

The NetBounds $`\mathcal E_F^{2e_i}`$ column evaluates **exactly the requested second-order quantity**

<a id="mathcal-e-definition"></a>

```math
\mathcal E_F^{2e_i}(y,\boldsymbol\varepsilon)
=|F_{ii}(y)|+\mathrm{Alg}_2(2e_i,F,y,\boldsymbol\varepsilon).\qquad (1)
```


## Grid, reported quantity, and guarantee

Here $`n`$ is the number of **cell centres per coordinate**, as confirmed for this experiment.
For $`j,k\in\{0,\ldots,n-1\}`$, define

```math
C_{jk}=[j/n,(j+1)/n]\times[k/n,(k+1)/n],\qquad
 y_{jk}=((j+1/2)/n,(k+1/2)/n),\qquad
 \boldsymbol\varepsilon=(1/(2n),1/(2n)).
```

Thus there are $`n^2`$ cells, with disjoint interiors, covering the entire closed unit square, including $`t=0`$ and every boundary.
The half-width vectors are $`(1/16,1/16)`$, $`(1/32,1/32)`$, $`(1/64,1/64)`$ and $`(1/128,1/128)`$ for $`n=8,16,32,64`$.
For $`n=128`$, each radius is $`1/256`$, with 16,384 cells per derivative. All these cell coordinates are exactly representable in binary64.

The $`\mathcal E_F^{2e_i}`$ evaluation returns a centre derivative $`b_{i,C}=F_{ii}(y_C)`$ and a variation radius $`r_{i,C}=\mathrm{Alg}_2(2e_i,F,y_C,\boldsymbol\varepsilon)`$.
The reported $`\mathcal E_F^{2e_i}`$ bound is

```math
U_i^{\mathrm{NB}}(n)=\max_C\bigl(|b_{i,C}|+r_{i,C}\bigr).
```

The native ∂-CROWN chain returns signed bounds $`L_{i,C}\le F_{ii}(z)\le H_{i,C}`$ on each cell. Its comparable reported number is

```math
U_i^{\partial\mathrm C}(n)=\max_C\max(-L_{i,C},H_{i,C}).
```

### $`F_{00}`$

**Independent numerical reference**

| Reference status | Observed maximum | Location $`(x,t)`$ | Evaluations | Evaluation grid | Nested-autograd CPU (s) | Evidence |
|:---|---:|:---:|---:|:---|---:|:---|
| Independent observation — **not an upper bound** | 7.22055593944 | $`(0.5859375,0)`$ | 66,049 | 257-by-257, endpoint-inclusive | 0.785495 | [Reference](../results/heat-second-order-e2-20260928-v1/rows/reference-F00/result.json) |

**Bound results**

| n | Cells | $`\mathcal E_F^{2e_0}`$ bound [(1)](#mathcal-e-definition) | $`\mathcal E_F^{2e_0}`$ CPU (s) | 4th-order expansion bound [(2)](#fourth-order-expansion-definition) | 4th-order expansion CPU (s) | ∂-CROWN bound | ∂-CROWN CPU (s) | Evidence |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 64 | 85.0817253698 | 0.007142 | 785.061956724 | 0.045224 | 7.8848853014 | 7.285417 | [$`\mathcal E_F^{2e_0}`$](../results/heat-second-order-e2-20260928-v1/rows/netbounds-e2-F00-n8/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F00-n8/result.json), [∂C](../results/heat-second-order-e2-20260928-v1/rows/partial-crown-F00-n8/result.json) |
| 16 | 256 | 8.6065342782 | 0.028384 | 27.3197255233 | 0.170797 | 7.48620263895 | 30.234633 | [$`\mathcal E_F^{2e_0}`$](../results/heat-second-order-e2-20260928-v1/rows/netbounds-e2-F00-n16/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F00-n16/result.json), [∂C](../results/heat-second-order-e2-20260928-v1/rows/partial-crown-F00-n16/result.json) |
| 32 | 1024 | 7.536663073 | 0.120274 | 7.47837323623 | 0.600595 | 7.29853796412 | 114.095469 | [$`\mathcal E_F^{2e_0}`$](../results/heat-second-order-e2-20260928-v1/rows/netbounds-e2-F00-n32/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F00-n32/result.json), [∂C](../results/heat-second-order-e2-20260928-v1/rows/partial-crown-F00-n32/result.json) |
| 64 | 4096 | 7.35579090334 | 0.538130 | 7.23645926946 | 2.461712 | 7.24043006652 | 444.512232 | [$`\mathcal E_F^{2e_0}`$](../results/heat-second-order-e2-n64-20260928-v1/rows/netbounds-e2-F00-n64/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F00-n64/result.json), [∂C](../results/heat-partial-crown-fine-20260928-v1/rows/partial-crown-F00-n64/result.json) |
| 128 | 16384 | 7.28605283969 | 3.729450 | 7.22384190433 | 9.793249 | not run | — | [$`\mathcal E_F^{2e_0}`$](../results/heat-second-order-e2-n128-20260928-v1/rows/netbounds-e2-F00-n128/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F00-n128/result.json) |

### $`F_{11}`$

**Independent numerical reference**

| Reference status | Observed maximum | Location $`(x,t)`$ | Evaluations | Evaluation grid | Nested-autograd CPU (s) | Evidence |
|:---|---:|:---:|---:|:---|---:|:---|
| Independent observation — **not an upper bound** | 5.0399340785 | $`(1,0.09765625)`$ | 66,049 | 257-by-257, endpoint-inclusive | 0.793639 | [Reference](../results/heat-second-order-e2-20260928-v1/rows/reference-F11/result.json) |

**Bound results**

| n | Cells | $`\mathcal E_F^{2e_1}`$ bound [(1)](#mathcal-e-definition) | $`\mathcal E_F^{2e_1}`$ CPU (s) | 4th-order expansion bound [(2)](#fourth-order-expansion-definition) | 4th-order expansion CPU (s) | ∂-CROWN bound | ∂-CROWN CPU (s) | Evidence |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 64 | 69.9055951591 | 0.007271 | 646.566696849 | 0.041912 | 5.57734284696 | 7.876738 | [$`\mathcal E_F^{2e_1}`$](../results/heat-second-order-e2-20260928-v1/rows/netbounds-e2-F11-n8/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F11-n8/result.json), [∂C](../results/heat-second-order-e2-20260928-v1/rows/partial-crown-F11-n8/result.json) |
| 16 | 256 | 7.67279263582 | 0.027784 | 23.3227524317 | 0.172013 | 5.23280665598 | 30.230704 | [$`\mathcal E_F^{2e_1}`$](../results/heat-second-order-e2-20260928-v1/rows/netbounds-e2-F11-n16/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F11-n16/result.json), [∂C](../results/heat-second-order-e2-20260928-v1/rows/partial-crown-F11-n16/result.json) |
| 32 | 1024 | 5.50056350295 | 0.128180 | 5.59906961725 | 0.659362 | 5.11273571409 | 115.110062 | [$`\mathcal E_F^{2e_1}`$](../results/heat-second-order-e2-20260928-v1/rows/netbounds-e2-F11-n32/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F11-n32/result.json), [∂C](../results/heat-second-order-e2-20260928-v1/rows/partial-crown-F11-n32/result.json) |
| 64 | 4096 | 5.23978110182 | 0.521769 | 5.06616134618 | 2.622950 | 5.05552564739 | 438.211375 | [$`\mathcal E_F^{2e_1}`$](../results/heat-second-order-e2-n64-20260928-v1/rows/netbounds-e2-F11-n64/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F11-n64/result.json), [∂C](../results/heat-partial-crown-fine-20260928-v1/rows/partial-crown-F11-n64/result.json) |
| 128 | 16384 | 5.1369479217 | 3.441643 | 5.04346125474 | 9.690299 | not run | — | [$`\mathcal E_F^{2e_1}`$](../results/heat-second-order-e2-n128-20260928-v1/rows/netbounds-e2-F11-n128/result.json), [Q1](../results/heat-second-order-q1-matched-20260928-v1/rows/netbounds-q1-expansion-F11-n128/result.json) |

### Interpretation

- $`F_{00}`$: Q1 is tighter than $`\mathcal E_F^{2e_0}`$ on $`n=32,64,128`$; $`\mathcal E_F^{2e_0}`$ is tighter or equal on $`n=8,16`$. At $`n=128`$, Q1 gives 7.22384190433 versus 7.28605283969; its excess above the retained sampled reference is 19.93 times smaller. This ratio uses an observed lower estimate, not the unknown exact supremum.
- $`F_{11}`$: Q1 is tighter than $`\mathcal E_F^{2e_1}`$ on $`n=64,128`$; $`\mathcal E_F^{2e_1}`$ is tighter or equal on $`n=8,16,32`$. At $`n=128`$, Q1 gives 5.04346125474 versus 5.1369479217; its excess above the retained sampled reference is 27.50 times smaller. This ratio uses an observed lower estimate, not the unknown exact supremum.
- ∂-CROWN is tighter than both NetBounds constructions on 6 of the six matched method/coordinate/grid comparisons at $`n=8,16,32`$. At $`n=64`$, the Q1 expansion is tighter than ∂-CROWN for $`F_{00}`$, while ∂-CROWN is tighter than Q1 for $`F_{11}`$; ∂-CROWN is tighter than $`\mathcal E_F^{2e_i}`$ for both. No $`n=128`$ ∂-CROWN comparison is included yet.
- Q1 pays for fourth-network-derivative envelopes and is not uniformly tighter: on coarse cells the large remainder envelope can dominate. These prescribed-grid timings are implementation costs, not equal-CPU search outcomes or a convergence-order proof. No minimum/intersection of the two NetBounds constructions is applied.


## NetBounds Q1 expansion of the second derivative

The construction used here is an **affine expansion with a quadratic remainder**, not a quadratic Taylor polynomial. For $`g_i=F_{ii}`$ and $`z=x-y_C`$,

```math
g_i(y_C+z)=g_i(y_C)+\sum_q \partial_qg_i(y_C)z_q+R_{i,C}(z),\qquad
\lvert R_{i,C}(z)\rvert\le\frac12\sum_{q,r}H^{(i)}_{qr,C}|z_q||z_r|,
```

where

```math
H^{(i)}_{qr,C}=|F_{iiqr}(y_C)|+\mathrm{Alg}_4((i,i,q,r),F,y_C,\boldsymbol\varepsilon)
\ge\sup_{x\in C}|F_{iiqr}(x)|.
```

The coordinate tuple in this formula denotes the fourth partial derivative; it is the native code's tuple convention. The reported expansion bound is

<a id="fourth-order-expansion-definition"></a>

```math
U_i^{\mathrm{Q1}}(n)=\max_C\left(|F_{ii}(y_C)|+\sum_q\varepsilon_q|F_{iiq}(y_C)|+
\frac12\sum_{q,r}\varepsilon_q\varepsilon_rH^{(i)}_{qr,C}\right).\qquad (2)
```

Both ordered mixed terms are included. This is a pointwise supremum enclosure, not Q1 moment integration or an $`L^2`$ bound. Its exact-arithmetic covering argument is the same as above; the float64 qualification remains unchanged.

The existing `independent_benchmark.second_derivative.run_netbounds` calls `RawQ1Integrator(..., 'second-diagonal', i, moments=False)` and `evaluate_fixed_grid(..., jets_only=True, batch_size=256)`, then its own `grid_reduction`. The native entry is `compute_up_to_fourth_order_bounds`: it supplies the centre second/third derivatives and the fourth-derivative envelopes. Default activation Taylor depths remain $`(5,4,3,2)`$. No propagation formula, truncation depth, reduction or bound is changed.

## $`L^2`$ norm bounds: scope and two quadrature rules

This additional experiment bounds

```math
\|g\|_{L^2([0,1]^2)}=\left(\int_{[0,1]^2}g(x,t)^2\,dx\,dt\right)^{1/2},
\qquad g\in\{F,F_{00},F_{11}\}.
```

The network, coordinates, and complete cell covers are exactly those defined above. These are norms of the **raw network and its pure second derivatives**, not of a boundary-masked solution, the derivatives of $`F^2`$, or a PDE residual. NetBounds uses $`n=8,16,32,64,128`$. Affine ∂-CROWN now has measured rows for $`n=8,16,32,64`$; the three $`n=128`$ rows remain pending publication. No adaptive partition, coarse-grid substitution, or interpolation is used.

### Fresh $`L^2`$ results and CPU time

#### $`\|F\|_{L^2}`$

**Independent numerical reference**

| Reference status | Observed $`L^2`$ value | Degree-100/200 absolute difference | Evidence |
|:---|---:|---:|:---|
| Independent Gauss–Legendre observation — **not a certified norm value** | 2.4537533347 | 4.44e-16 | [Reference](../results/heat-l2-matched-20260928-v1/rows/gauss-reference-F/result.json) |

**Bound results**

| n | Cells | NetBounds lower | NetBounds upper | NB CPU (s) | ∂-CROWN lower | ∂-CROWN upper | ∂C CPU (s) | Evidence |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 64 | 2.41164635031 | 2.4982063331 | 0.013453 | 2.43323271959 | 2.47360190034 | 1.590159 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F-n8/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F-n8/result.json) |
| 16 | 256 | 2.45224245217 | 2.45584861413 | 0.044342 | 2.44859896075 | 2.45866163871 | 5.400914 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F-n16/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F-n16/result.json) |
| 32 | 1024 | 2.45347357629 | 2.45417940524 | 0.147338 | 2.45245893226 | 2.45497675863 | 19.723776 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F-n32/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F-n32/result.json) |
| 64 | 4096 | 2.45368920248 | 2.45385405584 | 0.524673 | 2.45342559097 | 2.45406216066 | 73.102274 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F-n64/result.json), [∂C](../results/heat-partial-crown-fine-20260928-v1/rows/partial-crown-affine-l2-F-n64/result.json) |
| 128 | 16384 | 2.45373795439 | 2.45377786289 | 2.096138 | not run | not run | — | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F-n128/result.json) |

#### $`\|F_{xx}\|_{L^2}`$

**Independent numerical reference**

| Reference status | Observed $`L^2`$ value | Degree-100/200 absolute difference | Evidence |
|:---|---:|---:|:---|
| Independent Gauss–Legendre observation — **not a certified norm value** | 4.22968403274 | 1.78e-15 | [Reference](../results/heat-l2-matched-20260928-v1/rows/gauss-reference-F00/result.json) |

**Bound results**

| n | Cells | NetBounds lower | NetBounds upper | NB CPU (s) | ∂-CROWN lower | ∂-CROWN upper | ∂C CPU (s) | Evidence |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 64 | 0 | 98.6232850593 | 0.037010 | 3.94227638661 | 4.4964626309 | 8.122804 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F00-n8/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F00-n8/result.json) |
| 16 | 256 | 2.5705940009 | 5.92896692406 | 0.165906 | 4.15579872509 | 4.29912798613 | 31.010022 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F00-n16/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F00-n16/result.json) |
| 32 | 1024 | 4.19288404186 | 4.26995055698 | 0.637364 | 4.2110452522 | 4.24745650509 | 118.338605 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F00-n32/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F00-n32/result.json) |
| 64 | 4096 | 4.22815970672 | 4.23201095342 | 2.533729 | 4.22499313992 | 4.2341948424 | 484.960239 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F00-n64/result.json), [∂C](../results/heat-partial-crown-fine-20260928-v1/rows/partial-crown-affine-l2-F00-n64/result.json) |
| 128 | 16384 | 4.22952894705 | 4.2300396696 | 10.154266 | not run | not run | — | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F00-n128/result.json) |

#### $`\|F_{tt}\|_{L^2}`$

**Independent numerical reference**

| Reference status | Observed $`L^2`$ value | Degree-100/200 absolute difference | Evidence |
|:---|---:|---:|:---|
| Independent Gauss–Legendre observation — **not a certified norm value** | 2.36890890477 | 8.88e-16 | [Reference](../results/heat-l2-matched-20260928-v1/rows/gauss-reference-F11/result.json) |

**Bound results**

| n | Cells | NetBounds lower | NetBounds upper | NB CPU (s) | ∂-CROWN lower | ∂-CROWN upper | ∂C CPU (s) | Evidence |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 64 | 0 | 69.5467828089 | 0.040423 | 2.07561129716 | 2.69449874832 | 7.896473 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F11-n8/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F11-n8/result.json) |
| 16 | 256 | 0.879524551322 | 3.66403615823 | 0.176519 | 2.29247533136 | 2.45197048487 | 29.446782 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F11-n16/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F11-n16/result.json) |
| 32 | 1024 | 2.34053240952 | 2.39999083956 | 0.641883 | 2.34953235952 | 2.38991031967 | 117.983070 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F11-n32/result.json), [∂C](../results/heat-l2-matched-20260928-v1/rows/partial-crown-affine-l2-F11-n32/result.json) |
| 64 | 4096 | 2.36776061376 | 2.37068696931 | 2.512934 | 2.36403295916 | 2.37420711886 | 473.145962 | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F11-n64/result.json), [∂C](../results/heat-partial-crown-fine-20260928-v1/rows/partial-crown-affine-l2-F11-n64/result.json) |
| 128 | 16384 | 2.36879805952 | 2.36917708261 | 9.263451 | not run | not run | — | [NB](../results/heat-l2-matched-20260928-v1/rows/netbounds-q1-l2-F11-n128/result.json) |


### NetBounds: Q1 moments with a Hessian remainder

For $`C=y+[-\varepsilon_0,\varepsilon_0]\times[-\varepsilon_1,\varepsilon_1]`$, put

```math
a=g(y),\qquad b=\nabla g(y),\qquad
P(z)=a+b\cdot z,\qquad g(y+z)=P(z)+R(z).
```

The unchanged NetBounds derivative bounds provide $`H_{qr,C}\ge\sup_C|\partial_{qr}g|`$, hence

```math
\lvert R(z)\rvert\le D(z):=\frac12\sum_{q,r=0}^1 H_{qr,C}|z_q||z_r|,
\qquad |P(z)|\le A(z):=|a|+\sum_{q=0}^1|b_q||z_q|.
```

This requires network derivatives through order **two** for $`g=F`$, and through order **four** for $`g=F_{ii}`$, since $`\partial_{qr}g=F_{iiqr}`$. It is the same affine-plus-quadratic-remainder cell model as the earlier Q1 supremum rows, now integrated rather than maximized.

Define the three volume-included quantities

```math
Q_C=\int_C P^2=|C|\left(a^2+\sum_q b_q^2\frac{\varepsilon_q^2}{3}\right),
\qquad J_C=2\int_C A D,\qquad K_C=\int_C D^2.
```

All correction terms are evaluated by the exact-real absolute-monomial moments

```math
\int_{[-\boldsymbol\varepsilon,\boldsymbol\varepsilon]}|z_0|^{\alpha_0}|z_1|^{\alpha_1}\,dz
=|C|\prod_{q=0}^1\frac{\varepsilon_q^{\alpha_q}}{\alpha_q+1}.
```

Expanding $`g^2=P^2+2PR+R^2`$ proves

```math
Q_C-J_C\le\int_C g^2\le Q_C+J_C+K_C.
```

The heat-report reduction used here is therefore

```math
\boxed{\sqrt{\max\{0,\sum_C Q_C-\sum_C J_C\}}
\ \le\ \|g\|_{L^2}\ \le\ \sqrt{\sum_C(Q_C+J_C+K_C)}}.
```

The lower truncation is applied **once after global summation**, not separately on each cell. The upper uses native `upper_square`; the lower is the existing harness reduction of native `affine_square` and `cross_error`. The nonnegative $`R^2`$ term is omitted only in the lower estimate. This is not interval-squaring of the Q1 jet, and it is not a claim that a positive lower norm bound is available on every coarse grid.

Implementation: [`q1_moment_cell` and `RawQ1Integrator`](../netbounds_benchmark/uniform_q1.py), [`evaluate_fixed_grid`](../netbounds_benchmark/fixed_q1.py), with the global lower reduction [`reduce_q1`](../reproducers/heat_l2_matched_comparison.py). The reported lower is checked for ordering rather than silently repaired with `min(lower,upper)`.

### ∂-CROWN: integrate the affine square image

On each cell retain the **native input-affine pair**, not only its concretized scalar interval:

```math
\ell_C(x)=A_C^L\cdot x+c_C^L\le g(x)\le u_C(x)=A_C^U\cdot x+c_C^U.
```

For any ordered interval $`[\ell,u]`$, its squared image gives the pointwise bounds

```math
s_-(x)=\max\{\ell_C(x),0\}^2+\min\{u_C(x),0\}^2
\ \le\ g(x)^2\ \le\ s_+(x)=\max\{\ell_C(x)^2,u_C(x)^2\}.
```

In particular, the lower integrand is zero wherever the affine interval straddles zero. The resulting quadrature rule is

```math
\boxed{\sqrt{\sum_C\int_C s_-(x)\,dx}
\ \le\ \|g\|_{L^2}\ \le\ \sqrt{\sum_C\int_C s_+(x)\,dx}}.
```

These integrals are **piecewise-quadratic polygon integrals**, not midpoint or sampled quadrature. For the lower integral, clip the box by $`\ell_C\ge0`$ and by $`u_C\le0`$, and integrate the appropriate squared affine function. For the upper integral, the sign of $`\ell_C+u_C`$ determines which square is larger, because

```math
u_C^2-\ell_C^2=(u_C-\ell_C)(u_C+\ell_C),\qquad u_C-\ell_C\ge0.
```

Thus the separating set is a line; polygon subdivision and degree-two triangle moments evaluate the integral exactly in real arithmetic.
