# heat1 quadrature

**Empirical L2 residual norm (informal baseline, not a bound): 2.0004900479888725**

Source: https://github.com/65industries/PDEbenchmarking/commit/bd33cd785b7e81726e56e53e70a9f690d620bd4c

Source (NetBounds Q0): https://github.com/65industries/PDEbenchmarking/commit/6e3a9cae3259c9927e77c8a7c0cea520cbfbcc87

Runtime: cpu, float64, 1 CPU thread(s).
Ordinary floating point; not outward-rounded machine certificates.

## NetBounds Q0

| Setup | L2 lower | L2 upper | Wall (s) |
|---|---:|---:|---:|
| uniform-n8 | 4.787e-01 | 3.851e+01 | 3.954e-02 |
| uniform-n16 | 1.676e+00 | 2.442e+00 | 1.060e-01 |
| uniform-n32 | 1.964e+00 | 2.037e+00 | 4.269e-01 |
| uniform-n64 | 1.988e+00 | 2.013e+00 | 1.817e+00 |
| greedy-s64 | 1.715e+00 | 2.295e+00 | 4.322e-01 |
| greedy-s320 | 1.965e+00 | 2.035e+00 | 2.005e+00 |
| greedy-s1344 | 1.988e+00 | 2.013e+00 | 9.362e+00 |
| doerfler-s64-theta0p35 | 1.715e+00 | 2.295e+00 | 1.772e-01 |
| doerfler-s320-theta0p35 | 1.965e+00 | 2.035e+00 | 6.575e-01 |
| doerfler-s1344-theta0p35 | 1.988e+00 | 2.013e+00 | 2.420e+00 |

## NetBounds Q1

| Setup | L2 lower | L2 upper | Wall (s) |
|---|---:|---:|---:|
| uniform-n8 | 4.454e-01 | 5.744e+01 | 6.502e-02 |
| uniform-n16 | 1.687e+00 | 2.524e+00 | 1.853e-01 |
| uniform-n32 | 1.992e+00 | 2.009e+00 | 7.735e-01 |
| uniform-n64 | 2.000e+00 | 2.001e+00 | 2.744e+00 |
| greedy-s64 | 1.730e+00 | 2.286e+00 | 5.271e-01 |
| greedy-s320 | 1.994e+00 | 2.007e+00 | 2.757e+00 |
| greedy-s1344 | 2.000e+00 | 2.001e+00 | 1.194e+01 |
| doerfler-s64-theta0p35 | 1.730e+00 | 2.286e+00 | 2.902e-01 |
| doerfler-s320-theta0p35 | 1.994e+00 | 2.007e+00 | 1.112e+00 |
| doerfler-s1344-theta0p35 | 2.000e+00 | 2.001e+00 | 4.652e+00 |

## AdaQuad dense IBP

| Setup | L2 lower | L2 upper | Wall (s) |
|---|---:|---:|---:|
| uniform-n8 | 0.000e+00 | 7.132e+00 | 1.446e+00 |
| uniform-n16 | 1.715e-02 | 4.664e+00 | 5.609e+00 |
| uniform-n32 | 7.040e-01 | 3.345e+00 | 2.262e+01 |
| uniform-n64 | 1.335e+00 | 2.673e+00 | 8.898e+01 |
| greedy-s64 | 1.715e-02 | 4.664e+00 | 6.936e+00 |
| greedy-s320 | 7.065e-01 | 3.345e+00 | 2.801e+01 |
| greedy-s1344 | 1.336e+00 | 2.673e+00 | 1.200e+02 |
| doerfler-s64-theta0p35 | 1.715e-02 | 4.664e+00 | 6.761e+00 |
| doerfler-s320-theta0p35 | 7.065e-01 | 3.345e+00 | 2.812e+01 |
| doerfler-s1344-theta0p35 | 1.336e+00 | 2.673e+00 | 1.168e+02 |

## AdaQuad dense RecurJac

| Setup | L2 lower | L2 upper | Wall (s) |
|---|---:|---:|---:|
| uniform-n8 | 0.000e+00 | 5.242e+00 | 1.436e+00 |
| uniform-n16 | 6.160e-01 | 3.395e+00 | 5.690e+00 |
| uniform-n32 | 1.357e+00 | 2.645e+00 | 2.282e+01 |
| uniform-n64 | 1.693e+00 | 2.308e+00 | 8.903e+01 |
| greedy-s64 | 6.160e-01 | 3.395e+00 | 7.012e+00 |
| greedy-s320 | 1.359e+00 | 2.645e+00 | 2.855e+01 |
| greedy-s1344 | 1.694e+00 | 2.308e+00 | 1.208e+02 |
| doerfler-s64-theta0p35 | 6.160e-01 | 3.395e+00 | 6.855e+00 |
| doerfler-s320-theta0p35 | 1.359e+00 | 2.645e+00 | 2.900e+01 |
| doerfler-s1344-theta0p35 | 1.694e+00 | 2.308e+00 | 1.178e+02 |

## NetBounds Q0 / AdaQuad ratios

| Setup | NB/IBP lower | NB/IBP upper | NB/IBP width | NB/IBP wall | NB/RecurJac lower | NB/RecurJac upper | NB/RecurJac width | NB/RecurJac wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform-n8 | undefined | 5.399e+00 | 5.332e+00 | 2.734e-02 | undefined | 7.346e+00 | 7.255e+00 | 2.755e-02 |
| uniform-n16 | 9.771e+01 | 5.236e-01 | 1.649e-01 | 1.890e-02 | 2.721e+00 | 7.194e-01 | 2.757e-01 | 1.863e-02 |
| uniform-n32 | 2.789e+00 | 6.090e-01 | 2.772e-02 | 1.887e-02 | 1.447e+00 | 7.702e-01 | 5.685e-02 | 1.871e-02 |
| uniform-n64 | 1.489e+00 | 7.531e-01 | 1.907e-02 | 2.042e-02 | 1.174e+00 | 8.722e-01 | 4.153e-02 | 2.041e-02 |
| greedy-s64 | 9.998e+01 | 4.921e-01 | 1.248e-01 | 6.231e-02 | 2.784e+00 | 6.761e-01 | 2.088e-01 | 6.164e-02 |
| greedy-s320 | 2.782e+00 | 6.084e-01 | 2.654e-02 | 7.158e-02 | 1.447e+00 | 7.694e-01 | 5.442e-02 | 7.022e-02 |
| greedy-s1344 | 1.488e+00 | 7.530e-01 | 1.906e-02 | 7.802e-02 | 1.173e+00 | 8.722e-01 | 4.151e-02 | 7.750e-02 |
| doerfler-s64-theta0p35 | 9.998e+01 | 4.921e-01 | 1.248e-01 | 2.621e-02 | 2.784e+00 | 6.761e-01 | 2.088e-01 | 2.585e-02 |
| doerfler-s320-theta0p35 | 2.782e+00 | 6.084e-01 | 2.654e-02 | 2.338e-02 | 1.447e+00 | 7.694e-01 | 5.442e-02 | 2.268e-02 |
| doerfler-s1344-theta0p35 | 1.488e+00 | 7.530e-01 | 1.906e-02 | 2.071e-02 | 1.173e+00 | 8.722e-01 | 4.151e-02 | 2.053e-02 |

## NetBounds Q1 / AdaQuad ratios

| Setup | NB/IBP lower | NB/IBP upper | NB/IBP width | NB/IBP wall | NB/RecurJac lower | NB/RecurJac upper | NB/RecurJac width | NB/RecurJac wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform-n8 | undefined | 8.053e+00 | 7.991e+00 | 4.496e-02 | undefined | 1.096e+01 | 1.087e+01 | 4.529e-02 |
| uniform-n16 | 9.836e+01 | 5.412e-01 | 1.801e-01 | 3.303e-02 | 2.739e+00 | 7.435e-01 | 3.012e-01 | 3.256e-02 |
| uniform-n32 | 2.829e+00 | 6.007e-01 | 6.726e-03 | 3.419e-02 | 1.468e+00 | 7.598e-01 | 1.379e-02 | 3.390e-02 |
| uniform-n64 | 1.499e+00 | 7.484e-01 | 5.028e-04 | 3.084e-02 | 1.181e+00 | 8.669e-01 | 1.095e-03 | 3.082e-02 |
| greedy-s64 | 1.009e+02 | 4.901e-01 | 1.196e-01 | 7.599e-02 | 2.809e+00 | 6.733e-01 | 2.000e-01 | 7.518e-02 |
| greedy-s320 | 2.823e+00 | 5.999e-01 | 4.731e-03 | 9.842e-02 | 1.468e+00 | 7.586e-01 | 9.701e-03 | 9.655e-02 |
| greedy-s1344 | 1.498e+00 | 7.484e-01 | 4.017e-04 | 9.955e-02 | 1.181e+00 | 8.669e-01 | 8.750e-04 | 9.888e-02 |
| doerfler-s64-theta0p35 | 1.009e+02 | 4.901e-01 | 1.196e-01 | 4.292e-02 | 2.809e+00 | 6.733e-01 | 2.000e-01 | 4.234e-02 |
| doerfler-s320-theta0p35 | 2.823e+00 | 5.999e-01 | 4.731e-03 | 3.956e-02 | 1.468e+00 | 7.586e-01 | 9.701e-03 | 3.836e-02 |
| doerfler-s1344-theta0p35 | 1.498e+00 | 7.484e-01 | 4.017e-04 | 3.981e-02 | 1.181e+00 | 8.669e-01 | 8.750e-04 | 3.948e-02 |
