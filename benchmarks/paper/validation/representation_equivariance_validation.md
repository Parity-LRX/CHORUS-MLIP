# SO2/O2 and double-cover O3 equivariance validation

All tests were run in float64/complex128 on CPU. Residuals compare
transform-then-apply against apply-then-transform.

| test | max abs residual | max relative residual | detail |
|---|---:|---:|---|
| SO2 CG paths m<= 4 (31 paths) | 1.776e-15 | 3.458e-16 | rotation abs=1.776e-15, reflection abs=0.000e+00 |
| SO2 fully connected TP module m<= 4 | 5.329e-15 | 2.353e-16 | 31 weighted paths, random channel weights |
| O2 TP reflection-closed subset, active 0e/0o/1/2/3 | 5.329e-15 | 1.962e-16 | rotation abs=5.329e-15, reflection abs=0.000e+00; excludes 0o x frequency -> frequency paths |
| ICTC orbital harmonic values l<=3 | 2.220e-16 | 6.272e-16 | ordinary O3 parent carrier used by double-cover backend |
| double-cover O3 CG tensor products | 1.061e-14 | 2.416e-15 | (l=0,S=1/2,j=1/2,e) x (l=1,S=0,j=1,o) -> (l=1,S=1/2,j=3/2,o): abs=4.078e-15; (l=0,S=1/2,j=1/2,e) x (l=0,S=1/2,j=1/2,e) -> (l=0,S=0,j=0,e): abs=2.095e-15; (l=2,S=1,j=1,e) x (l=1,S=0,j=1,o) -> (l=2,S=1,j=2,o): abs=1.061e-14 |
| double-cover O3 matrix bases row x col* -> out | 4.511e-16 | 5.940e-16 | 3 covariance paths |
