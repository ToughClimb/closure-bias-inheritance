# Baseline Summary

Representative settings for method comparison. Each cell reports mean +/- std across seeds.

| Case | Setting | Method | ErrD | ErrR | Unseen |
| --- | --- | --- | --- | --- | --- |
| Case A | clean | strong_poly | 4.098e-04 +/- 1.144e-04 | 2.113e-03 +/- 8.855e-04 | 1.804e-05 +/- 5.128e-06 |
| Case A | clean | weak_poly | 8.151e-03 +/- 1.047e-03 | 3.090e-02 +/- 9.034e-03 | 8.307e-04 +/- 1.348e-04 |
| Case A | clean | neural | 6.926e-02 +/- 4.007e-03 | 3.128e-01 +/- 1.892e-02 | 4.828e-03 +/- 3.137e-04 |
| Case A | clean | neural+symbolic | 6.853e-02 +/- 4.266e-03 | 3.127e-01 +/- 1.894e-02 | 4.701e-03 +/- 3.516e-04 |
| Case A | noise 5% | strong_poly | 9.773e-01 +/- 6.887e-03 | 6.242e+00 +/- 1.121e+00 | 1.066e-01 +/- 5.667e-03 |
| Case A | noise 5% | weak_poly | 3.057e-02 +/- 1.141e-02 | 2.736e-01 +/- 1.390e-01 | 3.289e-03 +/- 7.700e-04 |
| Case A | noise 5% | neural | 8.672e-01 +/- 3.843e-03 | 2.361e+00 +/- 9.133e-02 | 1.029e-01 +/- 1.653e-03 |
| Case A | noise 5% | neural+symbolic | 8.672e-01 +/- 3.843e-03 | 2.365e+00 +/- 9.124e-02 | 1.029e-01 +/- 1.643e-03 |
| Case A | sparse x+t | strong_poly | 3.637e-02 +/- 5.490e-03 | 1.392e-01 +/- 2.847e-02 | 2.841e-03 +/- 5.359e-04 |
| Case A | sparse x+t | weak_poly | 6.108e-02 +/- 9.593e-03 | 2.224e-01 +/- 6.296e-02 | 5.634e-03 +/- 1.144e-03 |
| Case A | sparse x+t | neural | 5.046e-02 +/- 6.225e-03 | 3.786e-01 +/- 1.593e-02 | 5.042e-03 +/- 1.778e-04 |
| Case A | sparse x+t | neural+symbolic | 4.685e-02 +/- 7.746e-03 | 3.786e-01 +/- 1.593e-02 | 4.473e-03 +/- 3.214e-04 |
| Case B | clean | strong_poly | 4.018e-04 +/- 1.088e-04 | 2.597e-03 +/- 1.092e-03 | 1.651e-05 +/- 4.630e-06 |
| Case B | clean | weak_poly | 8.030e-03 +/- 6.990e-04 | 3.701e-02 +/- 8.150e-03 | 9.556e-04 +/- 1.252e-04 |
| Case B | clean | neural | 1.003e-01 +/- 5.687e-03 | 2.047e-01 +/- 1.727e-02 | 3.949e-03 +/- 3.374e-04 |
| Case B | clean | neural+symbolic | 1.001e-01 +/- 5.650e-03 | 2.036e-01 +/- 1.756e-02 | 3.727e-03 +/- 3.585e-04 |
| Case B | noise 5% | strong_poly | 9.784e-01 +/- 8.145e-03 | 7.914e+00 +/- 1.429e+00 | 8.492e-02 +/- 4.839e-03 |
| Case B | noise 5% | weak_poly | 3.506e-02 +/- 9.718e-03 | 3.629e-01 +/- 1.339e-01 | 3.865e-03 +/- 9.514e-04 |
| Case B | noise 5% | neural | 8.470e-01 +/- 4.160e-03 | 3.248e+00 +/- 4.678e-02 | 7.572e-02 +/- 7.982e-04 |
| Case B | noise 5% | neural+symbolic | 8.470e-01 +/- 4.160e-03 | 3.270e+00 +/- 4.865e-02 | 7.579e-02 +/- 6.476e-04 |
| Case B | sparse x+t | strong_poly | 4.345e-02 +/- 6.396e-03 | 2.233e-01 +/- 5.839e-02 | 2.316e-03 +/- 4.350e-04 |
| Case B | sparse x+t | weak_poly | 5.357e-02 +/- 6.987e-03 | 2.465e-01 +/- 6.013e-02 | 5.121e-03 +/- 8.136e-04 |
| Case B | sparse x+t | neural | 8.184e-02 +/- 1.636e-03 | 2.728e-01 +/- 2.275e-02 | 4.248e-03 +/- 7.400e-05 |
| Case B | sparse x+t | neural+symbolic | 8.159e-02 +/- 1.645e-03 | 2.723e-01 +/- 2.293e-02 | 3.893e-03 +/- 9.876e-05 |
| Case Exp | clean | strong_poly | 4.609e-02 +/- 2.701e-04 | 4.270e-01 +/- 1.115e-01 | 3.146e-03 +/- 2.813e-04 |
| Case Exp | clean | weak_poly | 5.230e-02 +/- 1.786e-03 | 8.712e-02 +/- 1.598e-02 | 2.738e-03 +/- 8.345e-05 |
| Case Exp | clean | neural | 4.772e-02 +/- 7.372e-04 | 4.400e-01 +/- 8.150e-03 | 3.212e-03 +/- 7.913e-05 |
| Case Exp | clean | neural+symbolic | 4.765e-02 +/- 8.388e-04 | 4.400e-01 +/- 8.151e-03 | 3.226e-03 +/- 9.908e-05 |
| Case Exp | noise 5% | strong_poly | 9.684e-01 +/- 8.892e-03 | 1.247e+01 +/- 2.298e+00 | 1.083e-01 +/- 6.465e-03 |
| Case Exp | noise 5% | weak_poly | 6.647e-02 +/- 1.608e-02 | 6.303e-01 +/- 2.836e-01 | 3.805e-03 +/- 6.149e-04 |
| Case Exp | noise 5% | neural | 8.379e-01 +/- 3.695e-03 | 4.695e+00 +/- 2.016e-01 | 1.038e-01 +/- 2.468e-03 |
| Case Exp | noise 5% | neural+symbolic | 8.379e-01 +/- 3.696e-03 | 4.691e+00 +/- 2.000e-01 | 1.038e-01 +/- 2.461e-03 |
| Case Exp | sparse x+t | strong_poly | 5.186e-02 +/- 1.141e-03 | 3.793e-01 +/- 3.657e-02 | 4.007e-03 +/- 3.234e-04 |
| Case Exp | sparse x+t | weak_poly | 8.916e-02 +/- 8.157e-03 | 4.750e-01 +/- 1.302e-01 | 6.132e-03 +/- 9.614e-04 |
| Case Exp | sparse x+t | neural | 7.167e-02 +/- 5.174e-03 | 4.759e-01 +/- 3.409e-03 | 4.359e-03 +/- 2.494e-04 |
| Case Exp | sparse x+t | neural+symbolic | 7.156e-02 +/- 4.922e-03 | 4.759e-01 +/- 3.410e-03 | 4.345e-03 +/- 2.567e-04 |
