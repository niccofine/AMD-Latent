# Extended Representation Space Analysis
### Full RETCLIP representation space analysis
On the following, a full RETCLIP latent space representation analysis is presented, taking into account within-class variability and uncertainty.

95% CIs are reported for inter- and intra-class distances using patient-cluster bootstrap resampling
(2,000 replicates).
Cosine distance evalution is coupled with a Fisher-like discriminability ratio that normalizes between-
class separation by within-class scatter.
Class-conditional separation is further addressed using Mahalanobis distance, Fréchet distance, and MMD, with permutation testing for MMD.

| Pair | Cosine [95% CI] | FDRij [95% CI] | Mahalanobis | Fréchet | MMD [95% CI] | perm. *p* (BH) |
| --- | --- | --- | --- | --- | --- | --- |
| 0–1 | 0.0049 [0.0034, 0.0085] | 0.000 [0.000, 0.000] | 2.10 | 0.171 | 0.109 [0.093, 0.147] | <0.0005 |
| 1–2 | 0.0047 [0.0029, 0.0103] | 0.000 [0.000, 0.001] | 2.43 | 0.197 | 0.105 [0.082, 0.162] | <0.0005 |
| 2–3 | 0.0089 [0.0058, 0.0170] | 0.000 [0.000, 0.001] | 2.78 | 0.226 | 0.139 [0.111, 0.198] | <0.0005 |
| 3–4 | 0.0057 [0.0034, 0.0122] | 0.000 [0.000, 0.001] | 2.61 | 0.202 | 0.112 [0.088, 0.169] | <0.0005 |
| 4–5 | 0.0241 [0.0162, 0.0374] | 0.003 [0.001, 0.007] | 4.45 | 0.292 | 0.233 [0.195, 0.296] | <0.0005 |
| 0–2 | 0.0161 [0.0118, 0.0243] | 0.001 [0.001, 0.003] | 2.84 | 0.254 | 0.200 [0.171, 0.247] | <0.0005 |
| 1–3 | 0.0233 [0.0161, 0.0343] | 0.003 [0.001, 0.006] | 3.20 | 0.281 | 0.234 [0.198, 0.288] | <0.0005 |
| 2–4 | 0.0245 [0.0179, 0.0361] | 0.003 [0.001, 0.006] | 3.33 | 0.286 | 0.236 [0.200, 0.290] | <0.0005 |
| 3–5 | 0.0383 [0.0263, 0.0571] | 0.007 [0.003, 0.015] | 4.97 | 0.343 | 0.289 [0.243, 0.352] | <0.0005 |
| 0–3 | 0.0452 [0.0359, 0.0575] | 0.010 [0.006, 0.017] | 3.52 | 0.349 | 0.329 [0.294, 0.372] | <0.0005 |
| 1–4 | 0.0436 [0.0336, 0.0572] | 0.009 [0.006, 0.017] | 3.79 | 0.341 | 0.322 [0.283, 0.372] | <0.0005 |
| 2–5 | 0.0686 [0.0526, 0.0893] | 0.022 [0.013, 0.039] | 5.44 | 0.422 | 0.384 [0.339, 0.445] | <0.0005 |
| 0–4 | 0.0707 [0.0592, 0.0848] | 0.026 [0.018, 0.039] | 4.09 | 0.408 | 0.410 [0.375, 0.451] | <0.0005 |
| 1–5 | 0.0964 [0.0765, 0.1198] | 0.046 [0.028, 0.074] | 5.81 | 0.473 | 0.461 [0.411, 0.517] | <0.0005 |
| 0–5 | 0.1265 [0.1052, 0.1489] | 0.083 [0.056, 0.120] | 5.93 | 0.528 | 0.531 [0.488, 0.578] | <0.0005 |

**On the table above**: Full pairwise separation across all AREDS severity classes under four metrics, with 95% CIs where available. Mahalanobis and Fréchet are reported as point estimates: because both require re-estimating a near-singular class covariance on every bootstrap resample, their percentile CIs carry a known small-sample upward bias and are omitted. MMD requires no covariance estimation, shows no such bias, and serves as the primary CI-bearing robustness check alongside FDR.