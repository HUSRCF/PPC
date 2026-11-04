# M9-M21 Historical Validation Run Catalog

Collected from Star on 2026-11-04. This contains **55 training runs and 7 eval-only exports**, not 62 independent trained models.

- All selected records cover 3,796 historical validation chains. These use ESMFold features and original context-specific labels, not the November exact-sequence-union/OpenFold3 labels.
- A training row reports the first maximum of its configured validation selection metric. For the three M21c head-only runs that metric is `neg_chain_ratio_head_mae`, not residue AP.
- `AP` below is pooled average precision, historically stored as `pr_auc`; it is **not trapezoidal AUPRC**.
- F1 and MCC share the threshold selected by F1 on that validation record. They are validation descriptions, not held-out test performance.
- `epochs` means largest training epoch recorded / configured budget; it is not a live Slurm-status check.
- Eval-only records with epoch 0 are prediction exports, not training from scratch. Do not infer the source checkpoint epoch from that 0; follow `config.init_checkpoint` in the JSON.
- Test metrics were absent from all 55 training runs' selected records. Six eval-only exports included test metrics; this catalog deliberately does not use their test-oracle gates to rank models.
- Source paths, config hashes, metric-log hashes and model configuration are in [JSON](star_m9_m21_runs.json); machine-readable values are in [CSV](star_m9_m21_validation.csv).
- This is an exhaustive export of existing config+metrics files matching the M9-M21 run-name pattern on Star at collection time, not proof that every experiment ever attempted on every host is retained.

| Run (without contact_site_cfsi30_ prefix) | Kind | Seed | Epochs | Selected epoch | Val macro AP | Val pooled AP | Val AUROC | Val F1@gate | Val MCC | Residue-mean ratio MAE | Head MAE |
|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| `m9star_m9a_tcn2_ratio1_seed42_e30_20260713_star` | train | 42 | 30/30 | 20 | 0.738834 | 0.762496 | 0.866930 | 0.690711@0.420000 | 0.564580 | 0.055958 | - |
| `m9star_m9b_tcn2_ratio5_seed42_e30_20260713_star` | train | 42 | 30/30 | 26 | 0.735475 | 0.750904 | 0.862299 | 0.685524@0.460000 | 0.557160 | 0.058193 | - |
| `m9star_m9c_tcn2_lr3e5_seed42_e30_20260713_star` | train | 42 | 30/30 | 24 | 0.735577 | 0.753861 | 0.869321 | 0.694133@0.730000 | 0.567293 | 0.138757 | - |
| `m9star_m9d_tcn2_lr8e5_seed42_e30_20260713_star` | train | 42 | 30/30 | 15 | 0.739284 | 0.762625 | 0.869061 | 0.695280@0.670000 | 0.571756 | 0.095242 | - |
| `m9star_m9e_tcn2_kernel9_seed42_e30_20260713_star` | train | 42 | 30/30 | 15 | 0.732070 | 0.754144 | 0.865610 | 0.685680@0.620000 | 0.556299 | 0.104827 | - |
| `m9star_m9f_tcn2_resmean_drop035_seed42_e30_20260713_star` | train | 42 | 30/30 | 19 | 0.738335 | 0.755392 | 0.867649 | 0.687420@0.670000 | 0.556835 | 0.120120 | - |
| `m10star_m10a_lr8e5_ratio025_seed42_e30_20260713_star` | train | 42 | 30/30 | 15 | 0.740132 | 0.758576 | 0.872150 | 0.694639@0.800000 | 0.566175 | 0.175548 | - |
| `m10star_m10b_lr8e5_ratio05_seed42_e30_20260713_star` | train | 42 | 30/30 | 13 | 0.741068 | 0.761876 | 0.871145 | 0.695451@0.620000 | 0.570509 | 0.100849 | - |
| `m10star_m10c_lr65e5_ratio025_seed42_e30_20260713_star` | train | 42 | 30/30 | 18 | 0.740289 | 0.766156 | 0.871020 | 0.694413@0.720000 | 0.571826 | 0.105958 | - |
| `m10star_m10d_lr1e4_ratio025_seed42_e30_20260713_star` | train | 42 | 30/30 | 12 | 0.737481 | 0.761904 | 0.872314 | 0.695565@0.680000 | 0.570966 | 0.122660 | - |
| `m10star_m10e_lr8e5_ratio025_drop040_seed42_e30_20260713_star` | train | 42 | 30/30 | 8 | 0.737277 | 0.756288 | 0.862018 | 0.683068@0.350000 | 0.554591 | 0.051578 | - |
| `m10star_m10f_lr8e5_ratio075_seed42_e30_20260713_star` | train | 42 | 30/30 | 19 | 0.741364 | 0.766602 | 0.870856 | 0.691719@0.750000 | 0.569250 | 0.107329 | - |
| `m10star_m10f_lr8e5_ratio075_seed43_e30_20260713_star` | train | 43 | 30/30 | 14 | 0.740139 | 0.755527 | 0.862456 | 0.682465@0.600000 | 0.550603 | 0.105579 | - |
| `m10star_m10f_lr8e5_ratio075_seed44_e30_20260713_star` | train | 44 | 30/30 | 13 | 0.752163 | 0.767181 | 0.869703 | 0.690168@0.490000 | 0.563921 | 0.064779 | - |
| `m11star_m11a_dualgated_tcn2_ratio075_seed42_e30_20260713_star` | train | 42 | 30/30 | 5 | 0.736365 | 0.760167 | 0.869589 | 0.685444@0.360000 | 0.553735 | 0.051379 | - |
| `m11star_m11b_esmcbase_addesm2_gated_tcn2_ratio075_seed42_e30_20260714_star` | train | 42 | 30/30 | 13 | 0.737820 | 0.753189 | 0.868449 | 0.683395@0.700000 | 0.552276 | 0.115151 | - |
| `m11star_m11c_esmcbase_addesm2_simplegate_tcn2_ratio075_seed42_e30_20260714_star` | train | 42 | 30/30 | 10 | 0.733384 | 0.753142 | 0.866496 | 0.680150@0.600000 | 0.543947 | 0.113550 | - |
| `m12star_m12a_convexmix_tcn2_ratio075_seed42_e30_20260714_star` | train | 42 | 30/30 | 5 | 0.727832 | 0.747895 | 0.861888 | 0.675095@0.330000 | 0.536479 | 0.064219 | - |
| `m12star_m12b_symbilinear_r64_tcn2_ratio075_seed42_e30_20260714_star` | train | 42 | 30/30 | 5 | 0.733444 | 0.748874 | 0.860392 | 0.677508@0.420000 | 0.547361 | 0.068439 | - |
| `m13star_m13a_esmc_lr65e5_ratio025_seed44_e30_20260714_star` | train | 44 | 30/30 | 15 | 0.744145 | 0.751793 | 0.854695 | 0.675817@0.330000 | 0.540227 | 0.060532 | - |
| `m13star_m13b_projectedconcat_d512_tcn2_ratio075_seed42_e30_20260714_star` | train | 42 | 30/30 | 27 | 0.743132 | 0.761550 | 0.871737 | 0.689821@0.470000 | 0.561811 | 0.067567 | - |
| `m13star_m13c_m10f_lr8e5_ratio075_seed45_e30_20260714_star` | train | 45 | 30/30 | 13 | 0.742099 | 0.761472 | 0.871513 | 0.693385@0.620000 | 0.564582 | 0.120793 | - |
| `m14star_m14b_frozen_logitres_r64_seed44_e20_20260714_star` | train | 44 | 20/20 | 1 | 0.742447 | 0.760290 | 0.871372 | 0.691343@0.500000 | 0.564962 | 0.075038 | - |
| `m14star_m14c_frozen_precontactres_r64_seed44_e20_20260714_star` | train | 44 | 20/20 | 1 | 0.738597 | 0.757287 | 0.867748 | 0.685229@0.460000 | 0.553661 | 0.073684 | - |
| `m15star_m15a_projectedconcat_d512_tcn2_ratio075_seed44_e30_20260715_star` | train | 44 | 30/30 | 15 | 0.740578 | 0.761218 | 0.868764 | 0.677759@0.350000 | 0.541998 | 0.053020 | - |
| `m15star_m15b_tcn2_k5_ratio075_seed42_e30_20260715_star` | train | 42 | 30/30 | 13 | 0.740512 | 0.757899 | 0.860551 | 0.679301@0.300000 | 0.543471 | 0.057815 | - |
| `m15star_m15c_tcn3_k7_ratio075_seed42_e30_20260715_star` | train | 42 | 30/30 | 16 | 0.741996 | 0.761602 | 0.867636 | 0.693127@0.540000 | 0.568188 | 0.076365 | - |
| `m15star_m15d_graph2_ratio075_seed42_e30_20260715_star` | train | 42 | 30/30 | 10 | 0.736252 | 0.762023 | 0.865641 | 0.685933@0.370000 | 0.558401 | 0.053064 | - |
| `m15star_m15e_bestchainap_testonly_20260715_star` | eval-only | 44 | -/0 | 0 | 0.751985 | 0.766997 | 0.869608 | 0.690174@0.490000 | 0.563990 | 0.065252 | - |
| `m15star_m15e_bestchainap_testonly_withtest_20260715_star` | eval-only | 44 | -/0 | 0 | 0.751985 | 0.766997 | 0.869608 | 0.690174@0.490000 | 0.563990 | 0.065252 | - |
| `m15star_m15e_frozen_logitres_r64_lr1e5_seed44_e20_20260715_star` | train | 44 | 20/20 | 1 | 0.751985 | 0.766997 | 0.869608 | 0.690174@0.490000 | 0.563990 | 0.065252 | - |
| `m16star_m16a_m10f_ratio075_seed46_e30_20260715_star` | train | 46 | 30/30 | 15 | 0.740061 | 0.760205 | 0.870145 | 0.691299@0.680000 | 0.563612 | 0.113120 | - |
| `m16star_m16b_classifierdrop035_ratio075_seed42_e30_20260715_star` | train | 42 | 30/30 | 18 | 0.735525 | 0.760757 | 0.863648 | 0.687528@0.430000 | 0.557505 | 0.067108 | - |
| `m16star_m16c_labelsmoothing002_ratio075_seed42_e30_20260715_star` | train | 42 | 30/30 | 15 | 0.738451 | 0.762968 | 0.867458 | 0.687751@0.440000 | 0.560792 | 0.052562 | - |
| `m16star_m16d_weightdecay005_ratio075_seed42_e30_20260715_star` | train | 42 | 30/30 | 16 | 0.740921 | 0.760202 | 0.865424 | 0.688656@0.490000 | 0.561397 | 0.063727 | - |
| `m16star_m16e_modeldrop020_ratio075_seed42_e30_20260715_star` | train | 42 | 30/30 | 14 | 0.737163 | 0.759622 | 0.863847 | 0.684715@0.440000 | 0.561378 | 0.051491 | - |
| `m17star_m17a_m10f_ratio075_seed47_e30_20260715_star` | train | 47 | 30/30 | 12 | 0.739843 | 0.754549 | 0.866897 | 0.684218@0.590000 | 0.555475 | 0.087768 | - |
| `m17star_m17b_m10f_ratio075_seed48_e30_20260715_star` | train | 48 | 30/30 | 14 | 0.744465 | 0.761741 | 0.868907 | 0.686688@0.640000 | 0.556662 | 0.097069 | - |
| `m18star_m18a_m10f_ratio075_seed49_e30_20260715_star` | train | 49 | 30/30 | 12 | 0.745166 | 0.767572 | 0.872959 | 0.693295@0.560000 | 0.565855 | 0.086536 | - |
| `m18star_m18b_m10f_ratio075_seed50_e30_20260715_star` | train | 50 | 30/30 | 30 | 0.744909 | 0.762281 | 0.866349 | 0.691501@0.520000 | 0.570101 | 0.051342 | - |
| `m19star_m19a_m10f_ratio0625_seed42_e30_20260716_star` | train | 42 | 30/30 | 18 | 0.745120 | 0.759988 | 0.868262 | 0.692135@0.490000 | 0.567393 | 0.066539 | - |
| `m19star_m19a_ratio0625_bestchainap_testonly_20260716_star` | eval-only | 42 | -/0 | 0 | 0.745120 | 0.759988 | 0.868262 | 0.692135@0.490000 | 0.567393 | 0.066539 | - |
| `m19star_m19b_m10f_ratio0875_seed42_e30_20260716_star` | train | 42 | 30/30 | 12 | 0.738445 | 0.759081 | 0.865151 | 0.688011@0.500000 | 0.559526 | 0.069400 | - |
| `m20confirm_m20a_tcn3_ratio0625_seed43_e30_20260717_star` | train | 43 | 30/30 | 11 | 0.734997 | 0.752421 | 0.860863 | 0.677239@0.370000 | 0.546063 | 0.056828 | - |
| `m20confirm_m20a_tcn3_ratio0625_seed44_e30_20260717_star` | train | 44 | 30/30 | 15 | 0.751053 | 0.776907 | 0.876918 | 0.701853@0.480000 | 0.580247 | 0.075250 | - |
| `m20confirm_m20c_m10f_ratio06875_seed43_e30_20260717_star` | train | 43 | 30/30 | 17 | 0.732588 | 0.754015 | 0.867296 | 0.689051@0.590000 | 0.559245 | 0.106001 | - |
| `m20confirm_m20c_m10f_ratio06875_seed44_e30_20260717_star` | train | 44 | 30/30 | 15 | 0.743805 | 0.762159 | 0.868621 | 0.688468@0.440000 | 0.557957 | 0.074560 | - |
| `m20ratioselect_m20a_seed42_bestratio_testonly_20260717_star` | eval-only | 42 | -/0 | 0 | 0.729871 | 0.752184 | 0.858644 | 0.681377@0.360000 | 0.549604 | 0.049241 | - |
| `m20ratioselect_m20a_tcn3_ratio0625_seed42_e30_20260717_star` | train | 42 | 30/30 | 17 | 0.739920 | 0.755361 | 0.868359 | 0.690598@0.690000 | 0.562876 | 0.095461 | - |
| `m20star_m20a_tcn3_ratio0625_seed42_e30_20260716_star` | train | 42 | 30/30 | 12 | 0.738023 | 0.757298 | 0.860926 | 0.683618@0.400000 | 0.555244 | 0.050756 | - |
| `m20star_m20b_m10f_ratio05625_seed42_e30_20260716_star` | train | 42 | 30/30 | 15 | 0.734403 | 0.759826 | 0.868732 | 0.693029@0.570000 | 0.567908 | 0.080965 | - |
| `m20star_m20c_m10f_ratio06875_seed42_e30_20260716_star` | train | 42 | 30/30 | 16 | 0.740178 | 0.758364 | 0.863610 | 0.681773@0.450000 | 0.546766 | 0.074399 | - |
| `m21c_locked_test_seed42_20260718_star` | eval-only | 42 | -/0 | 0 | 0.737981 | 0.766320 | 0.867293 | 0.696072@0.580000 | 0.575730 | 0.075196 | 0.047686 |
| `m21c_locked_test_seed43_20260718_star` | eval-only | 43 | -/0 | 0 | 0.732968 | 0.757247 | 0.859102 | 0.682891@0.420000 | 0.552550 | 0.061886 | 0.049370 |
| `m21c_locked_test_seed44_20260718_star` | eval-only | 44 | -/0 | 0 | 0.745334 | 0.762175 | 0.868864 | 0.688769@0.390000 | 0.557819 | 0.065661 | 0.051618 |
| `m21confirm_m21a_chainratio_detached_seed43_e30_20260718_star` | train | 43 | 30/30 | 13 | 0.732968 | 0.757247 | 0.859102 | 0.682891@0.420000 | 0.552550 | 0.061886 | 0.069228 |
| `m21confirm_m21a_chainratio_detached_seed44_e30_20260718_star` | train | 44 | 30/30 | 10 | 0.745334 | 0.762175 | 0.868864 | 0.688769@0.390000 | 0.557819 | 0.065661 | 0.071769 |
| `m21cstar_m21c_headonly_seed42_e12_20260718_star` | train | 42 | 12/12 | 12 | 0.737981 | 0.766320 | 0.867293 | 0.696072@0.580000 | 0.575730 | 0.075196 | 0.047686 |
| `m21cstar_m21c_headonly_seed43_e12_20260718_star` | train | 43 | 12/12 | 8 | 0.732968 | 0.757247 | 0.859102 | 0.682891@0.420000 | 0.552550 | 0.061886 | 0.049370 |
| `m21cstar_m21c_headonly_seed44_e12_20260718_star` | train | 44 | 12/12 | 11 | 0.745334 | 0.762175 | 0.868864 | 0.688769@0.390000 | 0.557819 | 0.065661 | 0.051618 |
| `m21star_m21a_chainratio_detached_seed42_e30_20260717_star` | train | 42 | 30/30 | 15 | 0.737981 | 0.766320 | 0.867293 | 0.696072@0.580000 | 0.575730 | 0.075196 | 0.053962 |
| `m21star_m21b_chainratio_joint_seed42_e30_20260717_star` | train | 42 | 30/30 | 14 | 0.739837 | 0.752227 | 0.868658 | 0.686297@0.770000 | 0.556522 | 0.147589 | 0.078624 |
