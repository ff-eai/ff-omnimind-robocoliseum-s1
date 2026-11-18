# Frozen programs

The 15 program files of the report's 16 builds, byte-identical to the files the builds
ran, named `<track>/<version>_<md5 prefix>.py`. Robustness v4.0 is one file that ran on two harness commits. The
table follows `reproduction_bundle/builds.csv` and adds the full md5.

| Track | Version | File | md5 | Prefix | Lines | Harness | Used in the report |
|---|---|---|---|---|---|---|---|
| Instruction Following | v3 | `instruction_following/v3_bd3ea1ff.py` | `bd3ea1fff160c30e01e0c87f05876deb` | `bd3ea1ff` | 1,166 | `971554e` | aiming error |
| Instruction Following | v3.2 | `instruction_following/v3.2_69162577.py` | `69162577047347452abfe2e786a60d4e` | `69162577` | 1,372 | `4d42d8f` | aiming error |
| Instruction Following | v3.4 | `instruction_following/v3.4_6988deb5.py` | `6988deb59423ffa0269e88a1ca6bf24c` | `6988deb5` | 1,997 | `2c45ad7` | paired comparison |
| Instruction Following | v3.5 | `instruction_following/v3.5_f8e44068.py` | `f8e44068795f2333ba84e8a397fad98d` | `f8e44068` | 2,365 | `2c45ad7` | paired comparison |
| Instruction Following | v3.6 | `instruction_following/v3.6_f937a2ae.py` | `f937a2aeb30082cfbe2e73e5b9d9f195` | `f937a2ae` | 2,594 | `689546a` | job 1688 |
| Instruction Following | v3.7 | `instruction_following/v3.7_21f9438a.py` | `21f9438ad54788e5c5855a58d75beca9` | `21f9438a` | 3,026 | `4a9c8c5` | paired comparison |
| Instruction Following | v3.8 | `instruction_following/v3.8_bfccdf95.py` | `bfccdf95e4bf55fe48dd9d28eca2843d` | `bfccdf95` | 3,393 | `4a9c8c5` | job 1699; paired comparisons |
| Instruction Following | v3.9 | `instruction_following/v3.9_504ba720.py` | `504ba72098304fd3f91c5b340037aa29` | `504ba720` | 3,596 | `4a9c8c5` | job 1701; paired comparisons; Section 5.1 |
| Robustness | v4.0 | `robustness/v4.0_31db9182.py` | `31db9182d42d1af598fa9acd592049e2` | `31db9182` | 3,852 | `9ae2dd6` | job 1702 |
| Robustness | v4.0 (comparison) | `robustness/v4.0_31db9182.py` | `31db9182d42d1af598fa9acd592049e2` | `31db9182` | 3,852 | `69313ed` | paired comparison; Figure 5 |
| Spatial Reasoning | v1 | `spatial_reasoning/v1_561a90df.py` | `561a90dfb681d15fc2cb97dbde7c5a4e` | `561a90df` | 3,427 | `c583d6f` | paired comparison |
| Spatial Reasoning | v2 | `spatial_reasoning/v2_d56ba517.py` | `d56ba517030304c10c936c34cf20d785` | `d56ba517` | 3,829 | `c583d6f` | job 1710; paired comparisons |
| Spatial Reasoning | v3 | `spatial_reasoning/v3_5a479bbd.py` | `5a479bbd27c2b9f16e20b10e9e5f4e13` | `5a479bbd` | 4,394 | `c583d6f` | paired comparisons |
| General Manipulation | v1 | `general_manipulation/v1_fcde3870.py` | `fcde3870ac448dd9bf9e2dd0d93a0459` | `fcde3870` | 4,835 | `c33b239` | job 1718; paired comparisons; Section 5.2 |
| General Manipulation | v2 | `general_manipulation/v2_98996298.py` | `98996298272e3a9ecd3c7c07fedbc37b` | `98996298` | 5,643 | `7df1039` | paired comparisons; Section 5.2 |
| General Manipulation | v3 | `general_manipulation/v3_f8859684.py` | `f8859684d1d2c198d7b329cab25427e7` | `f8859684` | 5,836 | `7df1039` | paired comparisons |

Used in the report: `job N` is a platform job (`reproduction_bundle/jobs.csv`); paired comparisons are in
Appendix C; the aiming error is Table 3, row 1, and Appendix E. A program runs on its harness commit: `src/` is
`7df1039`, and `harness_history/` recreates the others.
