# Reproduction bundle: FF OmniMind technical report

This bundle accompanies the technical report *Write Once, Run with Zero Cloud Tokens: Code-as-Policy Skills Discovered
in Simulation by a Coding Agent*. It lists the frozen programs, harness commits and RoboColiseum jobs behind every
reported number, the timeline of the General Manipulation instances without development rollouts, the recount of the
platform's task scores, and further analyses and records behind the reported numbers (`analyses/`). Section, table and
figure numbers refer to the report.

| File | Content | Report |
|---|---|---|
| `builds.csv` | every build that enters a reported number: program md5, harness commit, code lines | Sections 3-5; Appendices A-F |
| `jobs.csv` | our scored RoboColiseum jobs up to the leaderboard snapshot of 2026-11-03 14:30 PST | Tables 1 and 2; Appendix A |
| `general_manipulation_timeline.md` | when the instances without development rollouts were fixed and first run, and when v2 and v3 were developed and frozen | Section 4.1; Appendix D |
| `recount/` | recount of five platform jobs from their logged substep results: method, figure and data | Section 4.2; Appendix D |
| `analyses/scene_grouped.md` | the paired comparisons of Table 9 with the trials grouped by scene | Section 4.1; Appendix C |
| `analyses/qwen_calls.md` | the frozen programs' calls to Qwen3.6 (platform jobs and local runs) and the static call graph | Section 3.4 |
| `analyses/benchmark_checks.md` | checks of the benchmark: scenes across jobs, arm start offsets, pour read as four substeps | Appendices A and D |
| `analyses/instances_without_rollouts.md` | what the first version's post-freeze records gave, and a filter mistake during the third version's development | Section 4.1; Appendix D |
| `analyses/robustness_job_harness.md` | how the Robustness job's harness differs from the comparison's, and its smoke test | Table 3; Appendix E |
| `analyses/cost_and_run_time.md` | session dates and bookkeeping, input tokens by type, the online agent's quoted figures, the pick episode's timing | Section 4.4; Appendix B |
| `analyses/leaderboard_notes.md` | margins to neighboring entries, and a second row named π0.5 | Tables 1 and 2; Appendix A |
| `analyses/spatial_two_runs.md` | the two runs of the third Spatial Reasoning version: the reported run, their mean, floor drops | Section 4.3; Appendix C |

## Program names in the report

The report names each program by its role or by what changed, not by its version number. This table gives the version
of each name; `builds.csv` gives each version's md5, harness commit and code lines, and `programs/README.md` of the code
release gives its file.

| Name in the report | Track | Version |
|---|---|---|
| a version before the aiming fix, and one after it (Table 3, row 1; Appendix E) | Instruction Following | v3, v3.2 |
| the change of Table 3, row 1: aiming at the object's footprint center | Instruction Following | v3.1 and later |
| the shape and object-type fix: the version with it, and the one before (Appendix C) | Instruction Following | v3.5, v3.4 |
| the programs of the first, second and third scored jobs (Appendix A) | Instruction Following | v3.6, v3.8, v3.9 |
| the grasp and choice fixes: the version with them, and the one before (Appendix C) | Instruction Following | v3.8, v3.7 |
| the upright hold: the fixed program, and the program before it (Table 3, row 3; Appendix C) | Instruction Following | v3.9, v3.8 |
| the final Instruction Following program | Instruction Following | v3.9 |
| the first Instruction Following programs (Appendix B, excluded sessions) | Instruction Following | the first generic program, v3 to v3.2 |
| the Robustness program | Robustness | v4.0 |
| the first, second and third Spatial Reasoning versions | Spatial Reasoning | v1, v2, v3 |
| the Spatial Reasoning version of the platform job | Spatial Reasoning | v2 |
| the first, second and third General Manipulation versions | General Manipulation | v1, v2, v3 |
| the General Manipulation version of the platform job | General Manipulation | v1 |

## Builds

A *build* is a frozen program version together with the harness commit it runs on and the setup of its self-hosted
services. The harness is the ASPIRE `cap_harness`. The G2 adapter is part of the harness repository, so a harness
commit also fixes the adapter. Each reported score comes from one build.

`builds.csv` columns: `track`, `version`, `program_md5_prefix` (first 8 hex digits of the md5 of the frozen program
file), `harness_commit` (abbreviated commit of the harness repository), `code_lines` (lines of the program file),
`reported_in` (where the report uses the build: `job N` is a platform job of `jobs.csv`; `paired comparison(s)`,
Appendix C; `aiming error`, Table 3 row 1 and Appendix E).

- For v3 to v3.5 the program file was kept in the harness repository, and the commit given is the one that added it,
  except v3.4: its file was added in `193ffa5`, and its paired comparison with v3.5 ran both programs on the harness
  of `2c45ad7`, which the table gives.
- All builds use the same Qwen3.6-35B-A3B-FP8 weights and SAM 3.
- Every local number comes from services on H200 GPUs, except the v3.8 / v3.7 comparison, which ran on a
  second machine with an RTX A6000 GPU.
- Version numbers: Instruction Following versions start at v3, because no earlier version enters a reported number.
  The Robustness program continues that numbering (v4.0 extends v3.9). Spatial Reasoning and General Manipulation are
  numbered from v1.
- In the Robustness comparison (Appendix C), v4.0 ran on the comparison harness (`69313ed`) and v3.9 on the harness
  of jobs 1699 and 1701 (`4a9c8c5`). Job 1702 ran v4.0 on `9ae2dd6`; `analyses/robustness_job_harness.md` gives how it differs from `69313ed`.
- Harness commit `7df1039`, on which General Manipulation v2 and v3 run, added the adapter's forward kinematics for
  hypothetical body configurations. The v2 pot carry uses it (Section 5.2), and the program file of v2 states that it
  needs this harness.

## Jobs

`jobs.csv` lists the scored RoboColiseum jobs of FF OmniMind up to the leaderboard snapshot of 2026-11-03 14:30 PST, the
snapshot of every rank in the report. Jobs submitted after that snapshot are outside the report.

Columns: `track`, `job`, `program_version`, `program_md5_prefix`, `harness_commit`, `completed` (date), `score` (as
reported, to five decimals), `rank` and `rows` (position and number of rows in the first leaderboard snapshot after
the row appeared), `rank_snapshot_pt` (that snapshot, Pacific time).

- In the snapshot of 2026-11-03 14:30 PST the best Instruction Following job (1701) was third of 56, the Robustness
  job (1702) first of 35, and FF OmniMind first of the 27 entries with all four tracks in the site's Overall ranking
  (0.8698).
- Four earlier jobs ended without a track score: a connection probe (1635) and three submissions of v3.5 (1671, 1675
  and 1687).
- Each track score in Table 1 comes from one job: 1701, 1702, 1710 and 1718.
