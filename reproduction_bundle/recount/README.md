# Recount of the platform's task scores

Report: Section 4.2.

## Method

For every task of jobs 1688, 1699, 1701, 1710 and 1718, the recount recomputes the task score. At the end of each
episode, our server logs the substep results that the platform's simulator reports. The recount combines these
results by our scoring rules (report, Section 4.1 and Appendix D). It therefore reuses the platform's judgments,
including those of its scoop judge, and judges nothing again.

- On Instruction Following, the platform's episode counter gives the first two episodes of each task the same index,
  and both are scored.
- For job 1718, *pour* counts as one substep.

## Result

With these rules the recount equals the platform score to within the platform's rounding (largest difference 0.0001)
for 46 of the 48 task scores. The other two are in job 1688 (billiard color 0.0002, straighten 0.0031), where one
episode each has no final result in our logs. Four of that job's tasks were run three times under an early per-task
time limit, and only the last run counts. Job 1702 was not recounted.

## Files

| File | Content |
|---|---|
| `fig_recount.pdf` | (a) Per-task score recounted from each platform job's logged substep results, against the platform score (five jobs, 48 task scores; inset: 0.82-1.0). (b) Per-task score of a separate local run of the same version, against the platform score (Spatial Reasoning v2, General Manipulation v1). |
| `recount_task_scores.csv` | panel (a): `job`, `track`, `task` (as in the report's tables), `task_id` (the benchmark's task name), `recount` (5 decimals), `platform` (as the platform reports it) |
| `local_vs_platform.csv` | panel (b): `track`, `version`, `platform_job`, `task`, `task_id`, `local` (the local run's task score, 4 decimals), `platform` |

Panel (b): Spatial Reasoning v2 ran locally on the same 240 instances as job 1710. Seven of its eight tasks agreed
within one episode (stacking three blocks: 27 of 30 on the platform, 30 of 30 locally). General Manipulation v1 ran
locally on the instances of job 1718, and its largest task difference was *scoop* (0.500 locally, 0.750 on the
platform), which a different model judges on each side.

The figure and both tables come from the report's figure script (`make_figures.py`, with `recount_instruction.py` for
the three Instruction Following jobs), which reads the substep results logged at the end of each episode. The
per-episode logs are not part of this bundle.
