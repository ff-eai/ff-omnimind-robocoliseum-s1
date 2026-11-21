# Checks of how the benchmark behaves

Report: Appendix A (Table 4) and Appendix D (scoring).

## Instruction Following, the three scored jobs (Table 4)

The scenes appear fixed across jobs: two consecutive jobs received the same instruction at the same place in each
task's sequence of episodes, in all 22 scored episodes that we compared. In some episodes an arm started away from the
standard start pose. The offset is read at the simulator's reset, before the program acts. The second job had 1 such
episode of 907, against 37 of 905 scored episodes in the first, and we estimate that such starts cost the first job
about 0.008 of track score. The reset waits at most 1 s for the arms to reach the start pose. From the second job on,
the program first returns such an arm to the start pose.

## Pour (Table 7)

Read as four substeps, *pour* would give 0.4875 instead of 0.95 for the 20 episodes of the platform job.
