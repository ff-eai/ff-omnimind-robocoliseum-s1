# Scene-grouped analysis of the paired comparisons

Report: Section 4.1 (statistics) and Appendix C (Table 9).

Grouping the trials by scene changes no conclusion (table below). No interval changes whether it includes zero, and
every test gives the same result at the 0.05 level, with equal or larger *p*. A seed draws one instance and runs all its
instructions, so the trials of one scene share its layout. In 20 of the 50 configurations only one scene was run, so
Robustness scenes are also drawn from the pool of all scenes.

Trial-level (as in Table 9) and scene-grouped analyses of the comparisons whose trials share scenes: 90 % bootstrap
intervals of the track-score difference, and B / W (trials or scenes on which the newer version is better / worse) with
the two-sided sign test. A scene is a task and an instance. On Robustness it is either a perturbed scene (generator,
task and instance; 80 scenes) or a base scene (task and instance; 20 scenes, two per task, each run under three or five
generators). A scene counts as better when more of its trials have full credit only with the new version. Robustness:
perturbed scenes drawn within each configuration, then perturbed and base scenes drawn from the pool of all scenes
(wider intervals in these data).

| Comparison | Trials | Scenes | Trials: interval | B / W, *p* | Scenes: interval | B / W, *p* |
|---|---|---|---|---|---|---|
| Instruction Following: shape and object-type fix | 153 | 21 | [+0.2110, +0.3301] | 48 / 8, 4.7×10^-8 | [+0.2166, +0.3191] | 16 / 1, 2.7×10^-4 |
| Instruction Following: grasp and choice fixes, all tasks | 361 | 90 | [-0.0050, +0.0166] | 7 / 4, 0.55 | [-0.0052, +0.0172] | 7 / 4, 0.55 |
| Instruction Following: upright hold, 4-s delay | 40 | 29 | [+0.1000, +0.2500] | 15 / 1, 5.2×10^-4 | [+0.0952, +0.2500] | 11 / 1, 0.0063 |
| Robustness: self-calibration and other fixes; perturbed scenes, within configuration | 278 | 80 | [+0.0445, +0.0812] | 30 / 4, 6.2×10^-6 | [+0.0460, +0.0775] | 18 / 3, 0.0015 |
| Robustness: perturbed scenes, pooled | 278 | 80 | | | [+0.0293, +0.0995] | 18 / 3, 0.0015 |
| Robustness: base scenes, pooled | 278 | 20 | | | [+0.0287, +0.1013] | 13 / 2, 0.0074 |
