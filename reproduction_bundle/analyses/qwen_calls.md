# Calls of the frozen programs to Qwen3.6

Report: Section 3.4 (Models).

The table counts the programs' calls to Qwen3.6, all of them instruction-resolver calls, usually on the head image.

Calls of the frozen programs to Qwen3.6-35B-A3B: per-call traces of the platform jobs, and our local post-freeze runs of
the same versions. Some branches ask again: the block-number task on each wrist image (130 of the 1,059 calls of the
Instruction Following job), other branches on the head image, for example with the object category spelled out. In the
static call graph, four of the eight Spatial Reasoning task routines can reach the resolver, in the second version as in
the third, and no General Manipulation task routine of any version can. Model calls on the scoring side (the platform's
scoop judge, our local proxy judge) are not program calls and are not counted.

| Source | Track | Episodes | Episodes with a call | Calls |
|---|---|---|---|---|
| Platform job | Instruction Following | 907 | 907 | 1,059 |
| Platform job | Robustness | 885 | 885 | 1,095 |
| Platform job | Spatial Reasoning | 240 | 3 | 8 |
| Platform job | General Manipulation | 160 | 0 | 0 |
| Local post-freeze | Spatial Reasoning | 240 | 4 | 9 |
| Local post-freeze | General Manipulation | 160 | 0 | 0 |
