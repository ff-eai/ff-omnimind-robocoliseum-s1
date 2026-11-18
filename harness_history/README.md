# Harness history

`src/` is the harness at commit `7df1039`. Each `<commit>.patch` turns it into the harness of another build: the
files that the worker and the services load at that commit, with the same license lines and neutral text as
`src/`. The patches contain no program files.

| Commit | Builds | Files changed | Lines |
|---|---|---|---|
| `971554e` | Instruction Following v3 | 20 | +133 / -2,576 |
| `4d42d8f` | Instruction Following v3.2 | 20 | +133 / -2,576 |
| `2c45ad7` | Instruction Following v3.4, v3.5 | 14 | +79 / -1,184 |
| `689546a` | Instruction Following v3.6 | 12 | +74 / -1,069 |
| `4a9c8c5` | Instruction Following v3.7, v3.8, v3.9 | 11 | +60 / -963 |
| `69313ed` | Robustness v4.0 (comparison) | 11 | +69 / -426 |
| `9ae2dd6` | Robustness v4.0 | 10 | +58 / -414 |
| `c583d6f` | Spatial Reasoning v1, v2, v3 | 10 | +56 / -397 |
| `c33b239` | General Manipulation v1 | 1 | +8 / -32 |

Recreate a commit's harness in a new directory, outside any git repository, and check it against `SHA256SUMS`:

```bash
REL=$PWD                      # this repository
mkdir -p /tmp/h/c583d6f && cp -r src /tmp/h/c583d6f/
git -C /tmp/h/c583d6f apply "$REL/harness_history/c583d6f.patch"
cd /tmp/h && sha256sum -c --ignore-missing "$REL/harness_history/SHA256SUMS"
```

`SHA256SUMS` lists every file of every commit, `7df1039` included.

Builds v3 and v3.2 (`971554e`, `4d42d8f`) ran with the station profile in `station_v3_v3.2/`
(`--station-config-root harness_history/station_v3_v3.2`). It differs from the packaged `geniesim-default` profile
in four values: gripper `full_travel_s` 0.4, `final_joint_tolerance_rad` 0.20, `ws_port` 8997 and `keepalive_hold`
false.
