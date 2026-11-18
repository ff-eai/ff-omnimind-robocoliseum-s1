# FF Omnimind: RoboColiseum Season I

Code behind the technical report

*Write Once, Run with Zero Cloud Tokens: Code-as-Policy Skills Discovered in Simulation by a Coding Agent*.

FF Omnimind is the RoboColiseum Season I entry of FF EAI Robotics Ecosystem Inc.

| Path | Content |
|---|---|
| `programs/` | The frozen skill programs of the report's builds, byte-identical. `programs/README.md` lists their md5 and harness commits. |
| `src/cap_harness/` | The harness at commit `7df1039`: the ASPIRE `cap_harness` with the G2 adapter, limited to the files that the worker and the services load. It adds license lines and replaces internal paths and addresses in a few comments and settings. |
| `harness_history/` | The harness of every other build, as a patch on `src/`. |
| `patches/` | The GenieSim v3.2.0 patch that sends depth, used by builds v3 and v3.2. |
| `reproduction_bundle/` | The report's reproduction bundle: builds, jobs, timeline and recount. |
| `requirements-*.txt` | Package versions of the runs (Python 3.12.3). |

No model weights of our own: the services run Qwen3.6-35B-A3B-FP8, SAM 3 and cuRobo without modification.

## Run

1. **Services** (one GPU). Install `requirements-sam3.txt`, `requirements-curobo.txt` and `vllm==0.32.0` in one
   environment.
   - SAM 3: https://github.com/Max-Fu/sam3 at `6fe87d64a5beb9084923d7a9e002741178635b09` (`pip install -e`);
     checkpoint `facebook/sam3` at revision `3c879f39826c281e95690f02c7821c4de09afae7`.
   - cuRobo: https://github.com/NVlabs/curobo at `a35a708ecfbb26eb9ab2d7ef22c65919c4fae4a9` (`pip install -e . --no-deps`).

   ```bash
   vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --revision 95a723d08a9490559dae23d0cff1d9466213d989 \
     --served-model-name qwen36 --port 8300 --max-model-len 65536 --gpu-memory-utilization 0.5 \
     --limit-mm-per-prompt '{"image":4}' --reasoning-parser qwen3 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder --trust-remote-code
   PYTHONPATH=src python -m cap_harness.providers.sam3.service --host 127.0.0.1 --port 8114
   PYTHONPATH=src CAP_HARNESS_CUROBO_G2_STATION=geniesim-default \
     CAP_HARNESS_CUROBO_G2_LOCK_JOINTS='{"idx01_body_joint1": -0.834, "idx02_body_joint2": 1.217, "idx03_body_joint3": 0.100, "idx04_body_joint4": 0.0, "idx05_body_joint5": 0.0}' \
     python -m cap_harness.providers.curobo.service --host 127.0.0.1 --port 8118
   ```

2. **Worker** (CPU). Install `requirements-worker.txt`. Start one worker per simulator session; the platform jobs
   ran four, on ports 9101 to 9104.

   ```bash
   PYTHONPATH=src CAP_HARNESS_SAM3_URL=http://127.0.0.1:8114 CAP_HARNESS_CUROBO_URL=http://127.0.0.1:8118 \
     python -m cap_harness.cli serve --station geniesim-leaderboard \
       --program programs/general_manipulation/v2_98996298.py --output-root runs/w9101 \
       --host 127.0.0.1 --port 9101 --need-depth always
   ```

   The instruction resolver calls the Qwen3.6 service at `http://127.0.0.1:8300/v1`, model `qwen36`
   (`CAP_HARNESS_G2_VLM_URL`, `CAP_HARNESS_G2_VLM_MODEL`). Use `--host 0.0.0.0` if the simulator runs in a container.

3. **Simulator.** GenieSim (https://github.com/AgibotTech/genie_sim) at commit `6ca11c7`. Point its benchmark
   client at a worker, for example
   `geniesim benchmark run g2op_if_pick_block_color --infer-host=<worker host>:9101 --benchmark.num_episode=1 --benchmark.seed=1`.
   GenieSim judges *scoop* with a vision-language model (`BASE_URL`, `VL_MODEL`, `API_KEY`); our local runs used
   the Qwen3.6 service.

4. **Other builds.** Each program runs on its own harness commit (`programs/README.md`). `src/` is `7df1039`;
   `harness_history/README.md` recreates the others. Builds v3 and v3.2 ran one episode per process
   (`python -m cap_harness.cli run --benchmark g2_geniesim --station geniesim-default --station-config-root
   harness_history/station_v3_v3.2 ...`) on GenieSim v3.2.0 with `patches/geniesim_send_depth.patch` and
   `--benchmark.send_depth=true`.

## Licenses

- `LICENSE-FF.txt`: the files of FF EAI Robotics Ecosystem Inc. (programs, harness history, patches, reproduction
  bundle, documentation, the files under `src/` with FF's copyright line, and FF's changes to the files marked
  "Modified by FF EAI Robotics Ecosystem Inc.").
- `LICENSE` (Apache License 2.0) and `NOTICE`: the ASPIRE files of NVIDIA under `src/`.
- `src/cap_harness/g2/description/geniesim/`: the G2 description of GenieSim, under the Mozilla Public License 2.0.
