# Execution status

Started 2026-09-25 22:00:06 UTC; hard planning deadline 2026-09-28 22:00:06 UTC (Monday 17:00 Chicago).

## Current stage

Data acquisition and real-video stack validation. Formal label generation/training has not started. User authorized autonomous execution, Git, and auxiliary Mac compute; keep implementation lean and stay on main.

## Jobs and running processes

- Lop root: `/home/shic4/SE-VLM3DQA-1`.
- `427516`: first CPU preparation failed on unavailable PyAV wheel. Preserved `reports/prepare-427516.log`; no model ran.
- `427517`: replacement CPU preparation. Pure-wheel dependencies installed; imports passed: torch 2.4.1+cu118, torchvision 0.19.1+cu118, transformers 4.51.3, peft 0.15.2. Downloading pinned Qwen2.5-VL-3B-Instruct (revision in protocol/report).
- `427518`: GPU smoke, depends on 427517; queue gpu1@compute-1-20.local, four GTX 980 devices, 2h walltime. Briefly user-held to inspect side-by-side reference leakage; corrected code uploaded and user hold released. Dependency hold is expected until preparation completes. No duplicate submission needed.
- Local source selective download: exec session 66330, log `reports/source-download.log`.
- Local MUGSQA main streaming extraction: exec session 16607, log `reports/target-download.log`.
- Hourly app heartbeat `se-vlm3dqa-72` continues this task; only meaningful changes should notify. Pause it at completion/deadline.

## Important verified findings

- Explicit-NeRF-QA archive has exactly 440 video members (~3.126GB), across 22 named scenes; the full zip is 21.88GB. Only videos are being extracted via HTTP ranges.
- **Source PVS includes the pristine reference on the left and degraded video on the right.** Checked bowl variants, materials and a 14-scene contact sheet. Paper §II-E confirms side-by-side subjective protocol: https://arxiv.org/html/2407.08165 . Model adapter MUST crop right half before resizing/tokenization, and record original-coordinate mapping. Smoke's two bowl videos were inspected. Complete all-scene layout checks before formal runs. No reference pixels have yet been passed to a model.
- 3DGS-QA public HF revision lists MOS and `.ply` only, no rendered videos. MUGSQA main archive has rendered `.mp4`. Favor MUGSQA as primary target if no ready 3DGS videos can be found; determine this before any model metrics. Do not introduce rendering.
- Mac auxiliary environment `.venv` installed binary numpy/scipy/Pillow/imageio-ffmpeg/openpyxl. Package list recorded. Reward mathematical checks pass (monotonicity, swap symmetry, zero-variance behavior, post-normalization scaling).
- Sample source: 1600×800, 30fps, 10sec; use right 800×800. Sample target: 1080×1080, 30fps, 6sec. Full inventory pending download completion.

## Next actions

1. Ensure latest `scripts/vlm.py` is synchronized before smoke starts (training-mode checkpointing tweak). Check qstat/qacct and logs, not submission status alone.
2. Finish source/target downloads. Run `.venv/bin/python scripts/inventory.py`; inspect remaining source layouts, audit near duplicates/content overlap. Source MOS has not been read. Target MOS downloaded to `data/sealed` only after archive completes; do not inspect values for selection.
3. Resolve smoke failures within first 6h; verify actual teacher output, nonconstant rollout reward, finite nonzero gradient/parameter delta, adapter reload error, GPU peak memory, elapsed times. No arbitrary/fake rewards to force a pass.
4. Use measured throughput to freeze data/pair/visual/agent/training sizes. Prefer T0–T3 + T5, three training seeds if feasible, one outer round, full mechanisms. Source has 22 independent groups; don't inflate evidence using many related pairs.
5. Implement remaining evaluator/train/test using shared lean adapter and reward. T2/T3/T5 must use identical labels, samples and schedule; reliability after advantage normalization; KL separate. Preserve common-coverage and full-output views. No target-MOS tuning.

## Access and environment

SSH multiplex socket `/private/tmp/sevlm3dqa-ssh` currently attached to exec session 50846; use if alive, otherwise reconnect using already-authorized session credentials (do not persist credentials in files). SGE env: SGE_ROOT=/opt/gridengine, SGE_CELL=default, SGE_QMASTER_PORT=536, SGE_EXECD_PORT=537. Binaries `/opt/gridengine/bin/lx-amd64`.

Reused base Python: `/home/shic4/.cache/rl_verified_agent_20260911/conda/bin/python`; set LD_LIBRARY_PATH to its `lib`, and PYTHONPATH to this project's `runtime_deps:scripts`. Do not modify the shared base or other experiments.

Mac project files/data are canonical; remote has scripts and two smoke source files so far. No Git remote/push. Do not rewrite frozen scientific configs based on target results.
