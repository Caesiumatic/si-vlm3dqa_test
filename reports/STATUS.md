# Execution status

Start: 2026-09-25 22:00:06 UTC. Deadline: 2026-09-28 22:00:06 UTC (Monday 17:00 Chicago).
Updated during initial execution, ~22:41 UTC. Workdir `/Users/shichen/GitHub/SE-VLM3DQA-1`; Lop `/home/shic4/SE-VLM3DQA-1`. Main branch, no remote/push. Autonomous execution authorized; Mac auxiliary.

## Verified progress

- All 440 source videos (22 groups, 3.126 GB) and 1,970 MUGSQA main videos (55 groups, 2.234 GB) downloaded on Mac. No source MOS accessed. Target MOS sealed in `data/sealed/mos_main.xlsx`; no values used.
- All source groups visually checked: left reference/right degraded PVS. Model crops right half before processing. Original crop coordinates/timestamps preserved. Full source contact sheet in reports.
- Source train: 18 groups / 360 videos; dev: chair, phonograph, pizza, sofa / 80. Target pizza (33 videos) conservatively excluded. Target pool 54 groups / 1,937 videos. Similar-name piano/vase/chair/motorcycle subjects visually compared and differ; see audit images. No claim that names alone establish independence.
- Main target prospectively selected MUGSQA main: public 3DGS-QA HF release has no ready RGB videos. No target model metrics have been computed.
- Candidate minimum manifests: 180 source videos (10/group), 198 pairs, 432 target videos (8/group). Recommended candidate: 360 source, 378 pairs, 864 target. **Formal sizes not frozen.**
- Full video → teacher → student G=4 → nonzero gradient/update → adapter save/reload smoke passed on four GTX980: job427520, 119.1 seconds, max allocated2.45GiB/card, reload difference0. See reports/smoke-aligned.json.
- Exact segmented vision-attention implementation avoids masked cross-block allocations; forward/input-gradient agreement tested. Dispatch main device aligned with language embedding to fix Qwen cross-GPU cache position mismatch.
- Reward formula corrected before formal experiment to EvoQuality Eq.(2): both rollout-group variances + gamma. Initial engineering smoke used partner variance only, and is excluded from results. Numeric formula check added. See METHOD_NOTES.md.

## Job ledger

- 427516 failed CPU install: missing PyAV wheel. Preserved failure; replaced PyAV with imageio-ffmpeg.
- 427517 CPU preparation COMPLETE: model pinned revision66285546d2b821cf421d4f5eb2576359d3770cd3, 7.1GiB; isolated runtime imports passed.
- 427518 GPU FAILED global attention allocation/old NVML error.
- 427519 GPU FAILED cache position CUDA device mismatch; segmented attention check passed.
- 427520 GPU COMPLETE: full aligned smoke passed.
- 427521 dense local observation smoke COMPLETE (retrieve result); 8 global +16 native contiguous local frames at224px. Uses old engineering reward, not final formula.
- 427522 teacher development pilot RUNNING: two held-out pairs (chair, phonograph), baseline5 votes and full roles including conditional recursion. Log reports/teacher-pilot.log, outputs runs/teacher-pilot. Must inspect actual evidence/parse rates/calls/cost before freeze.
- Hourly app heartbeat `se-vlm3dqa-72` ACTIVE. Quiet unchanged; pause on completion/deadline. No new jobs after deadline.

## Next actions

1. Retrieve dense smoke and teacher pilot; inspect role outputs and actual recursion. Fix interfaces using only label-free dev evidence; do not tune to MOS. Record any engineering changes.
2. Test multi-pair/multi-step train.py with corrected reward and fixed-base KL, save/resume, real nonzero gradients. Current train.py/predict.py implemented but not run yet. Use batch >=2 pairs so reliability meaningfully changes relative gradient contributions under Adam. Limit checkpoint storage (current draft saves all batches; fix before formal).
3. Freeze numeric protocol using end-to-end costs. Primary common-coverage T0/T1/T2/T3/T5; three seeds goal. T2/T3/T5 share exact labels/schedule, weights globally normalized once; T5 permutes weights. No early selection on target metrics. Ensure training schedule covers selected scenes/videos at bounded pair budget. Clearly report actual training exposures/steps.
4. Complete all-target-group visual overlap review. Finalize and hash manifests/code/prompts/config, transfer selected data to Lop. Freeze labels and schedule, implement lightweight bundle creation/statistics and final metrics.
5. Submit bounded SGE label/train/test stages; inspect existing jobs before submission. Formal prediction outputs must be complete/frozen before target MOS metrics are read. Report raw PLCC, SRCC, paired content-group bootstrap intervals, coverage, weights/ESS, cost, failed/zero variance updates, limited claims.

## Access/environment

SSH multiplex socket `/private/tmp/sevlm3dqa-ssh`, interactive exec session50846. Reuse while alive; credentials authorized in original user context, never persist in files.
SGE: export SGE_ROOT=/opt/gridengine SGE_CELL=default SGE_QMASTER_PORT=536 SGE_EXECD_PORT=537; binaries /opt/gridengine/bin/lx-amd64.
Base Python `/home/shic4/.cache/rl_verified_agent_20260911/conda/bin/python`; LD_LIBRARY_PATH=base/lib, PYTHONPATH=project/runtime_deps:project/scripts. Base untouched. Project transformers4.51.3, torch2.4.1cu118, peft0.15.2.
GPU script: cuda PE4, slots_gpu1 (four GPUs), gpu1@compute-1-20.local. Default2h walltime; override only within actual queue limits/deadline. Latest code staged remotely; Mac files canonical. Remote currently only six development videos; most data on Mac.

## Latest continuation (~22:55 UTC)

- Dense smoke427521 confirmed passed:154.85sec, max allocated2.49GiB/card, exact reload.
- All55 target groups visually reviewed; no further apparent shared subjects. MOS file header and **name column only** audited:1,970 unique keys,432/432 candidate test names match; MOS numerical cells not consumed. Mapping report saved.
- Initial teacher427522 COMPLETE: all4 method/pair labels abstained primarily due key-name errors. Full raw outputs retained under reports/teacher-pilot.
- V2 teacher427523 RUNNING. First pair: baseline valid JSON but 2–2–1 disagreement → abstain; full near_equal,8 calls146sec with local recursion. Explicit schema and missing-status normalization fixed before this job. Preserve this as dev evidence; don't merge with formal labels.
- Train pilot427524 queued after427523: two-pair accumulated update with corrected Eq.(2) reward and fixed-base KL, then budget exit/save. Resume pilot427525 queued after427524: reload optimizer+adapter and perform second update. Outputs runs/train-pilot, logs reports/train-pilot-{1,2}.log. Uses repeated real-smoke teacher label only for engineering; never count as scientific training. Inspect grad_norm, second-update KL, memory, checkpoint and batch logs.
- Candidate minimum data transfer COMPLETE (1.877GB,615files):180 source +432 target and manifests now onLop in original relative paths.
- Draft numerical config configs/experiment-draft.json; not frozen. scripts/pipeline.py implements serial resumable label→bundle→T0→train/test across arms/seeds and never submits jobs. Not submitted yet. scripts/metrics.py implemented but not run; exact full prediction matrix required before MOS opening. Add deadline/partial-matrix reporting path if needed, without concealing unfinished arms.
- Checkpoint storage fixed: only latest two optimizer/adapter snapshots retained per run; all batch logs kept. Deadline buffer stops new model work30min before72hdeadline. Current queue permits h_rt=INFINITY; request a bounded duration for main pipeline within remaining deadline.
- Next: finish/check current pilot jobs; inspect real role records and training/resume; freeze final numeric config as configs/experiment.json (status FROZEN), manifests as data/manifests/frozen, source/code hashes and Git commit. Then upload latest scripts/configs and submit main pipeline once via gpu.sge with suitable bounded walltime, e.g.≤69h remaining. Record jobid. Main pipeline requires completed train-pilot status and exact frozen config. Do not run or select using target MOS before predictions freeze.
