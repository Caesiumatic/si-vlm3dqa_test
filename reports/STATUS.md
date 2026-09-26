# Execution status

Updated2026-09-26 12:43 UTC. Hard deadline2026-09-28 22:00:06 UTC (Monday17:00 Chicago). Local `/Users/shichen/GitHub/SE-VLM3DQA-1`; remote `/home/shic4/SE-VLM3DQA-1`. Main branch only. User authorizes autonomous execution; Lop primary, Mac auxiliary. No MOS values read for development.

## Current stage: T1 seed17 training

**Main SGE job427527 is RUNNING since2026-09-26 00:00:27 UTC. Supervision and432 T0 predictions completed; `STAGE train.py` for T1 seed17, gradients and checkpoints are verified.** Remote persistent receipt: `reports/main-job-id.txt`. Inspect qstat/logs before any new submission; never duplicate. Queue gpu1@compute-1-20.local, cuda PE4, four GTX980,69h walltime. Main log `reports/main-pipeline.log`; outputs `runs/main`.

The earlier automatic approval HTTP401 outage is RESOLVED. SSH checks and submission succeeded. No user reauthorization needed.

## Latest operational snapshot

11:42 UTC: original job427527 remains running. Supervision finished198/198 pairs for both methods (396 records) at2026-09-26 11:33:47 UTC. All198/198 full audits have valid numbered interfaces;37 exercised local recursion. Baseline153 valid/45 abstained; full197 valid/1 abstained. Common-valid152 pairs, spanning165 videos/18 groups;33 common labels differ. No human score has been read; these labels do not establish teacher accuracy.

Frozen schedule:108 unique pairs,165 videos/18 source groups, no repeats; weight mean1, range0.520–1.248, ESS102.44/108. T5 preserves the exact weight distribution. Both teacher labels and shuffled weights were checked against original records. Bundle SHA256 `ff7d80df01a31eb7f4f12a1cfb5861080116808c0d05b6c95fd954410a3bd71d` matches remote; retained as reports/frozen-label-bundle.json.

T0 predictions complete432/432, all valid with matching video/config hashes and no adapter. Completed at2026-09-26T12:18:23+00:00; no MOS read and no quality metrics computed.

12:43 UTC: T1 seed17 training active, 8/108 pair presentations, 2/27 updates saved. First two batches average500.2sec/4 pairs; gradient norms0.111 and0.145, nonzero fixed-base KL after the first update. All8 pairs processed,0/16 zero-variance rollout groups. Estimated current training remaining3.47h; following prediction roughly0.73h. Largest recorded GPU allocation2.49GiB. No stage failure or new job submission.

Early throughput extrapolation (two batches, excludes future startup overhead): first seed all four arms finishes about17.7h from this check; second seed about35.6h; all three seeds about53.5h, around2026-09-28T18:13:39.816904+00:00. Timing remains provisional and will be revised using more completed batches. Retain hourly monitoring and frozen scientific configuration.

## Frozen protocol

Commit `5971b83`; `configs/experiment.json` status FROZEN, interface numbered_audit_verdicts_v3. `reports/frozen-manifest.json` records exact code/config/data-manifest hashes; all635 checks (including612 videos) passed remotely. Do not silently change frozen core code during production.

- Source180 videos across18 groups,198 candidate pairs; source PVS right-half only, no reference pixels. Four separate development groups.
- Target MUGSQA main,432 videos across54 groups; pizza excluded conservatively. All groups visually audited,432/432 name-only MOS mappings match. Human numeric values remain sealed.
- Same pinned Qwen2.5-VL-3B teacher/student.224px,8 global +16 native consecutive local frames. Bilateral evidence search, numbered video audit, up to2 recursion depths,3 hyperjudges; baseline5 independent votes.
- Frozen labels → common valid schedule108 pair presentations,27 batches of4, G4/video. T1/T2/T3/T5 × seeds17/29/43. T2/T3/T5 identical labels/schedule; weights normalized globally; T5 shuffled distribution. LoRA r8/alpha16 q/v; lr1e-5; fixed-base KL.04.
- T0 plus12 final students, same greedy single-video test. Prediction/config/video/adapter hashes checked; incomplete prediction stages fail resumably, never mark done.
- Main contrast T3−T1 SRCC; raw PLCC and other contrasts. Group/seed paired bootstrap2000; no best-checkpoint selection.27 updates is a bounded pilot, not demonstrated convergence. Full-output student views, T4 and matched-compute/observation controls are extensions; main result alone cannot isolate recursion's effect.
- Details: `RUN_PROTOCOL.md`. Larger candidate exceeds conservative four-GTX980 budget. Main estimated57–65h from first measurements, including teacher, train and test; actual cost may differ. Model work stops30min before deadline.

## Verification and job ledger

| Job | Verified state/result |
|---|---|
|427516–427519|Earlier engineering failures fixed; retained in Git/logs. No formal outputs.|
|427520|Full video/update/save/reload smoke passed119sec.|
|427521|Dense temporal input smoke passed155sec, max allocated2.49GiB/card, exact reload.|
|427522|Initial teacher format failure; never used as formal labels.|
|427523|V2 finished exit0; raw audit inspection exposed text/objects in index lists. This version is excluded.|
|427524|First corrected-reward/KL update + optimizer checkpoint succeeded.|
|427525|Resume and second update succeeded, exit0; grad_norm0.554, nonzero fixed-base KL≈3e-5–5e-5. Both batches160/240sec for2 pairs.|
|427526|V3 numbered-audit dev check. First real pair passed: all4 verdicts bind correctly; support.5, resolved1, judge agreement2/3, reliability.5. Second pair also passed: reliability1.0; both numbered audit interfaces valid; exit complete.|
|427527|MAIN RUNNING: supervision and T0 prediction complete; T1 seed17 training/checkpoints verified.|

V3 CPU contract check covered quorum, numbered evidence binding, local recursion and known reliability; actual V2 had local video reinspection, V3 first pair resolved at root (conditional recursion still enabled). Source teacher/student and all model computation are onLop. Mac handled files, hashes and CPU math only.

## Next actions

1. Monitor427527 and its actual output counts/log. Retrieve427526 accounting when convenient; both V3 dev outputs are saved locally. If pipeline fails, inspect traceback; preserve outputs before fixing/restarting. Do not resubmit while427527 exists.
2. Monitor student training throughput, finite gradients/KL, invalid rollouts, zero-variance groups and checkpoint progress without MOS. Update timing from completed batches; do not change frozen settings or select checkpoints using outcomes.
3. `pipeline.py` generates `runs/main/bundle.json`, then T0 predictions, and trains/tests all arms/seeds serially. Stage failures exit; restart uses frozen existing outputs and latest optimizer boundary. Observe bundle's actual video/scene coverage and repeats, not nominal180 alone.
4. After GPU_STAGES_COMPLETE, synchronize predictions, bundle, training logs/checkpoints/accounting; verify all expected outputs. Only then run Mac `.venv/bin/python scripts/metrics.py` with configs/experiment.json, frozen target manifest, a local-path run index, and sealed MOS. Metrics runner currently requires full planned matrix. If deadline forces incomplete matrix, explicitly close/report incomplete scope before a separate final analysis; never present unfinished arms as completed.
5. Produce report with positive/null/negative findings, cost, coverage, weight ESS, gradient/zero-variance/invalid rollout rates and intervals. Claim only completed controls. Pause heartbeat at completion/deadline; no new experiment jobs afterward.

## Access

SSH multiplex `/private/tmp/sevlm3dqa-ssh` (current interactive session9968 with server keepalive; older20420 expired); if unavailable reconnect through approved tools using already-authorized user credentials, never store credentials in files. No bypass of approval service errors.
SGE env: SGE_ROOT=/opt/gridengine SGE_CELL=default SGE_QMASTER_PORT=536 SGE_EXECD_PORT=537; binaries /opt/gridengine/bin/lx-amd64.
Base Python `/home/shic4/.cache/rl_verified_agent_20260911/conda/bin/python`; project runtime_deps isolated. GPU script sets LD_LIBRARY_PATH/PYTHONPATH and assigned GPU UUIDs. Main70-ish-hour scheduling remains within original72hdeadline through per-stage deadline checks.
Adaptive heartbeat `se-vlm3dqa-72` active (user-authorized): currently every1h for student training/test. Use every2h only for stable teacher generation with >3h remaining. Estimate using measured throughput on each check and update this existing automation, never duplicate it. Diagnose observed failures in the current turn; do not lengthen the interval during failures or near the hard deadline. Keep quiet on unchanged state; notify meaningful progress/failure/completion only.
