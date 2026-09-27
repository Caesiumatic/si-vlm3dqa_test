# Execution status

Updated2026-09-27 23:08 UTC. Hard deadline2026-09-28 22:00:06 UTC (Monday17:00 Chicago). Local `/Users/shichen/GitHub/SE-VLM3DQA-1`; remote `/home/shic4/SE-VLM3DQA-1`. Main branch only. User authorizes autonomous execution; Lop primary, Mac auxiliary. No MOS values read for development.

## Current stage: T5 seed29 target prediction

**Main SGE job427527 is RUNNING; SSH connectivity restored. Started2026-09-26 00:00:27 UTC. Supervision, T0 and all four seed17 student runs completed; T1 seed29 training and prediction complete; T2 seed29 complete; T3 seed29 complete; T5 seed29 training complete; target prediction active.** Remote persistent receipt: `reports/main-job-id.txt`. Inspect qstat/logs before any new submission; never duplicate. Queue gpu1@compute-1-20.local, cuda PE4, four GTX980,69h walltime. Main log `reports/main-pipeline.log`; outputs `runs/main`.

The earlier automatic approval HTTP401 outage is RESOLVED. SSH checks and submission succeeded. No user reauthorization needed.

## Latest operational snapshot

19:04 UTC: SSH connectivity restored and reauthenticated. Original persisted job427527 is still running on the same four-GTX980 queue. Outputs advanced during the monitoring outage; no resubmission was needed.

11:42 UTC: original job427527 remains running. Supervision finished198/198 pairs for both methods (396 records) at2026-09-26 11:33:47 UTC. All198/198 full audits have valid numbered interfaces;37 exercised local recursion. Baseline153 valid/45 abstained; full197 valid/1 abstained. Common-valid152 pairs, spanning165 videos/18 groups;33 common labels differ. No human score has been read; these labels do not establish teacher accuracy.

Frozen schedule:108 unique pairs,165 videos/18 source groups, no repeats; weight mean1, range0.520–1.248, ESS102.44/108. T5 preserves the exact weight distribution. Both teacher labels and shuffled weights were checked against original records. Bundle SHA256 `ff7d80df01a31eb7f4f12a1cfb5861080116808c0d05b6c95fd954410a3bd71d` matches remote; retained as reports/frozen-label-bundle.json.

T0 predictions complete432/432, all valid with matching video/config hashes and no adapter. Completed at2026-09-26T12:18:23+00:00; no MOS read and no quality metrics computed.

16:46 UTC: T1 seed17 COMPLETE:27/27 updates,108 presentations;104 processed pairs and4 invalid-rollout pairs skipped under the frozen rule. All27 gradient norms finite/nonzero, KL finite;6/208 processed rollout groups had zero advantage variance. Batch computation totaled3.62h. All432 final predictions valid; video/config/adapter hashes checked against locally backed-up final LoRA weights. Prediction completion2026-09-26 16:42:33 UTC. Completion record: reports/T1_17-completion.json. No MOS accessed or model-quality metrics computed.

T2 seed17 COMPLETE:27 updates,108 presentations (103 processed,5 invalid-rollout pairs skipped);432/432 predictions valid. All prediction video/config/adapter hashes verified against locally backed-up final LoRA. Prediction completion2026-09-26T21:03:53+00:00. Report: reports/T2_17-completion.json. No MOS read.

T3 seed17 COMPLETE:27 updates,108 presentations (103 processed,5 invalid-rollout pairs skipped);432/432 predictions valid. All prediction video/config/adapter hashes verified against locally backed-up final LoRA. Prediction completion2026-09-27T01:25:56+00:00. Report: reports/T3_17-completion.json. No MOS read.

T5 seed17 COMPLETE:27 updates,108 presentations;{'processed': 103, 'invalid_rollout_pair_skipped': 5}. All gradients finite/nonzero, KL finite;6/206 processed rollout groups have zero advantage variance. Full-run shuffled weights, labels and T3 order verified. All432 predictions valid with matching video/config/adapter hashes; final LoRA backed up. Prediction completed2026-09-27T05:49:14+00:00. Record reports/T5_17-completion.json.

T1 seed29 COMPLETE:27 updates,108 presentations (106 processed,2 invalid-rollout pairs skipped);432/432 predictions valid. All prediction video/config/adapter hashes verified against locally backed-up final LoRA. Prediction completion2026-09-27T10:16:41+00:00. Record: reports/T1_29-completion.json. No MOS read.

T2 seed29 COMPLETE:27 updates,108 presentations; pair statuses{'processed': 107, 'invalid_rollout_pair_skipped': 1}. All gradient norms finite/nonzero, KL finite; zero-variance groups2/214. All432 predictions valid with video/config/adapter hashes verified against locally backed-up final LoRA. Prediction completion2026-09-27T14:44:36+00:00. Record: reports/T2_29-completion.json. No MOS read.

T3 seed29 COMPLETE:27 updates,108 presentations (106 processed,2 invalid-rollout pairs skipped);432/432 predictions valid. All prediction video/config/adapter hashes verified against locally backed-up final LoRA. Prediction completion2026-09-27T19:09:53+00:00. Record: reports/T3_29-completion.json. No MOS read.

23:08 UTC: T5 seed29 training COMPLETE,27 updates,108 presentations; statuses{'processed': 106, 'invalid_rollout_pair_skipped': 2}. All gradients finite/nonzero (0.112–0.395), KL finite; zero-variance groups2/212. Frozen full labels, shuffled weights, full T3 pair order and preserved weight distribution verified. Final LoRA backed up. Record: reports/T5_29-training.json.

Target prediction active: 150/432, statuses{'valid': 150}; video/config/adapter hashes verified. Recent30 intervals mean6.43sec/video; remaining approximately30.2min. Completed student runs7/12; T0 plus completed students total3456 predictions.

Forecast uses current prediction speed and completed-run averages: seed29 all arms0.50h from this check; all seeds18.0h, around2026-09-28T17:08:30.384859+00:00. Startup overhead excluded; future speeds provisional. Original job427527 healthy, no resubmission or frozen changes. MOS remains sealed. Retain hourly monitoring.

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
|427527|MAIN RUNNING; connection restored: supervision, T0 and all seed17 arms complete; T1 seed29 complete; T2 seed29 complete; T3 seed29 complete; T5 seed29 training complete; target prediction active.|

V3 CPU contract check covered quorum, numbered evidence binding, local recursion and known reliability; actual V2 had local video reinspection, V3 first pair resolved at root (conditional recursion still enabled). Source teacher/student and all model computation are onLop. Mac handled files, hashes and CPU math only.

## Next actions

1. Monitor427527 and its actual output counts/log. Retrieve427526 accounting when convenient; both V3 dev outputs are saved locally. If pipeline fails, inspect traceback; preserve outputs before fixing/restarting. Do not resubmit while427527 exists.
2. Monitor student training throughput, finite gradients/KL, invalid rollouts, zero-variance groups and checkpoint progress without MOS. Update timing from completed batches; do not change frozen settings or select checkpoints using outcomes.
3. `pipeline.py` generates `runs/main/bundle.json`, then T0 predictions, and trains/tests all arms/seeds serially. Stage failures exit; restart uses frozen existing outputs and latest optimizer boundary. Observe bundle's actual video/scene coverage and repeats, not nominal180 alone.
4. After GPU_STAGES_COMPLETE, synchronize predictions, bundle, training logs/checkpoints/accounting; verify all expected outputs. Only then run Mac `.venv/bin/python scripts/metrics.py` with configs/experiment.json, frozen target manifest, a local-path run index, and sealed MOS. Metrics runner currently requires full planned matrix. If deadline forces incomplete matrix, explicitly close/report incomplete scope before a separate final analysis; never present unfinished arms as completed.
5. Produce report with positive/null/negative findings, cost, coverage, weight ESS, gradient/zero-variance/invalid rollout rates and intervals. Claim only completed controls. Pause heartbeat at completion/deadline; no new experiment jobs afterward.

## Access

SSH multiplex `/private/tmp/sevlm3dqa-ssh` (active session23229 with server keepalive; older30682/90741/9968 closed); if unavailable reconnect through approved tools using already-authorized user credentials, never store credentials in files. No bypass of approval service errors.
SGE env: SGE_ROOT=/opt/gridengine SGE_CELL=default SGE_QMASTER_PORT=536 SGE_EXECD_PORT=537; binaries /opt/gridengine/bin/lx-amd64.
Base Python `/home/shic4/.cache/rl_verified_agent_20260911/conda/bin/python`; project runtime_deps isolated. GPU script sets LD_LIBRARY_PATH/PYTHONPATH and assigned GPU UUIDs. Main70-ish-hour scheduling remains within original72hdeadline through per-stage deadline checks.
Adaptive heartbeat `se-vlm3dqa-72` active (user-authorized): currently every1h for student training/test. Use every2h only for stable teacher generation with >3h remaining. Estimate using measured throughput on each check and update this existing automation, never duplicate it. Diagnose observed failures in the current turn; do not lengthen the interval during failures or near the hard deadline. Keep quiet on unchanged state; notify meaningful progress/failure/completion only.
