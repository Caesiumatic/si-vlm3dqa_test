# SE-VLM3DQA-1

72-hour video-only evaluator → GRPO student → independent MOS evaluation pilot.

Start: 2026-09-25 22:00:06 UTC. Deadline: 2026-09-28 22:00:06 UTC.

Main compute: Lop SGE. Mac: auxiliary data processing and analysis, optional measured model assistance.

Current stage: frozen main experiment is RUNNING as Lop SGE job427527. SSH restored2026-09-27 19:02 UTC; supervision, T0, all seed17 runs and seed29 T1/T2 complete. Seed29 T3 training/test complete, T5 training complete and target prediction underway. Protocol: `RUN_PROTOCOL.md` and `configs/experiment.json`; live continuation: `reports/STATUS.md`. No scientific outcome has been evaluated yet.

Only source videos and anonymous identities enter model workflows. Human scores remain evaluation-only. Main arms T0/T1/T2/T3 with T5 prioritized. Teacher and student share initialization; T2/T3/T5 share labels and schedule.

Do not interpret a successful smoke test as the completed scientific experiment. Preserve failed runs and report actual data, coverage, seeds, costs and uncertainty.
