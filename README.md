# SE-VLM3DQA-1

72-hour video-only evaluator → GRPO student → independent MOS evaluation pilot.

Start: 2026-09-25 22:00:06 UTC. Deadline: 2026-09-28 22:00:06 UTC.

Main compute: Lop SGE. Mac: auxiliary data processing and analysis, optional measured model assistance.

Completed: all T0 and T1/T2/T3/T5 × seeds17/29/43, 5,616 valid predictions. Lop job427527 exited0 on2026-09-28 17:15 UTC. Final MOS analysis: T3−T1 SRCC −0.0040, 95% interval [−0.0220,+0.0148]; no established improvement. See [final report](reports/FINAL_REPORT.md). Frozen protocol: `RUN_PROTOCOL.md` and `configs/experiment.json`.

Only source videos and anonymous identities enter model workflows. Human scores remain evaluation-only. Main arms T0/T1/T2/T3 with T5 prioritized. Teacher and student share initialization; T2/T3/T5 share labels and schedule.

Do not interpret a successful smoke test as the completed scientific experiment. Preserve failed runs and report actual data, coverage, seeds, costs and uncertainty.
