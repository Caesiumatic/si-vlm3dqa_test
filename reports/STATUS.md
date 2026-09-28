# Execution status

Updated 2026-09-28 17:24 UTC. **COMPLETE** within the72h budget. See [final report](FINAL_REPORT.md).

- Original Lop SGE job427527 finished2026-09-28 17:15:05 UTC, failed0/exit0, qstat empty. No replacement or new job submitted.
- Teacher396 records; T0 plus T1/T2/T3/T5 × seeds17/29/43 complete. 324 student updates,5,616/5,616 valid target predictions.
- All final adapters locally backed up; video/config/adapter hashes and fixed labels/weights/order verified. Frozen code/config/manifest unchanged.
- All prediction hashes frozen in commit de048da before first target MOS access. Source MOS unused. No metrics-based configuration/checkpoint changes.
- Predefined analysis completed; primary T3−T1 SRCC −0.004028,95% interval[−0.021997,+0.014832]. No established full-method or reliability benefit in this pilot.
- Full results/cost/limitations: reports/FINAL_REPORT.md and reports/final/*.json. Accounting: reports/main-accounting.txt. Raw predictions/labels/training JSON under runs/main; model work and retained optimizer checkpoints onLop.
- Heartbeat se-vlm3dqa-72 was deleted after completion. No remaining scheduled experiment work. New research extensions require a new scope.

## Scope limits

27 updates per student; one frozen teacher-label realization; three training seeds; one target dataset; constrained224px video observation. Full-output student runs, T4, matched-compute/observation and isolated recursion controls were not run. Do not infer their outcomes.
