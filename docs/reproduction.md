# Experiment and reproduction

[Project overview](../README.md)

## Reading the recorded experiment

From the repository root, using Python 3.10 or later:

```bash
python tools/summarize_results.py
```

The command prints means from
[`reports/final/metrics.json`](../reports/final/metrics.json). That file also
contains all prespecified contrasts. The [final report](../reports/FINAL_REPORT.md)
connects these results to the data counts, diagnostics, and execution record.

The result viewer uses only the Python standard library. It does not rerun
model inference or statistical resampling.

Twelve student runs cover T1, T2, T3, and T5 at seeds 17, 29, and 43. Together
with one base-model evaluation (T0), they produced 5,616 valid predictions.
Student rows below average the three seeds; raw PLCC uses uncalibrated scores.

| Arm | Teacher labels / weights | Mean SRCC | Mean raw PLCC |
| --- | --- | ---: | ---: |
| T0 | Base model | 0.1594 | 0.1194 |
| T1 | Five-vote baseline / uniform | 0.1737 | 0.1295 |
| T2 | Structured teacher / uniform | 0.1700 | 0.1322 |
| T3 | Structured teacher / reliability | 0.1697 | 0.1221 |
| T5 | Structured teacher / shuffled reliability | 0.1722 | 0.1250 |

The prespecified T3−T1 SRCC difference is −0.0040, with a paired-bootstrap 95%
interval of [−0.0220, +0.0148]. Resampling pairs target content groups and student
seeds. Per-seed values and the other contrasts are in the linked metrics file.

## Frozen configuration

| Component | Setting |
| --- | --- |
| Model | Qwen/Qwen2.5-VL-3B-Instruct |
| Revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Precision | FP16 |
| Observations | 8 global + 16 consecutive local frames, 224 pixels |
| Teacher baseline | 5 votes |
| Training | 108 pairs, 27 updates, 4 rollouts per video |
| LoRA | Language q_proj/v_proj, rank 8, alpha 16; frozen vision |
| Optimizer settings | Learning rate 1e-5, base-model KL coefficient 0.04 |
| Student seeds | 17, 29, 43 |
| Primary comparison | T3−T1 SRCC; 2,000 paired bootstrap resamples |

The authoritative settings and label policy are in
[`configs/experiment.json`](../configs/experiment.json). The
[run protocol](../RUN_PROTOCOL.md) and
[pre-experiment decisions](../PRE_EXPERIMENT_DECISIONS.md) record the original
selection rules.

## Training code

| Path | Purpose |
| --- | --- |
| [`evaluator.py`](../scripts/evaluator.py), [`label.py`](../scripts/label.py) | Teacher judgments and labels |
| [`reward.py`](../scripts/reward.py), [`train.py`](../scripts/train.py) | Pairwise rewards and GRPO training |
| [`predict.py`](../scripts/predict.py), [`metrics.py`](../scripts/metrics.py) | Target predictions and statistical analysis |
| [`reports/final/`](../reports/final/) | Metrics, diagnostics, and run provenance |

The sequence is teacher labeling (`scripts/label.py`), shared training-bundle
construction (`scripts/bundle.py`), student training (`scripts/train.py`),
prediction (`scripts/predict.py`), and evaluation (`scripts/metrics.py`).
`scripts/pipeline.py` orchestrated the original time-bounded run and retains its
historical deadline and pilot gate.

To prepare a fresh training run, supply the source and target videos, original
manifests, model weights, and a GPU environment with the model dependencies.
The repository includes source and result records; a portable, fully pinned
training environment remains to be assembled. Follow the stage sequence and
frozen protocol above, and write new outputs to a separate run directory.

## Data separation

Explicit-NeRF-QA supplies teacher comparisons. The common valid teacher subset
contains 152 pairs; the fixed training schedule selects 108 distinct pairs
covering 165 source videos. MUGSQA supplies the independent target: 54 content
groups and 432 videos. Human scores are used only after predictions are complete.

The reliability-weighting comparison holds labels and training schedule fixed
between T2, T3, and T5. The teacher has one label realization; the three student
seeds vary LoRA initialization, ordering, and sampling.
