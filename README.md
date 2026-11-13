# si-vlm3dqa_test

A video-only teacher–student experiment for quality assessment of rendered 3D scenes.
A structured teacher collects pairwise judgments with adversarial search, local
visual checks, and reliability estimates. A Qwen2.5-VL-3B student learns from
those frozen judgments through LoRA and pairwise GRPO, then scores videos from
an independent target dataset.

The completed pilot isolates two questions: whether structured teacher labels
improve on five-vote labels, and whether reliability weights help when labels
and the planned training schedule are held fixed. Uniform and shuffled-weight
controls separate label choice from weight assignment.

## Recorded experiment

The run completed 12 student trainings across four arms and three seeds, plus
one base-model evaluation. All 5,616 target predictions were valid. On MUGSQA,
the primary structured-and-weighted method (T3) averaged SRCC 0.1697 versus
0.1737 for the five-vote baseline (T1). The prespecified T3−T1 difference was
−0.0040, with a paired-bootstrap 95% interval of [−0.0220, +0.0148]; this pilot
did not establish a gain from T3.

Read the [complete comparison and frozen settings](docs/reproduction.md) or
[final experiment report](reports/FINAL_REPORT.md). To inspect the saved metrics
with Python 3.10 or later:

```bash
python tools/summarize_results.py
```

The viewer reads checked-in results with the Python standard library. Training
requires the external videos, model weights, and GPU dependencies described in
the reproduction guide.

## Repository layout

| Path | Contents |
| --- | --- |
| [`scripts/`](scripts/) | Labeling, training, prediction, and evaluation scripts |
| [`configs/`](configs/) | Experiment settings |
| [`data/`](data/) | Data manifests and metadata |
| [`runs/`](runs/) | Run files |
| [`reports/`](reports/) | Experiment records |
| [`tools/`](tools/) | Result-reading utilities |

## Documentation

- [Setup, script sequence, and recorded configuration](docs/reproduction.md)
- [Run protocol](RUN_PROTOCOL.md)
- [Design notes](DESIGN.md)
