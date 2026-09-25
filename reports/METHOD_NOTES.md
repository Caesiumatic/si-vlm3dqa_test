# Implementation decisions and corrections before protocol freeze

## Reward

Checked EvoQuality v5 §3.2 Eq. (2)–(6), https://arxiv.org/html/2509.25787v5 . For pair-centric adaptation, each score's preference probability uses its own score minus the partner group mean, divided by sqrt(var(own group)+var(partner group)+0.01), population variances. The fidelity reward is sqrt(y*p)+sqrt((1-y)*(1-p)). Near-equal is y=0.5; abstain is excluded.

The initial stack smoke used partner variance alone. This demonstrated execution but was **not** the final research formula. Corrected before any formal label generation or training; a numeric oracle now checks both variances. The first two successful engineering checkpoints must not enter experiment results.

GRPO will use one on-policy update per collected group, token-mean policy loss, population-standardized advantages, and reliability applied afterward. Fixed-base reference KL is separate. With a single policy update, the old-policy log probabilities equal detached current log probabilities (dropout is zero); no stale-data replay. LoRA initializes to the base function. Weight normalization is across the frozen training schedule, never separately inside a rollout group.

## Observation

The first successful stack test had 8 global frames and an 8-frame one-second local window at 224 pixels. The stronger pilot keeps 8 global frames and uses 16 consecutive frames over 16/30 seconds locally, preserving native 30 fps. This is a constrained spatial/temporal observation regime, not native-resolution full-video viewing.

## Comparisons

Baseline is an EvoQuality-style video adaptation, not a literal reproduction: JSON scalar outputs, explicit near-equal/abstain choices and evidence summaries replace the image prompt; comparison votes are independent and presentation order alternates. Student output formatting, observation and reward remain common to all arms.

## Development interface fixes

The initial two-pair held-out pilot exposed inconsistent field names (`comparison`, `quality`, etc. in place of `preference`) and omitted claim statuses. V2 supplies exact JSON key templates, treating absent/conflicting claim statuses as unresolved and requesting local reinspection. It does not infer preferences from alternative fields or force agreement. A 2–2–1 vote remains abstain. Repeated identical local windows stop recursion with an explicit reason. Real video evidence remains the model's fallible judgment, not verified human truth.

The dense-window stack test passed in154.85sec; parameter-update work78.56sec, maximum allocated2.49GiB/card, reload match exact. First V2 dev pair: baseline5 calls89.72sec, full8 calls146.25sec with a real local-audit call. These are engineering throughput observations from very few pairs, not a guarantee of end-to-end time.

Budget draft:198 candidate pairs across180 source videos /18 scenes;108 pair exposures in27 batches of4 per arm/seed; G=4 per video,3 seeds, T1/T2/T3/T5. Target432 videos /54 groups. Student work estimated about45h, teachers13h, scoring8h, leaving several hours for engineering/reporting if queue availability persists. Larger recommendation (360 source,378 pairs,864 test) exceeds the current four-GTX980-only budget at measured speed. Formal configuration is not yet frozen pending multi-step/resume validation.
