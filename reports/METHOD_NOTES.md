# Implementation decisions and corrections before protocol freeze

## Reward

Checked EvoQuality v5 §3.2 Eq. (2)–(6), https://arxiv.org/html/2509.25787v5 . For pair-centric adaptation, each score's preference probability uses its own score minus the partner group mean, divided by sqrt(var(own group)+var(partner group)+0.01), population variances. The fidelity reward is sqrt(y*p)+sqrt((1-y)*(1-p)). Near-equal is y=0.5; abstain is excluded.

The initial stack smoke used partner variance alone. This demonstrated execution but was **not** the final research formula. Corrected before any formal label generation or training; a numeric oracle now checks both variances. The first two successful engineering checkpoints must not enter experiment results.

GRPO will use one on-policy update per collected group, token-mean policy loss, population-standardized advantages, and reliability applied afterward. Fixed-base reference KL is separate. With a single policy update, the old-policy log probabilities equal detached current log probabilities (dropout is zero); no stale-data replay. LoRA initializes to the base function. Weight normalization is across the frozen training schedule, never separately inside a rollout group.

## Observation

The first successful stack test had 8 global frames and an 8-frame one-second local window at 224 pixels. The stronger pilot keeps 8 global frames and uses 16 consecutive frames over 16/30 seconds locally, preserving native 30 fps. This is a constrained spatial/temporal observation regime, not native-resolution full-video viewing.

## Comparisons

Baseline is an EvoQuality-style video adaptation, not a literal reproduction: JSON scalar outputs, explicit near-equal/abstain choices and evidence summaries replace the image prompt; comparison votes are independent and presentation order alternates. Student output formatting, observation and reward remain common to all arms.
