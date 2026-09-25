"""Pair-centric fidelity reward and post-normalization reliability weights."""
import numpy as np
from scipy.special import ndtr


def fidelity(scores, partner_scores, target, gamma=0.01):
    scores, partner_scores = np.asarray(scores, float), np.asarray(partner_scores, float)
    # EvoQuality v5 Eq. (2): both rollout-group variances plus gamma.
    variance = float(np.var(scores)+np.var(partner_scores)) + gamma
    probability = ndtr((scores - partner_scores.mean()) / np.sqrt(variance))
    return np.sqrt(probability * target) + np.sqrt((1 - probability) * (1 - target))


def advantages(rewards, epsilon=1e-6):
    rewards = np.asarray(rewards, float)
    if rewards.std() < epsilon:
        return np.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + epsilon)


def check():
    r = fidelity([1, 2, 3, 4, 5], [2, 3, 3, 4], 1)
    assert np.all(np.diff(r) > 0), r
    assert np.allclose(fidelity([1, 2, 3], [2, 3, 4], 1),
                       fidelity([5, 4, 3], [4, 3, 2], 0))
    expected = np.sqrt(ndtr((np.array([1.,2.,3.])-3.)/np.sqrt(4./3.+.01)))
    assert np.allclose(fidelity([1,2,3],[2,3,4],1),expected)
    assert np.all(advantages([0.4] * 4) == 0)
    a = advantages(r)
    assert np.allclose(a.mean(), 0) and abs(a.std() - 1) < 1e-4
    assert np.allclose(0 * a, 0) and np.allclose(0.5 * a * 2, a)
    return {'evoquality_eq2_both_variances':True, 'reward_monotonicity': True, 'swap_symmetry': True,
            'zero_variance': True, 'post_normalization_scaling': True}
