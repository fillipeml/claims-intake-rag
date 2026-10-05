"""The small amount of statistics a retrieval report needs to be honest.

Recall@5 of 0.82 over forty queries and Recall@5 of 0.82 over four thousand are different
claims that print identically, so nothing here returns a bare number: a rate comes back with
its Wilson interval and its denominator, and a mean comes back with a seeded percentile
bootstrap.

The same discipline, packaged as a general harness, is at
https://github.com/fillipeml/legal-llm-evals. This file is the subset a retrieval report uses;
it is deliberately not a dependency, because a measurement tool that drags a second repository
behind it does not get used.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

DEFAULT_CONFIDENCE = 0.95

#: Fixed so a reported interval cannot move between runs. A resampling procedure is exactly
#: where a number can drift without anybody intending it.
BOOTSTRAP_SEED = 20261005


@dataclass(frozen=True)
class Interval:
    """A value with the uncertainty around it, and the only way this package prints one."""

    point: float
    low: float
    high: float
    n: int
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_percent(self) -> str:
        if self.n == 0:
            return "no data"
        return f"{self.point * 100:.1f}% [{self.low * 100:.1f}, {self.high * 100:.1f}], n={self.n}"

    def as_score(self) -> str:
        """For a metric that is not a percentage of anything, such as MRR or nDCG."""
        if self.n == 0:
            return "no data"
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}], n={self.n}"


def wilson(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    """The Wilson score interval for a proportion.

    Over the normal approximation because a retrieval set is small and recall sits near the
    top of the range, where the naive interval runs past 1.0 and stops meaning anything.
    """
    if n < 0:
        raise ValueError("n cannot be negative")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be between 0 and n; got {successes} of {n}")
    if n == 0:
        return Interval(point=float("nan"), low=0.0, high=1.0, n=0, confidence=confidence)

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return Interval(
        point=p,
        low=max(0.0, centre - margin),
        high=min(1.0, centre + margin),
        n=n,
        confidence=confidence,
    )


def bootstrap_mean(
    values: list[float],
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = 4000,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """A percentile bootstrap for a mean, for scores with no closed form.

    MRR and nDCG are means of per-query values, not proportions, so Wilson is the wrong
    interval for them — it would be computed and would look entirely normal.
    """
    if not values:
        return Interval(point=float("nan"), low=0.0, high=1.0, n=0, confidence=confidence)
    if len(values) == 1:
        only = values[0]
        return Interval(point=only, low=only, high=only, n=1, confidence=confidence)

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()

    alpha = (1 - confidence) / 2
    return Interval(
        point=sum(values) / n,
        low=means[int(alpha * resamples)],
        high=means[min(resamples - 1, int((1 - alpha) * resamples))],
        n=n,
        confidence=confidence,
    )


@dataclass(frozen=True)
class PairedComparison:
    """Two retrieval configurations over the same queries. Only the differences inform."""

    better: int
    worse: int
    tied: int
    p_value: float
    n: int

    @property
    def discordant(self) -> int:
        return self.better + self.worse

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def describe(self) -> str:
        if self.discordant == 0:
            return f"the two configurations ranked identically on all {self.n} queries"
        if self.better > self.worse:
            direction = "better"
        elif self.worse > self.better:
            direction = "worse"
        else:
            direction = "unchanged on balance"
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.better} query(ies) improved, {self.worse} regressed, "
            f"{self.tied} unchanged out of {self.n}; the change looks {direction}, "
            f"p={self.p_value:.3f} ({verdict} at 0.05)"
        )


def sign_test(before: list[float], after: list[float]) -> PairedComparison:
    """An exact two-sided sign test over the queries whose score moved.

    The same queries run through both configurations, so this is paired data and the queries
    that scored the same under both carry no information about which is better. Comparing two
    mean nDCGs as though they were independent samples throws the pairing away along with most
    of the power.
    """
    if len(before) != len(after):
        raise ValueError(
            f"both runs must cover the same queries; got {len(before)} and {len(after)}"
        )
    n = len(before)
    if n == 0:
        raise ValueError("no queries to compare")

    better = sum(1 for b, a in zip(before, after, strict=True) if a > b)
    worse = sum(1 for b, a in zip(before, after, strict=True) if a < b)
    tied = n - better - worse
    discordant = better + worse

    if discordant == 0:
        p_value = 1.0
    else:
        # Exact binomial at p=0.5 over the discordant queries, two-sided. Not the normal
        # approximation: a retrieval set usually has fewer than twenty-five of them, which is
        # where the approximation stops being trustworthy.
        extreme = min(better, worse)
        tail = sum(math.comb(discordant, i) for i in range(extreme + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)

    return PairedComparison(better=better, worse=worse, tied=tied, p_value=p_value, n=n)
