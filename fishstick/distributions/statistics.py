"""
Statistical testing utilities.

Provides hypothesis tests, confidence intervals, and statistical measures
for analyzing distributions and data.
"""

from typing import Optional, Tuple, Callable, Union, List
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np
from scipy import stats as scipy_stats


@dataclass
class TestResult:
    """Container for statistical test results."""

    statistic: float
    p_value: float
    rejected: bool
    confidence_level: float


@dataclass
class ConfidenceInterval:
    """Container for confidence interval results."""

    lower: float
    upper: float
    mean: float
    confidence_level: float


def t_test_1sample(
    x: Tensor,
    mu: float = 0.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    One-sample t-test.

    Tests whether the mean of a sample is equal to a hypothesized value.

    Args:
        x: Sample data
        mu: Hypothesized mean
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        TestResult with statistic, p-value, and rejection decision
    """
    x_np = x.cpu().numpy()
    n = len(x_np)
    x_bar = np.mean(x_np)
    s = np.std(x_np, ddof=1)

    if s == 0:
        return TestResult(
            statistic=0.0, p_value=1.0, rejected=False, confidence_level=1 - alpha
        )

    t_stat = (x_bar - mu) / (s / np.sqrt(n))

    if alternative == "two-sided":
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))
    elif alternative == "less":
        p_value = scipy_stats.t.cdf(t_stat, df=n - 1)
    else:
        p_value = 1 - scipy_stats.t.cdf(t_stat, df=n - 1)

    return TestResult(
        statistic=t_stat,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def t_test_2sample(
    x: Tensor,
    y: Tensor,
    alpha: float = 0.05,
    equal_var: bool = True,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Two-sample t-test.

    Tests whether two independent samples have equal means.

    Args:
        x: First sample
        y: Second sample
        alpha: Significance level
        equal_var: Assume equal variances
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        TestResult with statistic, p-value, and rejection decision
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    if equal_var:
        statistic, p_value = scipy_stats.ttest_ind(x_np, y_np, equal_var=True)
    else:
        statistic, p_value = scipy_stats.ttest_ind(x_np, y_np, equal_var=False)

    if alternative == "two-sided":
        p_value = p_value
    elif alternative == "less":
        p_value = p_value / 2 if statistic > 0 else p_value
    else:
        p_value = 1 - p_value / 2 if statistic > 0 else p_value

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def paired_t_test(
    x: Tensor,
    y: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Paired t-test for dependent samples.

    Tests whether the mean difference between paired observations is zero.
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    statistic, p_value = scipy_stats.ttest_rel(x_np, y_np)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def chi2_test(
    observed: Tensor,
    expected: Optional[Tensor] = None,
    alpha: float = 0.05,
) -> TestResult:
    """
    Chi-squared goodness-of-fit test.

    Tests whether observed frequencies match expected frequencies.
    """
    obs_np = observed.cpu().numpy().flatten()

    if expected is None:
        expected_np = np.full_like(obs_np, obs_np.mean())
    else:
        expected_np = expected.cpu().numpy().flatten()

    if expected_np.sum() != 0:
        expected_np = expected_np * obs_np.sum() / expected_np.sum()

    chi2_stat = np.sum((obs_np - expected_np) ** 2 / (expected_np + 1e-10))
    df = len(obs_np) - 1
    p_value = 1 - scipy_stats.chi2.cdf(chi2_stat, df)

    return TestResult(
        statistic=chi2_stat,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def ks_test(
    x: Tensor,
    y: Optional[Tensor] = None,
    cdf: Optional[Callable] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Kolmogorov-Smirnov test.

    Tests whether a sample comes from a specified distribution.

    Args:
        x: Sample data
        y: Second sample (for two-sample KS test) or None
        cdf: Cumulative distribution function
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    """
    x_np = x.cpu().numpy()

    if y is not None:
        y_np = y.cpu().numpy()
        statistic, p_value = scipy_stats.ks_2samp(x_np, y_np, alternative=alternative)
    elif cdf is not None:
        statistic, p_value = scipy_stats.kstest(x_np, cdf, alternative=alternative)
    else:
        statistic, p_value = scipy_stats.kstest(x_np, "norm", alternative=alternative)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def shapiro_wilk_test(
    x: Tensor,
    alpha: float = 0.05,
) -> TestResult:
    """
    Shapiro-Wilk test for normality.

    Tests whether a sample comes from a normal distribution.
    """
    x_np = x.cpu().numpy()

    if len(x_np) < 3 or len(x_np) > 5000:
        return TestResult(
            statistic=np.nan, p_value=np.nan, rejected=False, confidence_level=1 - alpha
        )

    statistic, p_value = scipy_stats.shapiro(x_np)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def levene_test(
    *samples: Tensor,
    alpha: float = 0.05,
) -> TestResult:
    """
    Levene's test for equality of variances.

    Tests whether multiple samples have equal variances.
    """
    samples_np = [s.cpu().numpy() for s in samples]
    statistic, p_value = scipy_stats.levene(*samples_np)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def mann_whitney_test(
    x: Tensor,
    y: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).

    Non-parametric test for differences between two independent samples.
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    statistic, p_value = scipy_stats.mannwhitneyu(x_np, y_np, alternative=alternative)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def wilcoxon_test(
    x: Tensor,
    y: Optional[Tensor] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Wilcoxon signed-rank test.

    Non-parametric test for paired samples.
    """
    x_np = x.cpu().numpy()

    if y is not None:
        y_np = y.cpu().numpy()
        diff = x_np - y_np
    else:
        diff = x_np

    if len(diff) == 0 or np.all(diff == 0):
        return TestResult(
            statistic=0.0, p_value=1.0, rejected=False, confidence_level=1 - alpha
        )

    statistic, p_value = scipy_stats.wilcoxon(diff, alternative=alternative)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def kruskal_wallis_test(
    *samples: Tensor,
    alpha: float = 0.05,
) -> TestResult:
    """
    Kruskal-Wallis H-test.

    Non-parametric test for comparing multiple independent samples.
    """
    samples_np = [s.cpu().numpy() for s in samples]
    statistic, p_value = scipy_stats.kruskal(*samples_np)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def friedman_test(
    *samples: Tensor,
    alpha: float = 0.05,
) -> TestResult:
    """
    Friedman test for repeated measures.

    Non-parametric test for comparing multiple related samples.
    """
    samples_np = [s.cpu().numpy() for s in samples]
    statistic, p_value = scipy_stats.friedmanchisquare(*samples_np)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def confidence_interval_mean(
    x: Tensor,
    alpha: float = 0.05,
    method: str = "t",
) -> ConfidenceInterval:
    """
    Confidence interval for the mean.

    Args:
        x: Sample data
        alpha: Significance level (1-alpha is confidence level)
        method: 't' for t-based, 'z' for z-based

    Returns:
        ConfidenceInterval with lower, upper bounds and mean
    """
    x_np = x.cpu().numpy()
    n = len(x_np)
    x_bar = np.mean(x_np)
    s = np.std(x_np, ddof=1)

    if method == "t":
        se = s / np.sqrt(n)
        df = n - 1
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
    else:
        se = s / np.sqrt(n)
        t_crit = scipy_stats.norm.ppf(1 - alpha / 2)

    return ConfidenceInterval(
        lower=x_bar - t_crit * se,
        upper=x_bar + t_crit * se,
        mean=x_bar,
        confidence_level=1 - alpha,
    )


def confidence_interval_proportion(
    x: int,
    n: int,
    alpha: float = 0.05,
    method: str = "wilson",
) -> ConfidenceInterval:
    """
    Confidence interval for a proportion.

    Args:
        x: Number of successes
        n: Total sample size
        alpha: Significance level
        method: 'wald', 'wilson', or 'exact'
    """
    p_hat = x / n if n > 0 else 0

    if method == "wald":
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
        lower = p_hat - z_crit * se
        upper = p_hat + z_crit * se

    elif method == "wilson":
        z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
        denominator = 1 + z_crit**2 / n
        center = (p_hat + z_crit**2 / (2 * n)) / denominator
        radius = (
            z_crit
            * np.sqrt(p_hat * (1 - p_hat) / n + z_crit**2 / (4 * n**2))
            / denominator
        )
        lower = center - radius
        upper = center + radius

    else:  # exact
        lower = scipy_stats.beta.ppf(alpha / 2, x, n - x + 1)
        upper = scipy_stats.beta.ppf(1 - alpha / 2, x + 1, n - x)

    return ConfidenceInterval(
        lower=max(0, lower),
        upper=min(1, upper),
        mean=p_hat,
        confidence_level=1 - alpha,
    )


def bootstrap_ci(
    x: Tensor,
    statistic_fn: Callable[[Tensor], Tensor],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """
    Bootstrap confidence interval.

    Args:
        x: Sample data
        statistic_fn: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level

    Returns:
        ConfidenceInterval with bootstrap bounds
    """
    x_np = x.cpu().numpy()
    n = len(x_np)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(x_np, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(torch.tensor(sample)).item())

    bootstrap_stats = np.array(bootstrap_stats)
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    mean_stat = statistic_fn(torch.tensor(x_np)).item()

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        mean=mean_stat,
        confidence_level=1 - alpha,
    )


def anova_oneway(
    *samples: Tensor,
    alpha: float = 0.05,
) -> TestResult:
    """
    One-way ANOVA.

    Tests whether multiple groups have equal means.
    """
    samples_np = [s.cpu().numpy() for s in samples]
    f_stat, p_value = scipy_stats.f_oneway(*samples_np)

    return TestResult(
        statistic=f_stat,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def effect_size_cohens_d(
    x: Tensor,
    y: Optional[Tensor] = None,
) -> float:
    """
    Cohen's d effect size.

    Measures standardized difference between means.
    """
    x_np = x.cpu().numpy()
    n1 = len(x_np)
    mean1 = np.mean(x_np)
    var1 = np.var(x_np, ddof=1)

    if y is None:
        return mean1 / np.sqrt(var1)

    y_np = y.cpu().numpy()
    n2 = len(y_np)
    mean2 = np.mean(y_np)
    var2 = np.var(y_np, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std


def effect_size_eta_squared(
    *samples: Tensor,
) -> float:
    """
    Eta-squared effect size for ANOVA.

    Proportion of variance explained by the grouping factor.
    """
    all_data = np.concatenate([s.cpu().numpy() for s in samples])
    grand_mean = np.mean(all_data)

    ss_between = sum(
        len(s) * (np.mean(s.cpu().numpy()) - grand_mean) ** 2 for s in samples
    )

    ss_total = np.sum((all_data - grand_mean) ** 2)

    return ss_between / ss_total if ss_total > 0 else 0.0


def multiple_testing_correction(
    p_values: Tensor,
    alpha: float = 0.05,
    method: str = "bonferroni",
) -> Tuple[Tensor, Tensor]:
    """
    Multiple testing correction.

    Args:
        p_values: Array of p-values
        alpha: Significance level
        method: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)

    Returns:
        Tuple of (corrected p-values, rejected nulls)
    """
    p_np = p_values.cpu().numpy()
    n = len(p_np)

    if method == "bonferroni":
        p_corrected = np.minimum(p_np * n, 1.0)

    elif method == "holm":
        sorted_idx = np.argsort(p_np)
        sorted_p = p_np[sorted_idx]
        sorted_p_adj = np.minimum(sorted_p * np.arange(n, 0, -1), 1.0)
        sorted_p_adj = np.minimum.accumulate(sorted_p_adj[::-1])[::-1]
        p_corrected = np.zeros_like(p_np)
        p_corrected[sorted_idx] = sorted_p_adj

    elif method == "fdr_bh":
        sorted_idx = np.argsort(p_np)[::-1]
        sorted_p = p_np[sorted_idx]
        sorted_p_adj = sorted_p * n / np.arange(n, 0, -1)
        sorted_p_adj = np.minimum.accumulate(sorted_p_adj[::-1])[::-1]
        p_corrected = np.zeros_like(p_np)
        p_corrected[sorted_idx] = sorted_p_adj

    else:
        raise ValueError(f"Unknown method: {method}")

    rejected = p_corrected < alpha

    return torch.tensor(p_corrected), torch.tensor(rejected)


def permutation_test(
    x: Tensor,
    y: Tensor,
    statistic_fn: Callable[[Tensor, Tensor], float],
    n_permutations: int = 1000,
    alpha: float = 0.05,
) -> TestResult:
    """
    Permutation test for equality of distributions.

    Args:
        x: First sample
        y: Second sample
        statistic_fn: Function to compute test statistic
        n_permutations: Number of permutations
        alpha: Significance level
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    observed_stat = statistic_fn(x_np, y_np)

    combined = np.concatenate([x_np, y_np])
    n_x = len(x_np)
    n_total = len(combined)

    perm_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_x = combined[:n_x]
        perm_y = combined[n_x:]
        perm_stats.append(statistic_fn(perm_x, perm_y))

    perm_stats = np.array(perm_stats)
    p_value = (np.abs(perm_stats) >= np.abs(observed_stat)).mean()

    return TestResult(
        statistic=observed_stat,
        p_value=p_value,
        rejected=p_value < alpha,
        confidence_level=1 - alpha,
    )


def bayes_factor(
    x: Tensor,
    y: Tensor,
    prior_prob: float = 0.5,
) -> float:
    """
    Approximate Bayes factor for comparing two samples.

    Uses BIC approximation for Bayesian model selection.
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    n1, n2 = len(x_np), len(y_np)
    n = n1 + n2

    mean_combined = np.mean(np.concatenate([x_np, y_np]))

    ll1 = -n1 / 2 * np.log(2 * np.pi) - n1 / 2 * np.log(np.var(x_np)) - n1 / 2
    ll2 = -n2 / 2 * np.log(2 * np.pi) - n2 / 2 * np.log(np.var(y_np)) - n2 / 2
    ll0 = (
        -n / 2 * np.log(2 * np.pi)
        - n / 2 * np.log(np.var(np.concatenate([x_np, y_np])))
        - n / 2
    )

    bic1 = -2 * (ll1 - ll0) + np.log(n1)
    bic2 = -2 * (ll2 - ll0) + np.log(n2)

    bf = np.exp((bic2 - bic1) / 2)

    return bf
