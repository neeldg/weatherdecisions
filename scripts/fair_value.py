"""
Fair value = P(event) under a Gaussian model for daily high temperature.

The ML stack predicts the mean high (mu). Sigma is forecast uncertainty (ensemble spread,
historical RMSE, or a calibrated default). Contract fair value in cents is 100 * P(event).
"""
from __future__ import annotations

from scipy import stats


def prob_high_in_range(mu: float, sigma: float, low: float, high: float) -> float:
    """P(low <= T < high) for T ~ Normal(mu, sigma^2)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(stats.norm.cdf(high, loc=mu, scale=sigma) - stats.norm.cdf(low, loc=mu, scale=sigma))


def prob_high_geq(mu: float, sigma: float, threshold: float) -> float:
    """P(T >= threshold)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(1.0 - stats.norm.cdf(threshold, loc=mu, scale=sigma))


def prob_high_gt(mu: float, sigma: float, threshold: float) -> float:
    """P(T > threshold)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(1.0 - stats.norm.cdf(threshold, loc=mu, scale=sigma))


def fair_value_cents(p: float) -> int:
    """Map probability to nearest cent (0–100)."""
    return int(round(max(0.0, min(1.0, p)) * 100))
