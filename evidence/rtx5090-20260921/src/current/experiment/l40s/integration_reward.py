"""Rule-based reward for the deterministic arithmetic fixture (sum of reward scores)."""

import re


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    """Return 1.0 when ``#### <number>`` matches the ground truth, else 0.0."""
    del data_source, extra_info, kwargs
    match = re.findall(r"####\s*(-?[\d,]+)", solution_str)
    if not match:
        return 0.0
    predicted = match[-1].replace(",", "").strip()
    return 1.0 if predicted == str(ground_truth).strip() else 0.0
