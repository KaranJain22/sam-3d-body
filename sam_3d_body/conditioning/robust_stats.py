from __future__ import annotations

import numpy as np


def robust_log_dynamic_range(local_scale, lower_pct: float = 5.0, upper_pct: float = 95.0, eps: float = 1e-12) -> float:
    s = np.asarray(local_scale, dtype=np.float64)
    s = np.clip(s, eps, None)
    log_s = np.log(s)
    p_low = np.percentile(log_s, lower_pct)
    p_high = np.percentile(log_s, upper_pct)
    return float(np.exp(p_high - p_low))
