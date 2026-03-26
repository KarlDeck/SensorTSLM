#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np


class MetricAggregator:
    def __init__(
        self,
        *,
        prior_mean: float | None = None,
        prior_std: float | None = None,
        prior_count: float = 0.0,
    ):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.prior_count = prior_count

    def aggregate(self, series: np.ndarray) -> tuple | None:
        valid = series[~np.isnan(series)]
        if len(valid) == 0:
            return None

        n = len(valid)
        instance_mean = float(np.mean(valid))
        instance_std = float(np.std(valid))

        # Hybrid Bayesian blending: (N * instance + λ * prior) / (N + λ)
        if self.prior_count > 0 and self.prior_mean is not None:
            mean_v = (n * instance_mean + self.prior_count * self.prior_mean) / (n + self.prior_count)
        else:
            mean_v = instance_mean

        if self.prior_count > 0 and self.prior_std is not None:
            std_v = (n * instance_std + self.prior_count * self.prior_std) / (n + self.prior_count)
        else:
            std_v = instance_std

        return (mean_v, float(np.max(valid)), float(np.min(valid)), std_v)
