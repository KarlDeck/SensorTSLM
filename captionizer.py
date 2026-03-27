#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from timef.schema import DatasetManifest, CaptionResult
from transformer import Transformer
from annotator import Annotator
from reviewer import EvaluationResult, Reviewer


class Captionizer:
    def __init__(
        self,
        dataset,
        transformer: Transformer,
        annotator: Annotator,
        reviewer: Reviewer | None = None,
    ) -> None:
        self.dataset = dataset
        self.transformer = transformer
        self.annotator = annotator
        self.reviewer = reviewer

    def run(
        self, max_rows: int | None = None,
    ) -> tuple[CaptionResult, EvaluationResult | None]:
        manifest = DatasetManifest()
        for spec in self.transformer.get_signal_specs():
            manifest.signal_specs[spec.id] = spec
        for spec in self.annotator.get_annotation_specs():
            manifest.annotation_specs[spec.id] = spec

        result = CaptionResult(manifest=manifest)

        for i in range(len(self.dataset)):
            if max_rows is not None and i >= max_rows:
                break

            row = self.dataset[i]
            signals = self.transformer.transform_row(row)
            for signal in signals:
                result.signals[signal.id] = signal

            samples, annotations = self.annotator.annotate(signals)
            result.samples.extend(samples)
            result.annotations.extend(annotations)

        evaluation = self.reviewer.evaluate(result) if self.reviewer else None
        return result, evaluation


if __name__ == "__main__":
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer
    from mhc.constants import MHC_CHANNEL_CONFIG
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from models.local import LocalConfig, LocalModel
    from visualizer import plot_row
    import numpy as np

    dataset = MHCDataset(min_wear_pct=0.0)
    annotator = Annotator([
        StatisticalExtractor(MHC_CHANNEL_CONFIG),
        StructuralExtractor(MHC_CHANNEL_CONFIG),
    ])
    model = LocalModel(
        LocalConfig(model="google/gemma-3-4b-it"),
        MHC_CHANNEL_CONFIG,
    )
    reviewer = Reviewer(model)
    captionizer = Captionizer(dataset, MHCTransformer(), annotator, reviewer=reviewer)
    print(f"Dataset size: {len(dataset)}")

    result, evaluation = captionizer.run(max_rows=5)
    print(f"Signals: {len(result.signals)}")
    print(f"Samples: {len(result.samples)}")
    print(f"Annotations: {len(result.annotations)}")
    if evaluation:
        print(f"Evaluation: {len(evaluation.scores)} scores, mean={evaluation.mean_score}")

    # Plot first 4 rows with at least 5 active channels
    shown = 0
    for signals, samples, annotations in result.iter_rows():
        active = sum(
            s.metadata.get("has_any_data", True) for s in signals
        )
        if active >= 5:
            plot_row(signals, samples, annotations)
            shown += 1
            if shown >= 4:
                break
