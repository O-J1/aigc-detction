from __future__ import annotations

import torch

from micv.data.dataset import DistributedWeightedSampler


class LabelDataset:
    def __init__(self, labels: list[int]) -> None:
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> list[int]:
        return self._labels


def test_distributed_weighted_sampler_slices_one_global_draw() -> None:
    dataset = LabelDataset([0, 0, 0, 1])
    samplers = [
        DistributedWeightedSampler(dataset, num_replicas=2, rank=rank, replacement=True, seed=13)
        for rank in range(2)
    ]

    rank_indices = [list(sampler) for sampler in samplers]
    generator = torch.Generator().manual_seed(13)
    weights = torch.as_tensor([1 / 3, 1 / 3, 1 / 3, 1], dtype=torch.double)
    expected_global = torch.multinomial(weights, 4, True, generator=generator).tolist()

    assert rank_indices[0] == expected_global[0::2]
    assert rank_indices[1] == expected_global[1::2]
    assert [len(indices) for indices in rank_indices] == [2, 2]


def test_distributed_weighted_sampler_set_epoch_changes_global_draw() -> None:
    dataset = LabelDataset([0, 0, 1, 1])
    sampler = DistributedWeightedSampler(dataset, num_replicas=2, rank=0, replacement=True, seed=7)

    first_epoch = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    weights = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.double)
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(8)
    expected_first = torch.multinomial(weights, 4, True, generator=first_generator).tolist()[0::2]
    expected_second = torch.multinomial(weights, 4, True, generator=second_generator).tolist()[0::2]

    assert first_epoch == expected_first
    assert second_epoch == expected_second