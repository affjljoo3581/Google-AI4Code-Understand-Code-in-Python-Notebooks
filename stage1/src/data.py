from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Sampler
from transformers import DataCollator, PreTrainedTokenizer


@dataclass
class TextDataset(Dataset):
    texts: list[str]
    tokenizer: PreTrainedTokenizer
    max_length: int = 128

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length
        )


class DataCollatorDict:
    def __init__(self, **collators: DataCollator):
        self.collators = collators

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {k: dict(v(examples)) for k, v in self.collators.items()}


class BucketBatchSampler(Sampler):
    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = False):
        self.indices = np.argsort(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.indices) // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        iters = np.random.permutation(len(self)) if self.shuffle else range(len(self))
        for index in iters:
            yield self.indices[index * self.batch_size : (index + 1) * self.batch_size]
