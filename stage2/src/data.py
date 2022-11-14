from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


@dataclass
class AI4CodeDataset(Dataset):
    data: pd.DataFrame
    max_length: int = 512
    sample: bool = False

    def __len__(self) -> int:
        return len(self.data)

    def _create_position_ids(self, cell_types: list[str]) -> list[int]:
        position_ids, code_cell_position = [], 1
        for cell_type in cell_types:
            position_ids.append(code_cell_position if cell_type == "code" else 0)
            code_cell_position += 1 if cell_type == "code" else 0
        return torch.tensor(position_ids, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.data.iloc[index]
        embeddings = torch.load(example.embeddings, map_location="cpu")

        with open(example.notebook) as fp:
            notebook = json.load(fp)
            cell_names = list(notebook["cell_type"])
            cell_types = list(notebook["cell_type"].values())

        orders = [cell_names.index(name) for name in example.cell_order.split()]
        embeddings, cell_types = embeddings[orders], [cell_types[i] for i in orders]

        if self.sample and len(cell_types) > 10:
            length = min(len(cell_types), self.max_length)
            length = np.random.randint(int(length * 0.5), length)
            orders = sorted(np.random.permutation(len(cell_types))[:length])
            embeddings, cell_types = embeddings[orders], [cell_types[i] for i in orders]
        else:
            embeddings = embeddings[: self.max_length]
            cell_types = cell_types[: self.max_length]

        return {
            "inputs_embeds": embeddings,
            "attention_mask": torch.ones(embeddings.size(0), dtype=torch.long),
            "position_ids": self._create_position_ids(cell_types),
        }


@dataclass
class DataCollatorForEmbeddingInputs:
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def _pad(self, example: dict[str, torch.Tensor], max_length: int):
        padding = max_length - example["inputs_embeds"].size(0)
        example["inputs_embeds"] = F.pad(example["inputs_embeds"], (0, 0, 0, padding))
        example["attention_mask"] = F.pad(example["attention_mask"], (0, padding))
        example["position_ids"] = F.pad(example["position_ids"], (0, padding))

    def __call__(
        self, examples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        max_length = max(example["inputs_embeds"].size(0) for example in examples)
        max_length = self.max_length or max_length

        max_length = math.ceil(max_length / self.pad_to_multiple_of)
        max_length = max_length * self.pad_to_multiple_of

        for example in examples:
            self._pad(example, max_length)
        return {name: torch.stack([x[name] for x in examples]) for name in examples[0]}


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
