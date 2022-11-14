from __future__ import annotations

import json
import os
from typing import Any, Optional

import pandas as pd
import torch
import tqdm
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import GroupShuffleSplit
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForTokenClassification, get_scheduler

from data import AI4CodeDataset, BucketBatchSampler, DataCollatorForEmbeddingInputs

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class AI4CodeLightningModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        # self.margin = config.optim.ranking_loss_margin
        self.model = BertForTokenClassification(BertConfig(**config.model))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        attention_mask = batch["attention_mask"]
        loss_mask = (attention_mask[:, :, None] * attention_mask[:, None, :]).triu(1)

        logits = self.model(**batch).logits.float().squeeze(2)
        loss = (1.0 + logits[:, :, None] - logits[:, None, :]).relu()
        loss = ((loss_mask * loss).sum((1, 2)) / (loss_mask.sum((1, 2)) + 1e-12)).mean()
        return logits, attention_mask, loss
        """
        attention_mask = batch["attention_mask"]

        preds = self.model(**batch).logits.float().squeeze(2).sigmoid()
        labels = torch.arange(preds.size(1), device=preds.device)[None, :]
        labels = labels / attention_mask.sum(1, keepdim=True)
        loss = (attention_mask * (preds - labels).abs()).sum() / attention_mask.sum()
        return preds, attention_mask, loss
        """

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        _, _, loss = self(batch)
        self.log("step", self.global_step)
        self.log("train/loss", loss)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], idx: int
    ) -> list[torch.Tensor]:
        logits, attention_mask, loss = self(batch)
        self.log("step", self.global_step)
        self.log("val/loss", loss)
        return [x[y.bool()].argsort() for x, y in zip(logits, attention_mask)]

    def validation_epoch_end(self, outputs: list[list[torch.Tensor]]):
        batch_ranks = sum(outputs, [])

        # The below code is an implementation of the kandall-tau score in PyTorch.
        # First, we have to calculate the minimum transitions of the given rankings.
        # After that, the number of inversions will be divided by the number of maximum
        # available transitions.
        inversions = sum(
            (ranks[:, None] - ranks[None, :]).sign().relu().triu().sum()
            for ranks in batch_ranks
        )
        transitions = sum(len(ranks) * (len(ranks) - 1) for ranks in batch_ranks)
        self.log("val/score", 1 - 4 * inversions / transitions)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        do_decay = [p for p in self.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.ndim < 2]
        param_groups = [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = AdamW(param_groups, **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")


class AI4CodeDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def _load_dataset(self, dataset: DictConfig) -> pd.DataFrame:
        data = pd.read_csv(dataset.orders).dropna()
        notebook, embedding = dataset.notebook_dir, dataset.embedding_dir

        data["notebook"] = data.id.apply(lambda x: os.path.join(notebook, f"{x}.json"))
        data["embeddings"] = data.id.apply(lambda x: os.path.join(embedding, f"{x}.pt"))
        return data

    def _get_validation_clusters(self, dataset: DictConfig) -> pd.DataFrame:
        clusters = pd.read_csv(dataset.clusters)
        if self.config.data.validation_ratio == 1.0:
            return clusters

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.data.validation_ratio,
            random_state=42,
        )
        indices = next(splitter.split(clusters, groups=clusters.cluster))[1]
        return clusters.iloc[indices]

    def setup(self, stage: Optional[str] = None):
        data = pd.concat((self._load_dataset(x) for x in self.config.data.datasets))

        val_ancestors = [
            self._get_validation_clusters(dataset)
            for dataset in self.config.data.datasets
            if "clusters" in dataset
        ]
        val_mask = data.id.isin(pd.concat(val_ancestors).id)
        train_data, val_data = data[~val_mask], data[val_mask]

        notebook_lengths = []
        for example in tqdm.tqdm(train_data.itertuples(), total=len(train_data)):
            with open(example.notebook) as fp:
                notebook_lengths.append(len(json.load(fp)["source"]))

        self.train_dataset = AI4CodeDataset(
            train_data, self.config.data.max_length, sample=True
        )
        self.val_dataset = AI4CodeDataset(val_data, self.config.data.max_length)

        self.batch_sampler = BucketBatchSampler(
            notebook_lengths, self.config.train.batch_size, shuffle=True
        )
        self.collator = DataCollatorForEmbeddingInputs(pad_to_multiple_of=8)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.config.data.num_workers or os.cpu_count(),
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.data.num_workers or os.cpu_count(),
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=True,
        )
