from __future__ import annotations

import glob
import json
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    get_scheduler,
)

from data import BucketBatchSampler, DataCollatorDict, TextDataset

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class SimCSELightningModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.loss_mlm_weight = config.optim.loss_mlm_weight

        self.model = AutoModelForMaskedLM.from_pretrained(**config.model)
        self.classifier = nn.Linear(
            self.model.config.hidden_size,
            self.model.config.hidden_size,
        )

    def forward(
        self, batch: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        # Get sentence embeddings with different dropout noises.
        h1 = self.model.base_model(**batch["simcse"]).last_hidden_state[:, 0]
        h2 = self.model.base_model(**batch["simcse"]).last_hidden_state[:, 0]
        z1, z2 = self.classifier(h1).tanh(), self.classifier(h2).tanh()
        n1, n2 = F.normalize(h1.float()), F.normalize(h2.float())

        # Calculate cosine similarity scores and identical labels.
        logits = F.cosine_similarity(z1[:, None, :], z2[None, :, :], dim=2, eps=1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate the alignment and uniformity of the model.
        alignment = (n1 - n2).square().sum(1).mean()
        uniformity = (-2 * F.pdist(n1).square()).exp().mean().log()

        # The total loss consists of masked-lm and contrastive loss.
        loss_simcse = F.cross_entropy(logits / 0.05, labels)
        loss_mlm = self.model(**batch["mlm"]).loss
        loss = loss_simcse + self.loss_mlm_weight * loss_mlm

        return {
            "loss": loss,
            "loss_simcse": loss_simcse,
            "loss_mlm": loss_mlm,
            "alignment": alignment,
            "uniformity": uniformity,
        }

    def training_step(
        self, batch: dict[str, dict[str, torch.Tensor]], index: int
    ) -> torch.Tensor:
        metrics = self(batch)
        self.log("step", self.global_step)
        self.log_dict(metrics)
        return metrics["loss"]

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


class SimCSEDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(**self.config.model)

        filenames, texts = [], []
        for directory in self.config.data.notebooks:
            filenames += glob.glob(os.path.join(directory, "*.json"))

        # Read the notebooks and get all texts of which lengths are more than
        # `min_characters`.
        min_characters = self.config.data.min_characters
        for filename in tqdm.tqdm(filenames):
            with open(filename) as fp:
                notebook = json.load(fp)
            for text in notebook["source"].values():
                text = " ".join(text.split())
                if len(text) >= min_characters:
                    texts.append(text)

        # Because there should be some duplicated cell contents, we will sort them by
        # their lengths and deduplicate with simple neighbor comparing.
        texts = sorted(texts, key=lambda text: len(text))
        texts = [text for i, text in enumerate(texts[1:]) if text != texts[i]]

        self.dataset = TextDataset(texts, tokenizer, self.config.data.max_length)
        self.batch_sampler = BucketBatchSampler(
            [len(text) for text in texts], self.config.train.batch_size, shuffle=True
        )

        # Note that the SimCSE contrastive learning needs non-masked sequences, we will
        # use different data-collator for simple padding and generating mask tokens.
        self.collator = DataCollatorDict(
            simcse=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            mlm=DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.config.data.num_workers or os.cpu_count(),
            collate_fn=self.collator,
            persistent_workers=True,
        )
