from __future__ import annotations

import argparse
import os
import warnings

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from lightning import SimCSEDataModule, SimCSELightningModule

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(config: DictConfig, args: argparse.Namespace):
    name = f"simcse-{config.model.pretrained_model_name_or_path.split('/')[-1]}"
    checkpoint = ModelCheckpoint(save_last=True)

    Trainer(
        gpus=1,
        precision=16,
        amp_backend="apex",
        log_every_n_steps=config.train.log_every_n_steps,
        max_steps=config.optim.scheduler.num_training_steps,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        resume_from_checkpoint=args.resume_from,
        callbacks=[checkpoint],
        logger=WandbLogger(name, id=args.resume_id, project="ai4code-stage1"),
    ).fit(SimCSELightningModule(config), SimCSEDataModule(config))

    # Save the trained model weights and its tokenizer.
    module = SimCSELightningModule.load_from_checkpoint(
        checkpoint.last_model_path, config=config
    )
    module.model.save_pretrained(name)
    AutoTokenizer.from_pretrained(**config.model).save_pretrained(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(config, args)
