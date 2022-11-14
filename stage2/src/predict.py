from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification

from data import AI4CodeDataset, DataCollatorForEmbeddingInputs


def sample_cells(cell_order: str, max_length: int) -> str:
    cell_order = cell_order.split()
    subsequence = sorted(np.random.permutation(len(cell_order))[:max_length])
    return " ".join([cell_order[i] for i in subsequence])


def prepare_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, DataLoader]:
    data = pd.read_csv(args.init_orders)
    notebook, embedding = args.notebook_dir, args.embedding_dir

    data["notebook"] = data.id.apply(lambda x: os.path.join(notebook, f"{x}.json"))
    data["embeddings"] = data.id.apply(lambda x: os.path.join(embedding, f"{x}.pt"))
    data["length"] = data.cell_order.str.split().str.len()

    # Fix the random seed for sampling from the longer sequences to ensure the
    # reproducibility.
    np.random.seed(args.random_seed)
    multiple = pd.concat([data[data.length > args.max_length]] * args.num_repeats)
    multiple.cell_order = multiple.cell_order.apply(
        lambda x: sample_cells(x, args.max_length)
    )

    data = pd.concat((data[data.length <= args.max_length], multiple))
    data = data.sort_values("length").reset_index(drop=True)

    dataloader = DataLoader(
        AI4CodeDataset(data, args.max_length),
        batch_size=args.batch_size,
        num_workers=args.num_workers or os.cpu_count(),
        collate_fn=DataCollatorForEmbeddingInputs(pad_to_multiple_of=8),
        persistent_workers=True,
    )
    return data, dataloader


@torch.inference_mode()
def main(args: argparse.Namespace):
    data, dataloader = prepare_dataset(args)
    model = AutoModelForTokenClassification.from_pretrained(args.model).cuda().eval()
    model.to(torch.float16 if args.use_fp16 else torch.float32)

    example_iter, outputs = data.itertuples(), []
    for batch in tqdm.tqdm(dataloader):
        # Move the batch tensors to CUDA memory.
        batch = {k: v.cuda() for k, v in batch.items()}
        logits = model(**batch).logits.float().squeeze(2)
        for logits, mask in zip(logits, batch["attention_mask"]):
            example = next(example_iter)
            cells = example.cell_order.split()
            orders = [cells[i] for i in logits[mask.bool()].argsort().tolist()]
            outputs.append({"id": example.id, "cell_order": " ".join(orders)})
    pd.DataFrame(outputs).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", default="suborders.csv")
    parser.add_argument("--init-orders", default="resources/ai4code/train_orders.csv")
    parser.add_argument("--notebook-dir", default="resources/ai4code/train_cleaned")
    parser.add_argument("--embedding-dir", default="resources/embeddings")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--num-repeats", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--use-fp16", default=False, action="store_true")
    main(parser.parse_args())
