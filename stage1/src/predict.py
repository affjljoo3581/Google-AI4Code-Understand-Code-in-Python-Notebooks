from __future__ import annotations

import argparse
import json
import os

import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from data import TextDataset


def prepare_dataset(args: argparse.Namespace) -> tuple[list[str], DataLoader]:
    # Read the notebooks and extract all texts with their notebook names.
    names, texts = [], []
    for filename in tqdm.tqdm(os.listdir(args.notebook_dir)):
        name = os.path.splitext(filename)[0]
        with open(os.path.join(args.notebook_dir, filename)) as fp:
            notebook = json.load(fp)
        for text in notebook["source"].values():
            names.append(name)
            texts.append(" ".join(text.split()))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataloader = DataLoader(
        TextDataset(texts, tokenizer, args.max_length),
        batch_size=args.batch_size,
        num_workers=args.num_workers or os.cpu_count(),
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        persistent_workers=True,
    )
    return names, dataloader


@torch.inference_mode()
def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare the dataset and pretrained model.
    names, dataloader = prepare_dataset(args)
    model = AutoModel.from_pretrained(args.model).cuda().eval()
    model.to(torch.float16 if args.use_fp16 else torch.float32)

    index, embeddings = 0, []
    for batch in tqdm.tqdm(dataloader):
        # Move the batch tensors to CUDA memory.
        batch = {k: v.cuda() for k, v in batch.items()}
        for embedding in model(**batch).last_hidden_state[:, 0]:
            if index > 0 and names[index - 1] != names[index]:
                # If generating the sentence embeddings in the current notebook is done,
                # then we will gather the embeddings for saving to the file, and clear
                # the embedding buffer for new notebook cells.
                filename = os.path.join(args.output_dir, f"{names[index - 1]}.pt")
                torch.save(torch.stack(embeddings), filename)
                embeddings = []
            embeddings.append(embedding)
            index += 1

    # Save the last notebook embeddings if the embedding buffer is not empty.
    if embeddings:
        filename = os.path.join(args.output_dir, f"{names[-1]}.pt")
        torch.save(torch.stack(embeddings), filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--notebook-dir", default="resources/ai4code/train_cleaned")
    parser.add_argument("--output-dir", default="embeddings")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--use-fp16", default=False, action="store_true")
    main(parser.parse_args())
