from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import pandas as pd
import tqdm

_GENERATED_NAMES = set()


def generate_unique_name(length: int) -> str:
    while True:
        name = "".join(random.choices("0123456789abcdef", k=length))
        if name not in _GENERATED_NAMES:
            _GENERATED_NAMES.add(name)
            return name


def convert_jupyter_notebook(
    cells: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, str]], list[str]]:
    sources, cell_types, cell_names = {}, {}, []
    for cell in cells:
        if isinstance(cell, str) or cell["cell_type"] not in ["code", "markdown"]:
            continue

        # Some lower-version jupyter notebooks have different cell structure and content
        # attribute keys (e.g. `input` instead of `source`), therefore we check both
        # `input` and `source` keys with exception handling.
        content = ""
        if "input" in cell and cell["input"] is not None:
            content = "".join(cell["input"])
        elif "source" in cell and cell["source"] is not None:
            content = "".join(cell["source"])

        cell_name = generate_unique_name(10)
        cell_names.append(cell_name)

        sources[cell_name] = content
        cell_types[cell_name] = cell["cell_type"]
    return {"cell_type": cell_types, "source": sources}, cell_names


def main(args: argparse.Namespace):
    license_filters = args.license_filters.split(",")
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    parsed_outputs = []
    for filename in tqdm.tqdm(os.listdir(args.dataset_dir)):
        with open(os.path.join(args.dataset_dir, filename)) as fp:
            examples = [json.loads(line) for line in fp]

        # Extract the notebook contents from the json examples. Note that some examples
        # have wrong-encoded contents and even do not have the content, so we wrap the
        # parsing part with try-except block.
        notebooks = []
        for example in examples:
            try:
                notebooks.append(json.loads(example["content"]))
            except Exception:
                pass

        for notebook in notebooks:
            if "license" not in notebook or notebook["license"] not in license_filters:
                continue
            if "worksheets" in notebook:
                for worksheet in notebook["worksheets"]:
                    parsed_outputs.append(convert_jupyter_notebook(worksheet["cells"]))
            elif "cells" in notebook:
                parsed_outputs.append(convert_jupyter_notebook(notebook["cells"]))

    cell_orders = []
    for notebook, cell_names in parsed_outputs:
        notebook_name = generate_unique_name(16)
        filename = os.path.join(args.output_dir, "train", f"{notebook_name}.json")

        with open(filename, "w") as fp:
            json.dump(notebook, fp)
        cell_orders.append({"id": notebook_name, "cell_order": " ".join(cell_names)})

    cell_orders = pd.DataFrame(cell_orders)
    cell_orders.to_csv(os.path.join(args.output_dir, "train_orders.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="github-jupyter/data")
    parser.add_argument("--output-dir", default="codeparrot")
    parser.add_argument("--license-filters", default="mit,bsd-3-clause,apache-2.0")
    main(parser.parse_args())
