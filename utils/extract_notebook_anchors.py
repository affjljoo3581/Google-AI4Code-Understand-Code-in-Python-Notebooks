from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import tqdm


def main(args: argparse.Namespace):
    anchors = []
    for filename in tqdm.tqdm(os.listdir(args.notebook_dir)):
        with open(os.path.join(args.notebook_dir, filename)) as fp:
            notebook = json.load(fp)
        cells = " ".join([x for x, y in notebook["cell_type"].items() if y == "code"])
        anchors.append({"id": os.path.splitext(filename)[0], "cell_order": cells})
    pd.DataFrame(anchors).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-dir", default="resources/validation/val_cleaned")
    parser.add_argument("--output", default="anchors.csv")
    main(parser.parse_args())
