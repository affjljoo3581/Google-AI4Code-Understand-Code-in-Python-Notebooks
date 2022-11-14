from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main(args: argparse.Namespace):
    submission = pd.read_csv(args.submission, index_col="id")
    orders = pd.read_csv(args.orders, index_col="id").loc[submission.index]

    inversions, transitions = 0, 0
    for _, pred, label in pd.concat((submission, orders), axis=1).itertuples():
        pred, label = pred.split(), label.split()
        pred = np.array([label.index(cell_name) for cell_name in pred])
        comparing = np.sign(np.expand_dims(pred, 1) - np.expand_dims(pred, 0))

        inversions += np.triu(comparing == 1).sum()
        transitions += len(pred) * (len(pred) - 1)
    print(1 - 4 * inversions / transitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--orders", default="resources/validation/val_orders.csv")
    main(parser.parse_args())
