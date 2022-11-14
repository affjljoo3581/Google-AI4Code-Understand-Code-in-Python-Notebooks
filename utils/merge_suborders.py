from __future__ import annotations

import argparse
import itertools
from typing import Optional

import numpy as np
import pandas as pd
import tqdm


def merge_subranks(
    size: int, subranks: list[list[int]], anchors: Optional[list[int]] = None
) -> list[int]:
    conditions = np.zeros((size, size))
    for subrank in subranks:
        for i, j in itertools.product(range(len(subrank)), range(len(subrank))):
            conditions[subrank[i], subrank[j]] += np.sign(j - i)

    rank = anchors.copy() if anchors else []
    unordered = [i for i in np.abs(conditions).sum(1).argsort()[::-1] if i not in rank]

    for i in unordered:
        mask = 2 * np.triu(np.ones((len(rank) + 1, len(rank)))) - 1
        rank.insert((mask @ conditions[i][rank]).argmax(), i)
    return rank


def main(args: argparse.Namespace):
    init_orders = pd.read_csv(args.init_orders, index_col="id")
    suborders = pd.read_csv(args.suborders)
    total_anchors = pd.read_csv(args.anchors, index_col="id")

    merged_orders = []
    for name, cell_orders in tqdm.tqdm(suborders.groupby("id").cell_order):
        cells = list(init_orders.loc[name].cell_order.split())
        cell_orders = [[cells.index(y) for y in x.split()] for x in cell_orders]
        anchors = [cells.index(x) for x in total_anchors.loc[name].cell_order.split()]

        # After merging the subranks to be a global rank, the sorted cell names would be
        # reconstructed by mapping the sorted indices to their cell names.
        merged_rank = merge_subranks(len(cells), cell_orders, anchors)
        merged_rank = " ".join([cells[i] for i in merged_rank])
        merged_orders.append({"id": name, "cell_order": merged_rank})

    pd.DataFrame(merged_orders).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-orders", default="resources/validation/val_orders.csv")
    parser.add_argument("--suborders", default="suborders.csv")
    parser.add_argument("--anchors", default="anchors.csv")
    parser.add_argument("--output", default="submission.csv")
    main(parser.parse_args())
