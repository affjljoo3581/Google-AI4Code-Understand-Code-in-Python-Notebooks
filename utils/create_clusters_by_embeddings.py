import argparse
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.cluster import KMeans


def main(args: argparse.Namespace):
    ancestors = pd.read_csv(args.ancestors)
    names = ancestors.id.to_list()

    embeddings = []
    for name in tqdm.tqdm(names):
        filename = os.path.join(args.embedding_dir, f"{name}.pt")
        embedding = torch.load(filename, map_location="cpu").float().mean(0)
        embeddings.append(embedding)
    embeddings = F.normalize(torch.stack(embeddings), dim=1)

    anchors = []
    for _, notebooks in ancestors.groupby("ancestor_id"):
        anchors.append(embeddings[notebooks.index].mean(0))

    random.shuffle(anchors)
    anchors = torch.stack(anchors[: embeddings.size(0) // 10]).numpy()

    kmeans = KMeans(n_clusters=embeddings.size(0) // 10, init=anchors, verbose=True)
    kmeans.fit(embeddings.numpy())

    output = pd.DataFrame({"id": names, "cluster": kmeans.labels_})
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ancestors")
    parser.add_argument("--embedding-dir", default="resources/embeddings")
    parser.add_argument("--output", default="train_clusters.csv")
    main(parser.parse_args())
