import os
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance


class Glove(object):
    DISTANCES = {
        "cosine": distance.cosine,
        "euclidean": distance.euclidean,
    }

    def __init__(self):
        self._embeddings = {}

    @classmethod
    def load(cls, path: os.PathLike):
        res = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                res._embeddings[word] = coefs
        return res

    def __getitem__(self, item):
        return self._embeddings[item]

    def get(self, item, default=None):
        return self._embeddings.get(item, default)

    def __contains__(self, item):
        return item in self._embeddings

    def __len__(self):
        return len(self._embeddings)

    def query(
        self, embedding: NDArray[np.float32], k: int = 1, distance: Literal["cosine", "euclidean"] = "cosine"
    ) -> list[str]:
        dist = self.DISTANCES[distance]
        return sorted(self._embeddings, key=lambda x: dist(embedding, self._embeddings[x]))[:k]
