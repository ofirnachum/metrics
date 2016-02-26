"""NDCG - normalized discounted comulative gain.

"""
import numpy as np


def dcg(rel, k):
    rel = np.array(rel)
    if rel.ndim == 1:
        rel = np.expand_dims(rel, axis=0)
        return_scalar = True
    else:
        return_scalar = False
    assert rel.ndim == 2

    rel = rel[:, :k]
    if rel.shape[1] == 0:
        return 0.0
    ret = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.shape[1] + 2)),
                 axis=1)
    if return_scalar:
        return ret[0]
    return ret


def _max_dcg(rel, k):
    return dcg(np.sort(np.array(rel))[..., ::-1], k)


def ndcg(rel, k):
    max_dcg = _max_dcg(rel, k)
    return dcg(rel, k) / np.maximum(max_dcg, 0.00001)
