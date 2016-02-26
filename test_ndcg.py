import ndcg

import numpy as np


def assert_eq(a, b):
    assert abs(a - b) < 1e-7, '%f != %f' % (a, b)


def test_ndcg():
    x = []
    k = 4
    assert_eq(ndcg.dcg(x, k), 0)

    x = [1, 1, 1]
    k = 5
    assert_eq(ndcg.dcg(x, k), 1 * (1 + 1 / np.log2(3) + 1 / np.log2(4)))
    k = 1
    assert_eq(ndcg.dcg(x, k), 1)

    x = [[1, 1, 1], [1, 2, 3]]
    k = 3
    res = ndcg.dcg(x, k)
    assert_eq(res[0], 1 * (1 + 1 / np.log2(3) + 1 / np.log2(4)))
    assert_eq(res[1], 1 + 3 / np.log2(3) + 7 / np.log2(4))

    x = [1, 2, 3]
    opt_x = [3, 2, 1]
    k = 3
    assert_eq(ndcg.ndcg(x, k), ndcg.dcg(x, k) / ndcg.dcg(opt_x, k))

    x = [1, 1, 1]
    assert_eq(ndcg.ndcg(x, k), 1)

    x = []
    assert_eq(ndcg.ndcg(x, k), 0)
