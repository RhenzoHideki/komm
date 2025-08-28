import numpy as np
import pytest
from itertools import product
import komm


def test_lz77_empty_and_single():
    code = komm.LempelZiv77Code(2, 2, window_size=16, lookahead_size=4)
    assert code.encode([]).tolist() == []
    assert code.decode([]).tolist() == []
    for symbol in [0, 1]:
        msg = [symbol]
        np.testing.assert_equal(code.decode(code.encode(msg)), msg)


@pytest.mark.parametrize("source_cardinality", [2, 4, 8])
def test_lz77_roundtrip_random(source_cardinality):
    code = komm.LempelZiv77Code(source_cardinality, 2, window_size=32, lookahead_size=8)
    msg = np.random.randint(0, source_cardinality, size=100).tolist()
    np.testing.assert_equal(code.decode(code.encode(msg)), msg)


@pytest.mark.parametrize("k", range(2, 6))
def test_lz77_zero_runs(k):
    code = komm.LempelZiv77Code(2, 2, window_size=32, lookahead_size=8)
    msg = []
    for r in range(1, k + 1):
        msg.extend([0] * r)
    compressed = code.encode(msg)
    np.testing.assert_equal(code.decode(compressed), msg)


@pytest.mark.parametrize("k", range(2, 5))
def test_lz77_worst_case(k):
    code = komm.LempelZiv77Code(2, 2, window_size=64, lookahead_size=16)
    msg = []
    for r in range(1, k + 1):
        for bits in product([0, 1], repeat=r):
            msg.extend(bits)
    compressed = code.encode(msg)
    np.testing.assert_equal(code.decode(compressed), msg)
    assert len(compressed) <= len(msg) * 10  # sanity bound
