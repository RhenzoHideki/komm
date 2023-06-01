import numpy as np
import pytest

import komm


def test_lfsr_sequence():
    lfsr = komm.LFSRSequence(feedback_polynomial=komm.BinaryPolynomial.from_exponents([5, 2, 0]))
    assert np.array_equal(
        lfsr.bit_sequence, [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    )
    lfsr = komm.LFSRSequence(feedback_polynomial=0b10000001001)
    assert np.array_equal(
        lfsr.bit_sequence[:200], komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200)
    )


@pytest.mark.parametrize("num_states", range(2, 16))
def test_lfsr_mls(num_states):
    lfsr = komm.LFSRSequence.maximum_length_sequence(num_states)
    assert num_states == lfsr.feedback_polynomial.degree
    assert lfsr.length == 2**lfsr.feedback_polynomial.degree - 1
    cyclic_autocorrelation_expected = np.full(lfsr.length, fill_value=-1, dtype=int)
    cyclic_autocorrelation_expected[0] = lfsr.length
    assert np.array_equal(lfsr.cyclic_autocorrelation(), cyclic_autocorrelation_expected)
