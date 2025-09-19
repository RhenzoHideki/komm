from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt

from .util import integer_to_symbols, symbols_to_integer


@dataclass
class LempelZiv77Code:
    r"""
    Lempel–Ziv 77 (LZ77 or LZ1) code.

    It is a lossless data compression algorithm
    that replaces repeated data with references to previous occurrences within a sliding window.
    The algorithm achieves compression by identifying matches between the current position and
    patterns within the search window, encoding them as (distance, length, next_symbol) triples.
    For more details, see standard references on data compression algorithms.

    Parameters:
        source_cardinality: The source cardinality $S$. Must be an integer greater than or equal to $2$.
        target_cardinality: The target cardinality $T$. Must be an integer greater than or equal to $2$. Default is $2$ (binary).
        window_size: Sliding window size $W$. Must be an integer greater than or equal to $1$.
        lookahead_size: Lookahead buffer size $L$. Must be an integer greater than or equal to $1$.

    Encoding format (fixed-width per triple):
        d: distance in [0..W]  (0 means "no match")
        l: length   in [0..L]  (0 means "no match")
        c: next symbol in [0..S-1]

        Each field is emitted in base-T using:
            D = ceil(log(W+1, T)) symbols for d
            Lw = ceil(log(L+1, T)) symbols for l
            M = ceil(log(S,   T)) symbols for c


    Examples:
        >>> lz77 = komm.LempelZiv77Code(source_cardinality=2, target_cardinality=2, window_size=16, lookahead_size=4)
    """

    source_cardinality: int
    window_size: int
    lookahead_size: int
    target_cardinality: int

    def __post_init__(self) -> None:
        if self.source_cardinality < 2:
            raise ValueError("'source_cardinality' must be at least 2")
        if self.target_cardinality < 2:
            raise ValueError("'target_cardinality' must be at least 2")
        if self.window_size < 1:
            raise ValueError("'window_size' must be at least 1")
        if self.lookahead_size < 1:
            raise ValueError("'lookahead_size' must be at least 1")

        # Precompute field widths in base T.
        T: int = self.target_cardinality
        S: int = self.source_cardinality
        self._D: int = max(1, ceil(float(log(self.window_size + 1, T))))
        self._Lw: int = max(1, ceil(float(log(self.lookahead_size + 1, T))))
        self._M: int = max(1, ceil(float(log(S, T))))

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encode a sequence of source symbols using optimized LZ77, emitting a base-T stream.

        Args:
            input: 1D array with elements in $[0, S)$
            verbose: If True, prints each triple (d, l, c) as it's generated

        Returns:
            1D array of base-$T$ symbols representing concatenated triples $(d, l, c)$.

        Raises:
            ValueError: If input is not a 1D array or contains symbols outside valid range.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ... source_cardinality=2,
            ... target_cardinality=2,
            ... window_size=16,
            ... lookahead_size=4
            ... )
            >>> lz77.encode(np.zeros(15, dtype=int))
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0])

        """
        triples: list[tuple[int, int, int]] = self.source_to_triples(input)
        return self.triples_to_target(triples)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decodes a sequence of encoded symbols using the LZ77 decoding algorithm.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code). Also, the sequence must be a valid output of the `encode` method.

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code).

        Examples:
            >>> message = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0])
            >>> lz77 = komm.LempelZiv77Code(
            ... source_cardinality=2,
            ... target_cardinality=2,
            ... window_size=16,
            ... lookahead_size=4
            ... )
            >>> lz77.decode(message)
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        triples: list[tuple[int, int, int]] = self.target_to_triples(input)
        return self.triples_to_source(triples)

    def source_to_triples(self, input: npt.ArrayLike) -> list[tuple[int, int, int]]:
        """Convert source symbols to list of LZ77 triples (d, l, c)."""
        x: npt.NDArray[np.integer] = np.asarray(input, dtype=int)
        if x.ndim != 1:
            raise ValueError("'input' must be a 1D array")
        if x.size == 0:
            return []
        if np.any((x < 0) | (x >= self.source_cardinality)):
            raise ValueError("input symbols out of range")

        W: int = self.window_size
        L: int = self.lookahead_size
        triples: list[tuple[int, int, int]] = []
        n: int = x.size
        i: int = 0

        while i < n:
            if i == n - 1:
                triples.append((0, 0, int(x[i])))
                i += 1
                continue

            win_start: int = max(0, i - W)
            window: npt.NDArray[np.integer] = x[win_start:i]
            max_l: int = min(L, n - i - 1)
            lookahead: npt.NDArray[np.integer] = x[i : i + max_l]

            d, l = self._find_longest_match(window, lookahead)

            if d == 0 or l == 0:
                d, l, c = 0, 0, int(x[i])
                step: int = 1
            else:
                c = int(x[i + l])
                step = l + 1

            triples.append((d, l, c))
            i += step

        return triples

    def _find_longest_match(
        self, window: npt.NDArray[np.integer], lookahead: npt.NDArray[np.integer]
    ) -> tuple[int, int]:
        """Find longest match in window for lookahead buffer."""
        n: int = window.size
        if n == 0 or lookahead.size == 0:
            return 0, 0

        best_d: int = 0
        max_l: int = 0

        for start in range(n):
            d: int = n - start
            if d <= 0:
                continue

            l: int = 0
            while l < lookahead.size and window[start + (l % d)] == lookahead[l]:
                l += 1

            if l > max_l:
                max_l = l
                best_d = d

            if max_l == lookahead.size:
                break

        if max_l == 0:
            return 0, 0

        return best_d, max_l

    def triples_to_target(
        self, triples: list[tuple[int, int, int]]
    ) -> npt.NDArray[np.integer]:
        """Convert list of triples to transmission symbol stream."""
        T: int = self.target_cardinality
        D: int = self._D
        Lw: int = self._Lw
        M: int = self._M
        out: list[int] = []

        for d, l, c in triples:
            out.extend(integer_to_symbols(d, base=T, width=D))
            out.extend(integer_to_symbols(l, base=T, width=Lw))
            out.extend(integer_to_symbols(c, base=T, width=M))

        return np.array(out, dtype=int)

    def target_to_triples(self, input: npt.ArrayLike) -> list[tuple[int, int, int]]:
        """Parse transmission stream back to list of triples."""
        T: int = self.target_cardinality
        D: int = self._D
        Lw: int = self._Lw
        M: int = self._M
        y: npt.NDArray[np.integer] = np.asarray(input, dtype=int)

        if y.ndim != 1:
            raise ValueError("'input' must be a 1D array")
        if y.size == 0:
            return []
        if np.any((y < 0) | (y >= T)):
            raise ValueError("encoded symbols out of range for base T")

        triples: list[tuple[int, int, int]] = []
        i: int = 0
        triple_syms: int = D + Lw + M

        while i + triple_syms <= y.size:
            d: int = symbols_to_integer(y[i : i + D], base=T)
            i += D
            l: int = symbols_to_integer(y[i : i + Lw], base=T)
            i += Lw
            c: int = symbols_to_integer(y[i : i + M], base=T)
            i += M
            triples.append((d, l, c))

        if i != y.size:
            raise ValueError(
                "Invalid stream: leftover symbols not forming a complete triple"
            )

        return triples

    def triples_to_source(
        self, triples: list[tuple[int, int, int]]
    ) -> npt.NDArray[np.integer]:
        """Reconstruct original message from list of triples."""
        S: int = self.source_cardinality
        out: list[int] = []

        for d, l, c in triples:
            if c >= S:
                raise ValueError(f"Invalid stream: symbol c={c} not in [0,{S - 1}]")
            if d == 0 and l != 0:
                raise ValueError("Invalid stream: d=0 must imply l=0")
            if l > self.lookahead_size:
                raise ValueError("Invalid stream: length exceeds lookahead_size")

            if d == 0 and l == 0:
                out.append(c)
                continue

            start: int = len(out) - d
            if start < 0:
                raise ValueError("Invalid stream: distance exceeds produced output")

            for k in range(l):
                out.append(out[start + k])

            out.append(c)

        return np.array(out, dtype=int)
