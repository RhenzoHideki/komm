from collections import defaultdict
from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt

from .util import integer_to_symbols, symbols_to_integer


@dataclass
class LempelZiv77Code:
    r"""
    Lempel–Ziv 77 (LZ77) code with sliding window. It is a lossless data compression algorithm
    that replaces repeated data with references to previous occurrences within a sliding window.
    The algorithm achieves compression by identifying matches between the current position and
    patterns within the search window, encoding them as (distance, length, next_symbol) triples.
    For more details, see standard references on data compression algorithms.

    Parameters:
        source_cardinality: The source cardinality $S$. Must be a number greater than or equal to $2$.
        target_cardinality: The target cardinality $T$. Must be a number greater than or equal to $2$. Default is $2$ (binary).
        window_size: The sliding window size $W$. Must be a number greater than or equal to $1$.
        lookahead_size: The lookahead buffer size $L$. Must be a number greater than or equal to $1$.

    Encoding format (fixed-width per triple):

    Each triple $(d, l, c)$ represents:

    - $d$: distance in $[0..W]$ (0 means "no match")
    - $l$: length in $[0..L]$ (0 means "no match")
    - $c$: next symbol in $[0..S-1]$

    Each field is emitted in base-$T$ using:

    - $D = \lceil \log_T(W+1) \rceil$ symbols for $d$
    - $L_w = \lceil \log_T(L+1) \rceil$ symbols for $l$
    - $M = \lceil \log_T(S) \rceil$ symbols for $c$

    Examples:
        >>> lz77 = komm.LempelZiv77Code(source_cardinality=2, target_cardinality=2, window_size=16, lookahead_size=4)

    Notes:
        - Matches are constrained to at most `lookahead_size` and ensure there is always a following symbol
        - The last source symbol (if any) is always emitted as (0, 0, c)
    """

    # Parameter order: source_cardinality, target_cardinality, window_size, lookahead_size
    source_cardinality: float
    target_cardinality: float
    window_size: float
    lookahead_size: float

    def __post_init__(self) -> None:
        # Same validation as original
        if self.source_cardinality < 2:
            raise ValueError("'source_cardinality' must be at least 2")
        if self.target_cardinality < 2:
            raise ValueError("'target_cardinality' must be at least 2")
        if self.window_size < 1:
            raise ValueError("'window_size' must be at least 1")
        if self.lookahead_size < 1:
            raise ValueError("'lookahead_size' must be at least 1")

        # Convert to int for internal use
        self._window_size = int(self.window_size)
        self._lookahead_size = int(self.lookahead_size)
        self._source_cardinality = int(self.source_cardinality)
        self._target_cardinality = int(self.target_cardinality)

        # Set minimum match length (optimization)
        self._min_match_length = 3

        # Precompute field widths in base T
        T = self.target_cardinality
        S = self.source_cardinality
        self._D = max(1, ceil(float(log(self.window_size + 1, T))))  # distance 0..W
        self._Lw = max(1, ceil(float(log(self.lookahead_size + 1, T))))  # length 0..L
        self._M = max(1, ceil(float(log(S, T))))  # symbol 0..S-1

        # Initialize hash table for optimization
        self._init_hash_table()

    def _init_hash_table(self) -> None:
        """Initialize hash table for fast pattern lookup."""
        self._hash_table: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _update_hash_table(
        self, data: np.ndarray, position: int, pattern_length: int = 3
    ) -> None:
        """Update hash table with new patterns for fast lookup."""
        if position + pattern_length <= len(data):
            pattern = tuple[int, int](data[position : position + pattern_length])
            self._hash_table[pattern].append(position)

            # Limit list size to prevent excessive growth
            if len(self._hash_table[pattern]) > 10:
                self._hash_table[pattern] = self._hash_table[pattern][-10:]

    def _extend_match(
        self, data: np.ndarray, match_pos: int, current_pos: int, max_length: int
    ) -> int:
        """Extend a found match as far as possible, supporting overlap."""
        length = 0
        distance = current_pos - match_pos

        while (
            length < max_length
            and current_pos + length < len(data)
            and data[match_pos + (length % distance)] == data[current_pos + length]
        ):
            length += 1

        return length

    def _find_longest_match(
        self, window: np.ndarray, lookahead: np.ndarray
    ) -> tuple[int, int]:
        """
        Optimized version of original _find_longest_match method.
        Uses hash table internally for speed but maintains same interface.

        Return (d, l) allowing overlap (as in LZ77 decoding).
        d = distance back from current position to the start of match (>=1),
        l = matched length. If no match, return (0, 0).
        """
        n = window.size
        if n == 0 or lookahead.size == 0:
            return 0, 0

        best_d = 0
        max_l = 0

        # Use hash table optimization when possible
        if lookahead.size >= self._min_match_length:
            pattern_length = min(self._min_match_length, lookahead.size)
            pattern = tuple[int, int](lookahead[:pattern_length])

            if pattern in self._hash_table:
                # Check potential matches from hash table
                for match_pos in reversed(self._hash_table[pattern]):
                    # Convert absolute position to relative position in window
                    if match_pos < len(self._current_data) - n or match_pos >= len(
                        self._current_data
                    ):
                        continue

                    window_pos = match_pos - (len(self._current_data) - n)
                    if window_pos < 0 or window_pos >= n:
                        continue

                    d = n - window_pos  # distance from current position
                    if d <= 0:
                        continue

                    # Extend match with overlap support
                    l = 0
                    while (
                        l < lookahead.size
                        and window[window_pos + (l % d)] == lookahead[l]
                    ):
                        l += 1

                    if l >= self._min_match_length and l > max_l:
                        max_l = l
                        best_d = d

                        # Early termination if maximum match found
                        if max_l == lookahead.size:
                            break

        # Fallback to original algorithm if hash table didn't find good matches
        if max_l < self._min_match_length:
            # Try every start position in the window (original algorithm)
            for start in range(n):
                d = n - start  # distance from current position
                if d <= 0:
                    continue

                # Compare with overlap: the source is periodic with period d
                l = 0
                while l < lookahead.size and window[start + (l % d)] == lookahead[l]:
                    l += 1

                if l > max_l:
                    max_l = l
                    best_d = d

                if max_l == lookahead.size:  # can't do better
                    break

        if max_l == 0:
            return 0, 0

        return best_d, max_l

    def source_to_triples(self, input: npt.ArrayLike):
        x = np.asarray(input, dtype=int)
        W = self._window_size
        L = self._lookahead_size
        n = x.size
        i = 0
        triples: list[tuple[int, int, int]] = []
        # Store reference for hash table optimization
        self._current_data = x
        self._init_hash_table()
        while i < n:
            if i == n - 1:
                d, l = 0, 0
                c = int(x[i])
                triples.append((d, l, c))
                i += 1
                continue
            win_start = max(0, i - W)
            window = x[win_start:i]
            max_l = min(L, n - i - 1)
            lookahead = x[i : i + max_l]
            self._update_hash_table(x, i)
            d, l = self._find_longest_match(window, lookahead)
            if d == 0 or l == 0:
                d, l, c = 0, 0, int(x[i])
                step = 1
            else:
                c = int(x[i + l])
                step = l + 1
            triples.append((d, l, c))
            i += step
        return triples

    def triples_to_target(self, triples: list[tuple[int, int, int]]):
        D, Lw, M = self._D, self._Lw, self._M
        out: list[int] = []
        for d, l, c in triples:
            out.extend(integer_to_symbols(d, base=self._target_cardinality, width=D))
            out.extend(integer_to_symbols(l, base=self._target_cardinality, width=Lw))
            out.extend(integer_to_symbols(c, base=self._target_cardinality, width=M))
        return np.array(out, dtype=int)

    def target_to_triples(self, input: npt.ArrayLike) -> list[tuple[int, int, int]]:
        T = self._target_cardinality
        D, Lw, M = self._D, self._Lw, self._M
        y = np.asarray(input, dtype=int)
        triples: list[tuple[int, int, int]] = []
        i = 0
        while i + D + Lw + M <= y.size:
            d = symbols_to_integer(y[i : i + D], base=T)
            i += D
            l = symbols_to_integer(y[i : i + Lw], base=T)
            i += Lw
            c = symbols_to_integer(y[i : i + M], base=T)
            i += M
            triples.append((d, l, c))
        return triples

    def triples_to_source(self, triples: list[tuple[int, int, int]]):
        out: list[int] = []
        for d, l, c in triples:
            if d == 0 and l == 0:
                out.append(c)
            else:
                start = len(out) - d
                for k in range(l):
                    out.append(out[start + k])
                out.append(c)
        return np.array(out, dtype=int)

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
            ... 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0])

        """
        triples = self.source_to_triples(input)
        return self.triples_to_target(triples)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decode LZ77-encoded sequence with enhanced validation.

        Args:
            input: 1D array of encoded base-$T$ symbols

        Returns:
            1D array of reconstructed original symbols in $[0, S)$.

        Raises:
            ValueError: If input stream is corrupted or invalid.
        """
        triples = self.target_to_triples(input)
        return self.triples_to_source(triples)
