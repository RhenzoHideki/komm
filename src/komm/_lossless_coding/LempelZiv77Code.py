from collections import defaultdict
from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

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

    def encode(
        self, input: npt.ArrayLike, verbose: bool = False
    ) -> npt.NDArray[np.integer]:
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
        x = np.asarray(input, dtype=int)
        if x.ndim != 1:
            raise ValueError("'input' must be a 1D array")
        if x.size == 0:
            return np.array([], dtype=int)
        if np.any((x < 0) | (x >= self.source_cardinality)):
            raise ValueError("input symbols out of range")

        # Store reference for hash table optimization
        self._current_data = x

        # Reset hash table for new encoding
        self._init_hash_table()

        T = self._target_cardinality
        W = self._window_size
        L = self._lookahead_size
        D, Lw, M = self._D, self._Lw, self._M

        out: list[int] = []
        n = x.size
        i = 0

        # Show encoding header if verbose
        if verbose:
            print(f"Encoding {n} symbols with window_size={W}, lookahead_size={L}")
            print("Generating triples:")

        pbar = tqdm(total=n, desc="Compressing LZ77", delay=2.5)

        while i < n:
            # Remaining symbols; if only one left, no following 'c' exists => emit (0,0,c)
            if i == n - 1:
                d, l = 0, 0
                c = int(x[i])

                if verbose:
                    print(f"Encoding triple: (d={d}, l={l}, c={c}) at position {i}")

                out.extend(integer_to_symbols(d, base=T, width=D))
                out.extend(integer_to_symbols(l, base=T, width=Lw))
                out.extend(integer_to_symbols(c, base=T, width=M))
                i += 1
                pbar.update(1)
                continue

            # Build window and lookahead respecting limits.
            win_start = max(0, i - W)
            window = x[win_start:i]  # size up to W

            # Ensure there is always a following symbol c => max length <= n - i - 1
            max_l = min(L, n - i - 1)
            lookahead = x[i : i + max_l]

            # Update hash table for current position (optimization)
            self._update_hash_table(x, i)

            d, l = self._find_longest_match(window, lookahead)

            if d == 0 or l == 0:
                # No match: (0,0,c) with c = x[i]
                d, l, c = 0, 0, int(x[i])
                step = 1
            else:
                # Match found; c is the symbol after the match
                c = int(x[i + l])
                step = l + 1  # consume l matched + 1 following char

            # Simple verbose output - just like the original
            if verbose:
                print(f"Encoding triple: (d={d}, l={l}, c={c}) at position {i}")

            out.extend(integer_to_symbols(d, base=T, width=D))
            out.extend(integer_to_symbols(l, base=T, width=Lw))
            out.extend(integer_to_symbols(c, base=T, width=M))

            i += step
            pbar.update(step)

        pbar.close()

        if verbose:
            print(f"Encoding complete: {n} input symbols -> {len(out)} output symbols")

        return np.array(out, dtype=int)

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
        T = self._target_cardinality
        S = self._source_cardinality
        D, Lw, M = self._D, self._Lw, self._M

        y = np.asarray(input, dtype=int)
        if y.ndim != 1:
            raise ValueError("'input' must be a 1D array")
        if y.size == 0:
            return np.array([], dtype=int)
        if np.any((y < 0) | (y >= T)):
            raise ValueError("encoded symbols out of range for base T")

        out: list[int] = []
        i = 0
        triple_syms = D + Lw + M

        pbar = tqdm(total=y.size, desc="Decompressing LZ77", delay=2.5)

        while i + triple_syms <= y.size:
            d = symbols_to_integer(y[i : i + D], base=T)
            i += D
            l = symbols_to_integer(y[i : i + Lw], base=T)
            i += Lw
            c = symbols_to_integer(y[i : i + M], base=T)
            i += M

            pbar.update(triple_syms)

            # sanity checks against malformed streams
            if c >= S:
                raise ValueError(f"Invalid stream: symbol c={c} not in [0,{S - 1}]")
            if d == 0 and l != 0:
                raise ValueError("Invalid stream: d=0 must imply l=0")
            if l > self.lookahead_size:
                raise ValueError("Invalid stream: length exceeds lookahead_size")

            if d == 0 and l == 0:
                out.append(c)
                continue

            # match copy with overlap (standard LZ77 behavior)
            start = len(out) - d
            if start < 0:
                raise ValueError("Invalid stream: distance exceeds produced output")

            for k in range(l):
                out.append(
                    out[start + k]
                )  # may reference elements appended in this loop

            out.append(c)

        pbar.close()

        if i != y.size:
            raise ValueError(
                "Invalid stream: leftover symbols not forming a complete triple"
            )

        return np.array(out, dtype=int)
        return np.array(out, dtype=int)
