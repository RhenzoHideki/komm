from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .util import integer_to_symbols, symbols_to_integer


@dataclass
class LempelZiv77Code:
    r"""
    Lempel–Ziv 77 (LZ77) code with a sliding window and fixed-width triple (d, l, c) output.

    Parameters:
        source_cardinality: The source alphabet size S (>= 2).
        target_cardinality: The target alphabet size T (>= 2). Default is 2 (binary).
        window_size:       Sliding window size W (>= 1).
        lookahead_size:    Lookahead buffer size L (>= 1).

    Encoding format (fixed-width per triple):
        d: distance in [0..W]  (0 means "no match")
        l: length   in [0..L]  (0 means "no match")
        c: next symbol in [0..S-1]

        Each field is emitted in base-T using:
            D = ceil(log(W+1, T)) symbols for d
            Lw = ceil(log(L+1, T)) symbols for l
            M = ceil(log(S,   T)) symbols for c

    Notes:
        * We constrain matches to at most `lookahead_size` and ensure there is
          always a following symbol `c`. That is, at position i we search for a
          match length up to `min(L, n - i - 1)`. The last source symbol (if any)
          is always emitted as (0, 0, c).
        * This is a straightforward O(W * L) search per step (naive longest-match).
          It’s simple and correct, though not optimized.

    Examples:
        >>> lz77 = LempelZiv77Code(2)  # binary I/O, default window/lookahead
        >>> x = np.array([0,0,0,0,0,0,0,0], dtype=int)
        >>> y = lz77.encode(x)
        >>> np.all(lz77.decode(y) == x)
        True
    """

    source_cardinality: int
    target_cardinality: int = 2
    window_size: int = 4096
    lookahead_size: int = 18

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
        T = self.target_cardinality
        S = self.source_cardinality
        self._D = max(1, ceil(log(self.window_size + 1, T)))  # distance 0..W
        self._Lw = max(1, ceil(log(self.lookahead_size + 1, T)))  # length 0..L
        self._M = max(1, ceil(log(S, T)))  # symbol 0..S-1

    def _find_longest_match(
        self,
        window: np.ndarray,
        lookahead: np.ndarray,
    ) -> tuple[int, int]:
        """
        Return (d, l) with the longest match of 'lookahead' in 'window'.
        d = distance back from current position to the start of match (>=1),
        l = matched length.
        If no match, return (0, 0).
        """
        if window.size == 0:
            return 0, 0

        max_l = 0
        best_d = 0

        # Naive search: try every starting pos in window
        # window[-d] is last element; distance d in 1..len(window)
        for start in range(window.size):
            # max match length at this start
            l = 0
            # Compare sequentially without overrunning bounds
            while (
                l < lookahead.size
                and start + l < window.size
                and window[start + l] == lookahead[l]
            ):
                l += 1
            if l > max_l:
                max_l = l
                best_d = window.size - start  # distance from current pos

                # Early exit if we matched the entire lookahead
                if max_l == lookahead.size:
                    break

        if max_l == 0:
            return 0, 0
        return best_d, max_l

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encode a sequence of source symbols using LZ77, emitting a base-T stream.

        Input:
            input: 1D array with elements in [0:S)

        Output:
            1D array of base-T symbols representing concatenated triples (d, l, c).
        """
        x = np.asarray(input, dtype=int)
        if x.ndim != 1:
            raise ValueError("'input' must be a 1D array")
        if x.size == 0:
            return np.array([], dtype=int)
        if np.any((x < 0) | (x >= self.source_cardinality)):
            raise ValueError("input symbols out of range")

        T = self.target_cardinality
        W = self.window_size
        L = self.lookahead_size
        D, Lw, M = self._D, self._Lw, self._M

        out: list[int] = []
        n = x.size
        i = 0

        pbar = tqdm(total=n, desc="Compressing LZ77", delay=2.5)
        while i < n:
            # Remaining symbols; if only one left, no following 'c' exists => emit (0,0,c)
            if i == n - 1:
                d, l = 0, 0
                c = int(x[i])
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

            d, l = self._find_longest_match(window, lookahead)

            if d == 0 or l == 0:
                # No match: (0,0,c) with c = x[i]
                d, l, c = 0, 0, int(x[i])
                step = 1
            else:
                # Match found; c is the symbol after the match
                c = int(x[i + l])
                step = l + 1  # consume l matched + 1 following char

            # Emit triple (d,l,c) in base T.
            out.extend(integer_to_symbols(d, base=T, width=D))
            out.extend(integer_to_symbols(l, base=T, width=Lw))
            out.extend(integer_to_symbols(c, base=T, width=M))

            i += step
            pbar.update(step)
        pbar.close()

        return np.array(out, dtype=int)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decode a base-T stream of concatenated triples (d, l, c) back to source symbols.

        Input:
            input: 1D array with elements in [0:T). Must be a valid output of `encode`.

        Output:
            1D array with elements in [0:S).
        """
        y = np.asarray(input, dtype=int)
        if y.ndim != 1:
            raise ValueError("'input' must be a 1D array")
        if y.size == 0:
            return np.array([], dtype=int)
        if np.any((y < 0) | (y >= self.target_cardinality)):
            raise ValueError("encoded symbols out of range")

        T = self.target_cardinality
        D, Lw, M = self._D, self._Lw, self._M

        out: list[int] = []

        i = 0
        total_triple_syms = D + Lw + M
        pbar = tqdm(total=y.size, desc="Decompressing LZ77", delay=2.5)

        while i + total_triple_syms <= y.size:
            d = symbols_to_integer(y[i : i + D], base=T)
            i += D
            l = symbols_to_integer(y[i : i + Lw], base=T)
            i += Lw
            c = symbols_to_integer(y[i : i + M], base=T)
            i += M
            pbar.update(total_triple_syms)

            # Reconstruct match
            if d == 0 or l == 0:
                # no match, just output c
                out.append(c)
            else:
                start = len(out) - d
                if start < 0:
                    raise ValueError("Invalid stream: distance exceeds produced output")
                # Copy l symbols (allow overlap, as in true LZ77)
                for k in range(l):
                    out.append(out[start + k])
                # Then append c
                out.append(c)

        pbar.close()
        return np.array(out, dtype=int)
