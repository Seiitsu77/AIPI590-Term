"""Conjugate-gradient half-step of implicit-feedback ALS, copied from Anime Compass.

Source: github.com/Seiitsu77/anime-compass, commit 81e5979,
backend/anime_agent/evaluation/collaborative_baselines.py, function `_solve_factors`
(MIT licence). The function body below is that code, unedited; only the leading
underscore of the name is dropped. It is copied rather than imported so that the
experimental model cannot change when that repository does, and so that nothing in this
experiment reaches into the production code base.

SHA-256 of the source file at the time of copying:
84cfad50207dd91d7b9d811498703120f47e7085befb52b0c55d03692a048074
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def solve_factors(
    factors: npt.NDArray[Any],
    fixed: npt.NDArray[Any],
    indptr: npt.NDArray[Any],
    indices: npt.NDArray[Any],
    regularization: float,
    alpha: float,
    cg_steps: int,
    data: npt.NDArray[Any] | None = None,
) -> None:
    """One ALS half-step, in place, using conjugate gradient.

    Solving each row exactly costs O(f^3) and is prohibitive at this scale, so
    this uses the conjugate-gradient formulation from Takács et al. (2011),
    which needs only matrix-vector products against the fixed factor matrix.

    ``data`` carries the per-entry preference intensity r_ui, giving confidence
    ``c_ui = 1 + alpha * r_ui``. When it is None every observed entry has
    r_ui = 1 and the arithmetic reduces exactly to the binary formulation, so
    the frozen binary reference is unaffected by this parameter existing.
    """
    gramian = fixed.T @ fixed + regularization * np.eye(fixed.shape[1], dtype=np.float32)

    for row in range(factors.shape[0]):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if start == stop:
            factors[row] = 0.0
            continue
        columns = indices[start:stop]
        liked = fixed[columns]
        weights = None if data is None else np.asarray(data[start:stop], dtype=np.float32)

        current = factors[row]
        # Residual of (YtY + Yu^T (C-I) Yu + reg I) x = Yu^T C p, with p = 1 and
        # C - I = alpha * r on observed entries.
        if weights is None:
            target = liked.sum(axis=0)
            weighted_product = liked.T @ (liked @ current)
        else:
            target = (liked * weights[:, None]).sum(axis=0)
            weighted_product = liked.T @ (weights * (liked @ current))
        residual = alpha * target - gramian @ current - alpha * weighted_product
        direction = residual.copy()
        residual_squared = float(residual @ residual)
        if residual_squared < 1e-10:
            continue

        for _step in range(cg_steps):
            if weights is None:
                curvature = liked.T @ (liked @ direction)
            else:
                curvature = liked.T @ (weights * (liked @ direction))
            product = gramian @ direction + alpha * curvature
            denominator = float(direction @ product)
            if abs(denominator) < 1e-12:
                break
            step_size = residual_squared / denominator
            current += step_size * direction
            residual -= step_size * product
            updated_squared = float(residual @ residual)
            if updated_squared < 1e-10:
                break
            direction = residual + (updated_squared / residual_squared) * direction
            residual_squared = updated_squared
