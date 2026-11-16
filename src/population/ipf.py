"""Iterative proportional fitting (IPF) for an N-dimensional contingency table.

We know several low-dimensional tables from the census (for example
age x sex x income, and age x sex x work) but not the full joint distribution
of all attributes together. IPF starts from a seed table and repeatedly rescales
it so that it matches each known table in turn. When it converges, the result
agrees with every input table at once.

A table may use broader categories than the joint table (for example a
15-24 age band where the joint table has 15-19 and 20-24). `Constraint.groups`
handles that: the fit is made at the broad level and the split inside each
broad band is left to the other tables.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Constraint:
    """A known marginal table.

    dims:   names of the attributes the table covers, e.g. ("age", "sex")
    target: array of proportions with one axis per name in `dims`, summing to 1
    groups: optional {dim: [broad index for each fine category]} for dims where
            the table is broader than the joint table
    labels: optional {dim: [broad category names]}, used only for reporting
    """
    dims: tuple
    target: np.ndarray
    groups: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)


@dataclass
class IPFResult:
    table: np.ndarray      # fitted joint distribution, sums to 1
    iterations: int
    max_error: float       # largest absolute gap to any constraint cell
    converged: bool


def _marginal(table: np.ndarray, axes: tuple) -> np.ndarray:
    """Sum the table over every axis except `axes`, keeping `axes` in the given order."""
    other = tuple(i for i in range(table.ndim) if i not in axes)
    summed = table.sum(axis=other)
    # after summing, remaining axes are in ascending order; reorder to match `axes`
    order = np.argsort(np.argsort(axes))
    return np.transpose(summed, order)


def _expand(values: np.ndarray, axes: tuple, ndim: int) -> np.ndarray:
    """Reshape a marginal so it broadcasts against the full table."""
    ascending = np.argsort(axes)
    values = np.transpose(values, ascending)
    shape = [1] * ndim
    for size, axis in zip(values.shape, sorted(axes)):
        shape[axis] = size
    return values.reshape(shape)


def coarsen(values: np.ndarray, maps: list) -> np.ndarray:
    """Add up fine categories into broad ones. maps[i] is None or the list of
    broad indices for axis i."""
    for axis, m in enumerate(maps):
        if m is None:
            continue
        m = np.asarray(m)
        out_shape = list(values.shape)
        out_shape[axis] = int(m.max()) + 1
        out = np.zeros(out_shape)
        np.add.at(out, tuple(m if a == axis else slice(None) for a in range(values.ndim)), values)
        values = out
    return values


def _refine(values: np.ndarray, maps: list) -> np.ndarray:
    """Copy each broad value to every fine category inside it."""
    for axis, m in enumerate(maps):
        if m is not None:
            values = np.take(values, np.asarray(m), axis=axis)
    return values


def ipf(dim_names, dim_sizes, constraints, seed=None, tol=1e-6, max_iter=500) -> IPFResult:
    """Fit a joint distribution over `dim_names` that matches every constraint.

    seed: optional starting table. Cells that are zero in the seed stay zero,
          which is how impossible combinations are ruled out.
    """
    table = np.ones(dim_sizes, dtype=float) if seed is None else seed.astype(float).copy()
    table /= table.sum()

    prepared = []
    for c in constraints:
        axes = tuple(dim_names.index(d) for d in c.dims)
        maps = [c.groups.get(d) for d in c.dims]
        target = np.asarray(c.target, dtype=float)
        expected = tuple(dim_sizes[a] if m is None else max(m) + 1 for a, m in zip(axes, maps))
        for a, m in zip(axes, maps):
            if m is not None and len(m) != dim_sizes[a]:
                raise ValueError(f"Constraint {c.dims}: group map has wrong length")
        if target.shape != expected:
            raise ValueError(f"Constraint {c.dims} has shape {target.shape}, expected {expected}")
        prepared.append((axes, maps, target / target.sum()))

    def gap():
        return max(np.abs(coarsen(_marginal(table, axes), maps) - target).max()
                   for axes, maps, target in prepared)

    max_error = np.inf
    for iteration in range(1, max_iter + 1):
        for axes, maps, target in prepared:
            current = coarsen(_marginal(table, axes), maps)
            ratio = np.divide(target, current, out=np.zeros_like(target), where=current > 0)
            table = table * _expand(_refine(ratio, maps), axes, table.ndim)
        max_error = gap()
        if max_error < tol:
            return IPFResult(table / table.sum(), iteration, float(max_error), True)
    return IPFResult(table / table.sum(), max_iter, float(max_error), False)
