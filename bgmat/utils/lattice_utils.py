#!/usr/bin/python
#
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright 2025 Maximilian Schebek, Freie Universität Berlin
# Modified: 2025-10-03 - Adapted and extended for bgmat project. Further modifications should be documented here.


"""Utilities to build lattice-based particle models."""

from typing import Tuple, Union
import chex
import numpy as np
import bgmat.utils.observable_utils as obs_utils
Array = chex.Array


def gcd(values: Array, tol: float = 3e-2) -> Union[int, float]:
  """GCD of a list of numbers (possibly floats)."""
  def _gcd2(a, b):
    if np.abs(b) < tol:
      return a
    else:
      return _gcd2(b, a % b)
  x = values[0]
  for v in values[1:]:
    x = _gcd2(x, v)
  return x


def make_simple_lattice(lower: Array,
                        upper: Array,
                        cell_aspect: Array,
                        n: int) -> Tuple[Array, Array, Array]:
  """Returns a shifted cubic lattice with atoms at the unit cell centres."""
  dim = len(lower)
  assert len(upper) == dim
  assert len(cell_aspect) == dim
  box_size = upper - lower
  normalized_box = (upper - lower) / cell_aspect
  integer_aspect = np.round(normalized_box / gcd(normalized_box)).astype(int)
  num_per_dim = np.round((n/np.prod(integer_aspect)) ** (1/dim)).astype(int)
  repeats = num_per_dim * integer_aspect
  if np.prod(repeats) != n:
    raise ValueError(f'The number of lattice points {n} does not '
                     f'match the box size {box_size} and cell aspect '
                     f'{cell_aspect}, got integer aspect {integer_aspect}, '
                     f'{repeats} repeats.')
  points = [np.linspace(lower[i], upper[i], repeats[i], endpoint=False).T
            for i in range(dim)]
  xs = np.meshgrid(*points)
  lattice = np.concatenate([x[..., None] for x in xs], axis=-1)
  lattice = lattice.reshape(np.prod(repeats), dim)
  lattice_constant = (upper - lower) / repeats

  return lattice, lattice_constant, repeats


def make_lattice(lower: Array,
                 upper: Array,
                 cell_aspect: Array,
                 atom_positions_in_cell: Array,
                 n: int) -> Tuple[Array, Array, Array]:
  """An orthorhombic lattice of repeated unit cells.

  Args:
    lower: vector of lower limits of lattice box (number of elements
      determines the dimensionality)
    upper: vector of upper limits of lattice box (number of elements
      must be the same as `lower`)
    cell_aspect: relative lengths of unit cell edges. A cubic cell would have
      `cell_aspect==[1, 1, 1]` for example. The box basis `lower - upper`,
      divided by `cell_aspect`, should have low-integer length ratios.
    atom_positions_in_cell: a n_u x dimensionality matrix with the fractional
      positions of atoms within each unit cell. `n_u` will be the number
      of atoms per unit cell.
    n: number of atoms in lattice. Should be the product of the low-integer
      length ratios of the aspect-normalized box, times some integer to the
      power of the number of dimensions, times the number of atoms per
      cell.

  Returns:
    A 3-tuple (lattice, lattice_constant, repeats):
      lattice: n x dimension array of lattice sites.
      lattice_constant: a vector of length equal to the dimensionality, with
        the side lengths of the unit cell.
      repeats: an integer vector of length equal to the dimensionality, with
        the number of cells in each dimension. `repeats x lattice_constant`
        equals `upper - lower`.
  """
  num_cells = n // len(atom_positions_in_cell)
  if num_cells * len(atom_positions_in_cell) != n:
    raise ValueError(f'Number of particles {n} is not divisible by the number '
                     f'of particles per cell {len(atom_positions_in_cell)}')
  base, lc, repeats = make_simple_lattice(lower, upper, cell_aspect, num_cells)
  sites = atom_positions_in_cell * lc
  lattice = base[..., None, :] + sites
  lattice = lattice.reshape(-1, lattice.shape[-1])
  return lattice, lc, repeats

import numpy as np

def compute_neighbor_shells(positions, box, atom_index=0, tolerance=0.001, max_shell=None):
    """
    Compute neighbor shells around a given atom in a periodic system.
    
    Args:
        positions (np.ndarray): (N, 3) array of atomic positions.
        box (np.ndarray): (3, 3) periodic cell matrix.
        atom_index (int): Index of the reference atom.
        tolerance (float): Width of distance bins for shells.
    
    Returns:
        dict: Mapping from shell distance to number of atoms in that shell.
    """
    central_pos = positions[atom_index]
    displacements = positions - central_pos  # (N, 3)

    # Remove self
    mask = np.arange(len(positions)) != atom_index
    displacements = displacements[mask]
    others = positions[mask]

    # Vectorized PBC distances
    diffs = central_pos - others
    distances = np.sqrt(obs_utils.squared_distance_pbc(diffs, box))  # (N-1,)
    # Bin distances into shells 
    
    # Find unique distances (and their indices) within the tolerance
    unique_distances, shell_ids = np.unique(np.round(distances / tolerance) * tolerance, return_inverse=True)

    # Count occurrences of each unique distance (shell size)
    counts = np.bincount(shell_ids)
    
    # Compute cumulative count
    cumulative_counts = np.cumsum(counts)

    # Apply max_shell limit
    if max_shell is not None:
        unique_distances = unique_distances[:max_shell]
        counts = counts[:max_shell]
        cumulative_counts = cumulative_counts[:max_shell]

    return {
        round(d, 8): (count, cum)
        for d, count, cum in zip(unique_distances, counts, cumulative_counts)
    }

def cumulative_neighbors_up_to_shell(positions, box,  shell_number, atom_index=0, tolerance=0.05):
    """
    Return the total number of atoms within the first `shell_number` neighbor shells.

    Args:
        positions (np.ndarray): Atomic positions (N, 3).
        box (np.ndarray): Periodic cell (3, 3).
        atom_index (int): Index of the reference atom.
        shell_number (int): Number of shells to include.
        tolerance (float): Distance tolerance for shell grouping.

    Returns:
        int: Total number of atoms in the first `shell_number` shells.
    """
    shells = compute_neighbor_shells(
        positions, box, atom_index,
        tolerance=tolerance,
        max_shell=shell_number
    )

    if not shells:
        return 0

    # Get last shell's cumulative count
    return list(shells.values())[-1][1]