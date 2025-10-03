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


"""Monatomic (mW) water system."""

import math
from typing import Optional

import chex
from bgmat.systems import energies 
from bgmat.utils import observable_utils as obs_utils
import jax
import jax.numpy as jnp

Array = chex.Array

kJ_2_kcal = 0.2390057361

SW_A = 7.049556277
SW_B = 0.6022245584
SW_GAMMA = 1.2

MW_REDUCED_CUTOFF = 1.8
MW_COS = math.cos(109.47 / 180. * math.pi)

# Only quantities that differ for monatomic water / Silicon
MW_EPSILON = 6.189   # kcal/mol
MW_SIGMA = 2.3925   # Angstrom
MW_LAMBDA = 23.15

SI_EPSILON = 50.003  # kcal/mol
SI_SIGMA = 2.0951 # Angstrom
SI_LAMBDA = 21



class _TwoBodyEnergy(energies.PairwisePotentialEnergy):
  """Implements the two-body component of the monatomic-water energy."""
  
  def __init__(self,sigma, epsilon,
               min_distance, linearize_below):
    super().__init__(min_distance=min_distance, linearize_below=linearize_below)
    self._sigma = sigma
    self._epsilon = epsilon

  def _unclipped_pairwise_potential(self, r2: Array) -> Array:
    r2 /= self._sigma**2
    r = jnp.sqrt(r2)
    mask = jnp.array(r < MW_REDUCED_CUTOFF)
    # Distances on or above the cutoff can cause NaNs in the gradient of
    # `term_2` below, even though they're masked out in the forward computation.
    # To avoid this, we set these distances to a safe value.
    r = jnp.where(mask, r, 2. * MW_REDUCED_CUTOFF)
    term_1 = SW_A * self._epsilon * (SW_B / r2**2 - 1.)
    term_2 = jnp.where(mask, jnp.exp(1. / (r - MW_REDUCED_CUTOFF)), 0.)
    energy = term_1 * term_2
    return energy


class StillingerWeberEnergy(energies.PotentialEnergy):
  """Evaluates the monatomic water energy with periodic boundary conditions.

  The monatomic water model, or mW model, consists of point particles that
  interact with each other via two-body interactions (between pairs of
  particles) and three-body interactions (between triplets of particles).

  The energy is decomposed as follows:
  ```
      energy = sum of all two-body interactions over distinct pairs +
               sum of all three-body interactions over distinct triplets
  ```
  More details on the specific functional form of the individual interaction
  terms can be found in the paper of Molinero and Moore (2009):
  https://arxiv.org/abs/0809.2811.
  """

  def __init__(self,
               sigma: float ,
               epsilon: float,
               lambda_three_body: Optional[float] = None,
               box_length: Optional[Array] = None,
               min_distance: float = 0.,
               linearize_below: Optional[float] = None):
    """Constructor.

    Args:
      box_length: array of shape [dim], side lengths of the simulation box. If
        None, the box length must be passed as an argument to the class methods.
      min_distance: we clip the pairwise distance to this value in the
        calculation of the two-body term. This can be used to remove the
        singularity of the two-body term at zero distance.
      linearize_below: we linearize the two-body term below this value. If None,
        no linearization is done.
    """
    super().__init__(box_length)

    
    self._lambda_three_body = lambda_three_body    
    self._sigma = sigma    
    self._epsilon = epsilon    

    self._two_body_energy = _TwoBodyEnergy(sigma =sigma, epsilon = epsilon,
        min_distance=min_distance, linearize_below=linearize_below)

  def _three_body_energy(self, dr: Array, lambda_three_body: float = None) -> Array:
    """Compute three-body term for one sample.

    Args:
      dr: [num_particles, num_particles, 3] array of distance vectors
        between the particles.
    Returns:
      The three-body energy contribution of the sample (a scalar).
    """
    if lambda_three_body==None:
      lambda_3b = self._lambda_three_body
    else:
      lambda_3b = lambda_three_body
    def _one_particle_contribution(dri: Array) -> Array:
      # dri is (num_particles-1) x 3.
      raw_norms = jnp.linalg.norm(dri, axis=-1)
      keep = raw_norms < MW_REDUCED_CUTOFF
      norms = jnp.where(keep, raw_norms, 1e20)
      norm_energy = jnp.exp(SW_GAMMA/(norms - MW_REDUCED_CUTOFF))
      norm_energy = jnp.where(keep, norm_energy, 0.)
      normprods = norms[None, :] * norms[:, None]
      # Note: the sum below is equivalent to:
      # dotprods = jnp.dot(dri, dri[..., None]).squeeze(-1)
      # but using jnp.dot results in loss of precision on TPU,
      # as evaluated by comparing to MD samples.
      dotprods = jnp.sum(dri[:, None, :] * dri[None, :, :], axis=-1)

      cos_ijk = dotprods / normprods

      energy = lambda_3b * self._epsilon * (MW_COS - cos_ijk)**2
      energy *= norm_energy
      energy = jnp.triu(energy, 1)
      energy = jnp.sum(energy, axis=-1)
      return jnp.dot(energy, norm_energy)

    # Remove diagonal elements [i, i, :], changing the shape from
    # [num_particles, num_particles, 3] to [num_particles, num_particles-1, 3].
    clean_dr = jnp.rollaxis(jnp.triu(jnp.rollaxis(dr, -1), 1)[..., 1:]+
                            jnp.tril(jnp.rollaxis(dr, -1), -1)[..., :-1],
                            0, dr.ndim)
    # Vectorize over particles.
    energy = jnp.sum(jax.vmap(_one_particle_contribution)(clean_dr))
    return energy

  def energy(self,
             coordinates: Array,
             box_length: Optional[Array] = None) -> Array:
    """Computes energies for an entire batch of particles.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim], side lengths of the simulation
        box. If None, the default box length will be used instead.

    Returns:
      energy: array with shape [...] containing the computed energies.
    """
    if box_length is None:
      box_length = self.box_length
    dr = obs_utils.pairwise_difference_pbc(coordinates, box_length)
    dr /= self._sigma
    two_body_energy = self._two_body_energy(coordinates, box_length)
    # Vectorize over samples.

    # three_body_energy = jnp.vectorize(self._three_body_energy,
                                      # signature='(m,m,n)->()')(dr)

    three_body_energy = jax.vmap(self._three_body_energy)(dr)

    return two_body_energy + three_body_energy

  def energy_with_lambda(self,
             coordinates: Array,
             lambdas_three_body: Array,
             box_length: Optional[Array] = None) -> Array:
    """Computes energies for an entire batch of particles.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim], side lengths of the simulation
        box. If None, the default box length will be used instead.

    Returns:
      energy: array with shape [...] containing the computed energies.
    """
    if box_length is None:
      box_length = self.box_length
    dr = obs_utils.pairwise_difference_pbc(coordinates, box_length)
    dr /= self._sigma
    two_body_energy = self._two_body_energy(coordinates, box_length)
    # Vectorize over samples.

    # three_body_energy = jnp.vectorize(self._three_body_energy,
                                      # signature='(m,m,n)->()')(dr)

    three_body_energy = jax.vmap(self._three_body_energy)(dr, lambdas_three_body)
    return two_body_energy + three_body_energy
class MonatomicWaterEnergySW(StillingerWeberEnergy):
  """Evaluates the monatomic water energy with periodic boundary conditions.

  The monatomic water model, or mW model, consists of point particles that
  interact with each other via two-body interactions (between pairs of
  particles) and three-body interactions (between triplets of particles).

  The energy is decomposed as follows:
  ```
      energy = sum of all two-body interactions over distinct pairs +
               sum of all three-body interactions over distinct triplets
  ```
  More details on the specific functional form of the individual interaction
  terms can be found in the paper of Molinero and Moore (2009):
  https://arxiv.org/abs/0809.2811.
  """

  def __init__(self,
               box_length: Optional[Array] = None,
               min_distance: float = 0.,
               linearize_below: Optional[float] = None):
    """Constructor.

    Args:
      box_length: array of shape [dim], side lengths of the simulation box. If
        None, the box length must be passed as an argument to the class methods.
      min_distance: we clip the pairwise distance to this value in the
        calculation of the two-body term. This can be used to remove the
        singularity of the two-body term at zero distance.
      linearize_below: we linearize the two-body term below this value. If None,
        no linearization is done.
    """
    super().__init__(lambda_three_body=MW_LAMBDA,
                     sigma=MW_SIGMA,
                     epsilon=MW_EPSILON,
                     box_length=box_length,
                     min_distance=min_distance,
                     linearize_below=linearize_below)

class SiliconEnergySW(StillingerWeberEnergy):
  """Evaluates the monatomic water energy with periodic boundary conditions.

  The monatomic water model, or mW model, consists of point particles that
  interact with each other via two-body interactions (between pairs of
  particles) and three-body interactions (between triplets of particles).

  The energy is decomposed as follows:
  ```
      energy = sum of all two-body interactions over distinct pairs +
               sum of all three-body interactions over distinct triplets
  ```
  More details on the specific functional form of the individual interaction
  terms can be found in the paper of Molinero and Moore (2009):
  https://arxiv.org/abs/0809.2811.
  """

  def __init__(self,
               box_length: Optional[Array] = None,
               min_distance: float = 0.,
               linearize_below: Optional[float] = None):
    """Constructor.

    Args:
      box_length: array of shape [dim], side lengths of the simulation box. If
        None, the box length must be passed as an argument to the class methods.
      min_distance: we clip the pairwise distance to this value in the
        calculation of the two-body term. This can be used to remove the
        singularity of the two-body term at zero distance.
      linearize_below: we linearize the two-body term below this value. If None,
        no linearization is done.
    """
    super().__init__(lambda_three_body=SI_LAMBDA,
                     sigma=SI_SIGMA,
                     epsilon=SI_EPSILON,
                     box_length=box_length,
                     min_distance=min_distance,
                     linearize_below=linearize_below)