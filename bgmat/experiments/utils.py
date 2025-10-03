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


"""Utilities for running experiments."""

from typing import Callable, Optional, Sequence, Union, Tuple
import chex
import jax.numpy as jnp
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import jax
from functools import partial
Array = chex.Array
PRNGKey = Array
import pathlib 
import os

import shutil
def abs_f_einstein(n_particles, lambda_de_broglie,volume,
                   einstein_spring_constant,beta):
    """ Returns beta / N * F for the Einstein crystal following Wirnsberger et al. (SI)
    """
    n_particles = float(n_particles)
    einstein_spring_constant = float(einstein_spring_constant)
    
    second_term = 3./2. * (1 - 1. / n_particles) * np.log(beta * einstein_spring_constant * lambda_de_broglie**2. / np.pi)
    first_term = 1. / n_particles * np.log(n_particles * lambda_de_broglie**3. / volume)
    third_term = -3./(2.*n_particles)*np.log(n_particles)
    return first_term + second_term + third_term

def _num_particles(system: str) -> int:
    return int(system.split("_")[-1])

def _extract_lattice_and_number(system):
    parts = system.split("_")
    assert len(parts) == 3
    return parts[1], int(parts[2])

def sample_box_dims_from_density_and_ratios(
    densities: jnp.ndarray,
    ratio_params: dict,
    key: jax.random.PRNGKey,
    n_particles: int,
    repeats: jnp.array,
    get_box_fn
):
    """
    Vectorized box dimension sampling from density and uniform b/a, c/a ratios.

    Args:
        densities: (N,) array of densities
        ratio_params: dict with 'min_ba', 'max_ba', 'min_ca', 'max_ca'
        key: PRNG key
        n_particles: number of atoms in the box
        get_box_fn: function like utils.get_beta_tin_box_lengths

    Returns:
        box_dims: (N, 3) array of box lengths [a, b, c]
        ratios: dict of sampled r_ba and r_ca
    """

    N = densities.shape[0]
    key_ba, key_ca = jax.random.split(key)

    r_ba = jax.random.uniform(key_ba, shape=(N,), minval=ratio_params["min_ba"], maxval=ratio_params["max_ba"])
    r_ca = jax.random.uniform(key_ca, shape=(N,), minval=ratio_params["min_ca"], maxval=ratio_params["max_ca"])

    shape_factors = jnp.stack([jnp.ones_like(r_ba), r_ba, r_ca], axis=-1)
    def single_box(density, shape_factor):
        return get_box_fn(
            num_particles=n_particles,
            density=density,
            dim=3,
            shape_factor=shape_factor,  # convert to Python list
            repeats=repeats,
        )

    # Vectorize using vmap
    box_dims = jax.vmap(single_box)(densities, shape_factors)

    return box_dims, {"r_ba": r_ba, "r_ca": r_ca}

def copy_file(src_filename, target_dir, target_filename):
    """
    Copies a file to the target directory with a new name.

    Parameters:
    - src_filename: str - the source file path
    - target_dir: str - the destination directory
    - target_filename: str - the new name for the copied file
    """
    # Make sure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Full path for the new file
    target_path = os.path.join(target_dir, target_filename)

    # Copy the file
    shutil.copy2(src_filename, target_path)

    print(f"File copied to {target_path}")

Numeric = Union[Array, int, float]


def get_lr_schedule(base_lr: float,
                    lr_decay_steps: Sequence[int],
                    lr_decay_factor: float) -> Callable[[Numeric], Numeric]:
  """Returns a callable that defines the learning rate for a given step."""
  if not lr_decay_steps:
    return lambda _: base_lr

  lr_decay_steps = jnp.array(lr_decay_steps)
  if not jnp.all(lr_decay_steps[1:] > lr_decay_steps[:-1]):
    raise ValueError('Expected learning rate decay steps to be increasing, got '
                     f'{lr_decay_steps}.')

  def lr_schedule(update_step: Numeric) -> Array:
    i = jnp.sum(lr_decay_steps <= update_step)
    return base_lr * lr_decay_factor**i

  return lr_schedule
def get_orthorhombic_box_lengths_jax(
    num_particles: int,
    density: float,
    dim: int,
    shape_factor: jnp.ndarray,
    repeats: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Returns edge lengths of an orthorhombic box using JAX."""
    assert dim == shape_factor.shape[0]

    if repeats is None:
        repeats = jnp.ones(dim, dtype=int)

    vol = num_particles / density
    base = (vol / jnp.prod(shape_factor * repeats)) ** (1.0 / dim)
    return base * shape_factor * repeats

def get_beta_tin_box_lengths_jax(
    num_particles: int,
    density: float,
    dim: int,
    repeats: Optional[jnp.ndarray],
    shape_factor: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Returns edge lengths of an orthorhombic beta-tin box using JAX."""
    return get_orthorhombic_box_lengths_jax(
        num_particles=num_particles,
        density=density,
        dim=dim,
        shape_factor=shape_factor,
        repeats=repeats,
    )

def get_orthorhombic_box_lengths(
    num_particles: int, density: float, dim: int, shape_factor: Array,
    repeats: Optional[Array]) -> Array:
  """Returns edge lengths of an orthorhombic box."""
  assert dim == len(shape_factor)
  vol = num_particles / density
  if repeats is None:
    repeats = np.ones(dim, dtype=int)
  base = (vol / np.prod(shape_factor * repeats)) ** (1./dim)
  return base * shape_factor * repeats


def get_hexagonal_box_lengths(
    num_particles: int, density: float, dim: int,
    repeats: Optional[Array] = None) -> Array:
  """Returns edge lengths of an orthorhombic box for Ih packing."""
  shape_factor = np.array([1.0, np.sqrt(3), np.sqrt(8/3)])
  return get_orthorhombic_box_lengths(
      num_particles, density, dim, shape_factor, repeats)

def get_fcc_box_lengths(
    num_particles: int, density: float, dim: int,
    repeats: Optional[Array] = None) -> Array:
  """Returns edge lengths of an orthorhombic box for Ih packing."""
  shape_factor = np.array([1.0, np.sqrt(3), np.sqrt(6)])
  return get_orthorhombic_box_lengths(
      num_particles, density, dim, shape_factor, repeats)

def get_beta_tin_box_lengths(
    num_particles: int, density: float, dim: int,
    repeats: Optional[Array] = None) -> Array:
  """Returns edge lengths of an orthorhombic box for Ih packing."""
  shape_factor = np.array([1.0, 1.0, 0.52])
  return get_orthorhombic_box_lengths(
      num_particles, density, dim, shape_factor, repeats)

def get_cubic_box_lengths(
    num_particles: int, density: float, dim: int, as_jax:bool=False, repeats: Optional[Array]=None) -> Array:
  """Returns the edge lengths of a cubic simulation box."""
  edge_length = (num_particles / density) ** (1./dim)
  if as_jax:
    return jnp.full([dim], edge_length)
  else:
    return np.full([dim], edge_length)

@partial(jax.jit,static_argnums=(1,))
def reshape_key(keys, multi_gpu):
    return keys if multi_gpu  else keys[0]

@partial(jax.jit,static_argnums=(1,))
def select_one_device(pytree, multi_gpu=True):
    if multi_gpu:
      return jax.tree_util.tree_map(lambda x: x[0], pytree)
    else:
        return pytree

def concatenate_pmap_outputs(pmap_output):
    """Concatenate outputs from all pmapped replicas into a single batch."""
    from jax import tree_util, device_get
    import jax.numpy as jnp

    host_data = device_get(pmap_output)
    return tree_util.tree_map(lambda x: jnp.concatenate(x, axis=0), host_data)


def print_results(metrics, temp=None):
        df = metrics["beta_f_per_particle"]
        ess = metrics["ess"]
        dfm = metrics["beta_f_per_particle_marginal"]
        essm = metrics["ess_marginal"]
        if temp is not None:
          print(f'Temperature: {temp}\n')
            
        print(f'beta_f_per_particle: {df}\n'
            f'ess: {ess}\n'
            f'beta_f_per_particle_marginal: {dfm}\n'
            f'ess_marginal: {essm}')
        print("\n")
        
def plot_results(losses, efficiencies, deltaFs, config, abs_f_ref_per_particle=None):
        clear_output(wait=True)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Loss")
        plt.xlabel(f"Epoch")
        plt.plot(losses )
        if abs_f_ref_per_particle is not None:
            plt.axhline(abs_f_ref_per_particle  , linestyle="--", color="black")

        plt.subplot(1, 3, 2)
        plt.title(r"$\beta F / N$")
        plt.xlabel(f"Epoch / {config.test.test_every}")
        if abs_f_ref_per_particle is not None:
            plt.axhline(abs_f_ref_per_particle  , linestyle="--", color="black")
        plt.plot(deltaFs)

        plt.subplot(1, 3, 3)
        plt.plot(efficiencies)
        plt.title("Efficiencies in %")
        plt.xlabel(f"Epoch / {config.test.test_every}")
        # plt.ylim(0, 105)
        plt.tight_layout()
        plt.show()
def flatten_jax_dict(input_dict):
    return jax.tree.map(lambda x: jnp.concatenate(x, axis=0), input_dict)



def init_fn_single_devices(init_fn: Callable, init_key: chex.PRNGKey, optimizer) -> Tuple:
    """Initialise the state. `common_key` ensures that the same initialisation is used for params on all devices."""
    params = init_fn(init_key)
    opt_state = optimizer.init(params)
    return params, opt_state
