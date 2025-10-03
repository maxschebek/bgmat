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


"""Embeddings."""

import chex
import jax.numpy as jnp
import numpy as np
Array = chex.Array


def circular(x: Array,
             lower: float,
             upper: float,
             num_frequencies: int) -> Array:
  """Maps angles to points on the unit circle.

  The mapping is such that the interval [lower, upper] is mapped to a full
  circle starting and ending at (1, 0). For num_frequencies > 1, the mapping
  also includes higher frequencies which are multiples of 2 pi/(lower-upper)
  so that [lower, upper] wraps around the unit circle multiple times.

  Args:
    x: array of shape [..., D].
    lower: lower limit, angles equal to this will be mapped to (1, 0).
    upper: upper limit, angles equal to this will be mapped to (1, 0).
    num_frequencies: number of frequencies to consider in the embedding.

  Returns:
    An array of shape [..., 2*num_frequencies*D].
  """
  base_frequency = 2. * jnp.pi / (upper - lower)
  frequencies = base_frequency * jnp.arange(1, num_frequencies+1)
  angles = frequencies * (x[..., None] - lower)
  # Reshape from [..., D, num_frequencies] to [..., D*num_frequencies].
  angles = angles.reshape(x.shape[:-1] + (-1,))
  cos = jnp.cos(angles)
  sin = jnp.sin(angles)
  return jnp.concatenate([cos, sin], axis=-1)

def circular_nd(x: jnp.ndarray,
                lower: jnp.ndarray,
                upper: jnp.ndarray,
                num_frequencies: int) -> jnp.ndarray:
    """
    Multi-dimensional circular embedding with dimension-specific bounds.

    Args:
        x: array of shape [..., D]
        lower: array of shape [D], lower bounds for each dimension
        upper: array of shape [D], upper bounds for each dimension
        num_frequencies: number of frequency bands

    Returns:
        Embedded array of shape [..., 2 * num_frequencies * D]
    """
    base_frequency = 2. * jnp.pi / (upper - lower)  # shape: [D]
    frequencies = base_frequency * jnp.arange(1, num_frequencies + 1)[:, None]  # [num_freqs, D]
    deltas = x - lower  # [..., D]
    angles = frequencies.T[None, ...] * deltas[..., None]  # [..., D, num_freqs]
    angles = angles.reshape(x.shape[:-1] + (-1,))  # flatten freq dim
    return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)


def positional_encoding(x, lower, upper, num_frequencies, l0=100):
    """
    Generalized sinusoidal positional encoding for any scalar input.

    Args:
        x: scalar or array of shape (B,1) — input value(s)
        x_min, x_max: min and max values of x
        N: highest index n (embedding has dimension N+1)
        l0: frequency scaling hyperparameter

    Returns:
        Array of shape (..., N+1) — positional embeddings
    """
    x_mean = (upper + lower) / 2.0
    x_hat = (x - x_mean) / (upper - lower)  # normalized x

    n = jnp.arange(num_frequencies + 1)  # n = 0 to N
    freqs = jnp.where(n % 2 == 0,
                      1 + n // 2,
                      1 + (n - 1) // 2)
    angles = freqs * jnp.pi * x_hat / l0  # (..., N+1)

    embedding = jnp.where(n % 2 == 0,
                          jnp.cos(angles),
                          jnp.sin(angles))

    return embedding


class PositionalEncoding:
    def __init__(self, lower, upper, num_frequencies, l0=100):
        self.lower = lower
        self.upper = upper
        self.num_frequencies = num_frequencies
        self.l0 = l0

        # Determine once whether bounds are degenerate
        self.degenerate = np.isclose(lower, upper)

        # Precompute frequency pattern
        n = jnp.arange(num_frequencies + 1)
        self.n=n
        self.freqs = jnp.where(n % 2 == 0,
                               1 + n // 2,
                               1 + (n - 1) // 2)
        self.even_mask = (n % 2 == 0)

    def __call__(self, x):
        if self.degenerate:
            # Return ones with fixed shape
            return jnp.ones((x.shape[0], 1))

        # Standard positional encoding
        x_mean = (self.upper + self.lower) / 2.0
        x_hat = (x - x_mean) / (self.upper - self.lower)

        angles = self.freqs * jnp.pi * x_hat / self.l0  # shape (B, N+1)
        
        embedding = jnp.where(self.n % 2 == 0,
                          jnp.cos(angles),
                          jnp.sin(angles))
        return embedding