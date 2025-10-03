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
#
# Modifications copyright 2025 Maximilian Schebek, Freie Universität Berlin
# Modified: 2025-10-03 - Adapted and extended for bgmat project. Further modifications should be documented here.

import e3nn_jax as e3nn

import math
from typing import Optional
import functools
from bgmat.models import embeddings
from bgmat.utils.observable_utils import (
    difference_pbc,
    pairwise_difference_pbc,
)
from bgmat.models.center_of_mass import center_of_mass

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from typing import Tuple, Callable

Array = chex.Array


def get_senders_and_receivers_fully_connected(
    n_nodes: int,
) -> Tuple[chex.Array, chex.Array]:
    """
    Generate sender and receiver indices for a fully connected graph (excluding self-connections).
    Args:
        n_nodes: Number of nodes in the graph.
    Returns:
        Tuple of (senders, receivers) as arrays of indices.
    """
    receivers = []
    senders = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            receivers.append(i)
            senders.append((i + 1 + j) % n_nodes)
    return jnp.array(senders), jnp.array(receivers)

def _layer_norm(x: Array, name: Optional[str] = None) -> Array:
  """Apply a unique LayerNorm to `x` with default settings."""
  return hk.LayerNorm(axis=-1,
                      create_scale=True,
                      create_offset=True,
                      name=name)(x)

def get_top_n_connections_vectorized(
    positions: jnp.ndarray, n: int, box
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    For each node, find the indices of its n nearest neighbors (excluding itself) using periodic boundary conditions.
    Args:
        positions: Array of shape (n_nodes, dim), node coordinates.
        n: Number of nearest neighbors to connect to each node.
        box: Simulation box size or shape.
    Returns:
        senders: Indices of sending nodes (flattened).
        receivers: Indices of receiving nodes (flattened).
    """
    n_nodes = positions.shape[0]
    diffs = pairwise_difference_pbc(positions, box)
    dist2 = jnp.sum(diffs**2, axis=-1)

    # Exclude self-distances
    mask = jnp.eye(n_nodes, dtype=bool)
    dist2 = jnp.where(mask, jnp.inf, dist2)

    # Get n nearest neighbors for each node
    nearest_indices = jnp.argsort(dist2, axis=1)[:, :n]  # shape: (n_nodes, n)

    # Each node i receives messages from its n nearest neighbors
    receivers = jnp.repeat(jnp.arange(n_nodes), n)  # shape: (n_nodes * n,)
    senders = nearest_indices.reshape(-1)  # flatten to (n_nodes * n,)

    return senders, receivers


class _DenseBlock(hk.Module):
    """
    An MLP with one hidden layer, whose output has the size of the input divided by a reducing factor.
    Used as a building block for node and edge feature transformations in GNNs.
    """

    def __init__(
        self,
        w_init: hk.initializers.Initializer,
        w_init_final: hk.initializers.Initializer,
        widening_factor: int = 1,
        reducing_factor: int = 1,
        name: Optional[str] = None,
    ):
        """
        Initialize a _DenseBlock module.
        Args:
            w_init: Initializer for the first linear layer.
            w_init_final: Initializer for the final linear layer.
            widening_factor: Factor to increase hidden layer size.
            reducing_factor: Factor to reduce output size.
            name: Optional name for the module.
        """
        super().__init__(name=name)
        self._widening_factor = widening_factor
        self._reducing_factor = reducing_factor
        self._w_init = w_init
        self._w_init_final = w_init_final

    def __call__(self, x: Array) -> Array:
        """
        Forward pass for the dense block: applies a linear layer, GELU activation, and a final linear layer.
        Args:
            x: Input array of shape (..., features).
        Returns:
            Output array with reduced feature dimension.
        """
        num_dims = x.shape[-1]
        num_hiddens = self._widening_factor * num_dims
        x = hk.Linear(num_hiddens, w_init=self._w_init)(x)
        x = jax.nn.gelu(x)
        num_out = num_dims // self._reducing_factor
        return hk.Linear(num_out, w_init=self._w_init_final)(x)


def _layer_norm(x: Array, name: Optional[str] = None) -> Array:
    """
    Apply a unique LayerNorm to `x` with default settings.
    Args:
        x: Input array.
        name: Optional name for the layer.
    Returns:
        Normalized array.
    """
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class GNNBlock(hk.Module):
    """
    GNN block.
    Performs message aggregation and feature update for nodes using edge and node features.
    """

    def __init__(
        self,
        n_local: int,
        w_init: Optional[hk.initializers.Initializer] = None,
        w_init_final: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a GNNBlock module.
        Args:
            n_local: Number of local neighbors for message passing.
            w_init: Initializer for the first linear layer (optional).
            w_init_final: Initializer for the final linear layer (optional).
            name: Optional name for the module.
        """
        super().__init__(name=name)
        default = hk.initializers.VarianceScaling(1.0)
        self._w_init = default if w_init is None else w_init
        self._w_init_final = default if w_init_final is None else w_init_final
        self._n_local = n_local

    def __call__(
        self,
        node_features: Array,
        edge_features: Array,
        included_positions,
        senders: Array,
        receivers: Array,
    ) -> Array:
        """
        Forward pass for a GNN block with message passing.
        Args:
            node_features: Array of node features, shape (batch, n_nodes, features).
            edge_features: Array of edge features, shape (batch, n_edges, features).
            included_positions: String, how to include positions in edge features.
            senders: Indices of sending nodes.
            receivers: Indices of receiving nodes.
        Returns:
            Updated node features after message passing.
        """

        n_nodes = node_features.shape[-2]
        # print('shape Nodes:',n_nodes)
        avg_num_neighbours = self._n_local
        # Create batch indices to align with idx
        batch_indices = jnp.arange(node_features.shape[0])[:, None]  # shape (128, 1)

        node_features_send = node_features[batch_indices, senders]
        node_features_receive = node_features[batch_indices, receivers]

        if included_positions == "self_and_neighbor":
            edge_feat_in = jnp.concatenate(
                [node_features_send, node_features_receive, edge_features], axis=-1
            )
            reducing_factor = 3
        elif included_positions == "self":
            edge_feat_in = jnp.concatenate(
                [node_features_receive, edge_features], axis=-1
            )
            reducing_factor = 2
        elif included_positions == "none":
            edge_feat_in = edge_features
            reducing_factor = 1
        else:
            raise ValueError("Please specify positions treatment.")

        m_ij = _DenseBlock(
            w_init=self._w_init,
            w_init_final=self._w_init_final,
            reducing_factor=reducing_factor,
            name="mlp_edge",
        )(edge_feat_in)

        m_i = jax.vmap(
            lambda p, r: e3nn.scatter_sum(data=p, dst=r, output_size=n_nodes)
        )(m_ij, receivers) / jnp.sqrt(avg_num_neighbours)

        phi_h_in = jnp.concatenate([m_i, node_features], axis=-1)

        features_out = _DenseBlock(
            widening_factor=1,
            w_init=self._w_init,
            w_init_final=self._w_init_final,
            reducing_factor=2,
            name=None,
        )(phi_h_in)
        # print('shape feat out:',features_out.shape)
        return features_out


def wrap(x: Array, lower: Array, upper: Array) -> Array:
    """
    Wraps `x` back into the box defined by lower and upper bounds (periodic boundary conditions).
    Args:
        x: Input coordinates.
        lower: Lower bound of the box.
        upper: Upper bound of the box.
    Returns:
        Wrapped coordinates within the box.
    """
    width = upper - lower
    return jnp.mod(x - lower, width) + lower


class GNN(hk.Module):
    """
    Graph Neural Network (GNN) model for atomistic systems.
    Supports positional encoding, message passing, and optional context conditioning.
    """

    def __init__(
        self,
        num_layers: int,
        embedding_size: int,
        lower_encode_diffs: float,
        upper_encode_diffs: float,
        lower_encode_pos: float,
        upper_encode_pos: float,
        num_frequencies: int,
        encode_diffs: bool,
        included_positions: str,
        use_interaction: bool = True,
        recompute_neighbors: bool = False,
        upper: Array = None,
        lower: Array = None,
        n_local: int = None,
        dropout_rate: float = 0.0,
        use_layernorm: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        w_init_final: Optional[hk.initializers.Initializer] = None,
        lattice: Array = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a GNN module.
        Args:
            num_layers: Number of GNN layers.
            embedding_size: Size of node embeddings.
            lower_encode_diffs: Lower bound for difference encoding.
            upper_encode_diffs: Upper bound for difference encoding.
            lower_encode_pos: Lower bound for position encoding.
            upper_encode_pos: Upper bound for position encoding.
            num_frequencies: Number of frequencies for encoding.
            encode_diffs: Whether to encode differences.
            included_positions: How to include positions in edge features.
            use_interaction: Whether to use interaction layers.
            recompute_neighbors: Whether to recompute neighbors dynamically.
            upper: Upper bound of the box.
            lower: Lower bound of the box.
            n_local: Number of local neighbors.
            dropout_rate: Dropout rate.
            use_layernorm: Whether to use layer normalization.
            w_init: Initializer for the first linear layer (optional).
            w_init_final: Initializer for the final linear layer (optional).
            lattice: Lattice positions for encoding.
            name: Optional name for the module.
        """
        super().__init__(name=name)
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate

        self._encode_diffs = encode_diffs
        self._included_positions = included_positions
        self._lattice = lattice
        self._recompute_neighbors = recompute_neighbors
        self._use_interaction = use_interaction
        self._lower = lower
        self._upper = upper
        n_nodes = lattice.shape[0]
        if n_local is None:
            n_local = n_nodes - 1
        self._n_local = n_local

        box_length = upper - lower
        if box_length.ndim < 3:
            box_length *= jnp.ones(3)
        self._box_length = box_length
        self._senders, self._receivers = get_top_n_connections_vectorized(
            lattice, n_local, box_length
        )

        if lower is not None:
            self._width = upper - lower
        else:
            self._width = None

        self._encoding_fn_diffs = functools.partial(
            embeddings.circular,
            lower=lower_encode_diffs,
            upper=upper_encode_diffs,
            num_frequencies=num_frequencies,
        )

        self._encoding_fn_pos = functools.partial(
            embeddings.circular_nd,
            lower=lower_encode_pos,
            upper=upper_encode_pos,
            num_frequencies=num_frequencies,
        )

        if use_layernorm:
            self._maybe_layer_norm = _layer_norm
        else:
            self._maybe_layer_norm = lambda h, name: h
        default = hk.initializers.VarianceScaling(2.0 / math.sqrt(num_layers))
        self._w_init = default if w_init is None else w_init
        self._w_init_final = default if w_init_final is None else w_init_final

    def __call__(
        self,
        x: Array,
        context: Array = None,
        lattice: Array = None,
        box_length: Array = None,
        is_training: Optional[bool] = None,
    ) -> Array:
        """
        Applies the GNN model to input positions and optional context.
        Args:
            x: Positions in cartesian coordinates of shape (batch, num_points, 3).
            context: Optional conditional variable (e.g., temperature), already encoded if desired.
            lattice: Optional lattice positions for encoding (default: self._lattice).
            box_length: Optional box size for periodic boundary conditions (default: self._box_length).
            is_training: Whether in training mode (affects dropout).
        Returns:
            Array of shape (batch, num_points, embedding_size): output node features.
        """
        n_batch, n_nodes, _ = x.shape
        if self._dropout_rate != 0.0 and is_training is None:
            raise ValueError("`is_training` must be specified when dropout is used.")
        dropout_rate = self._dropout_rate if is_training else 0.0

        if lattice is None:
            lattice = jnp.tile(self._lattice, (n_batch, 1, 1))
        if box_length is None:
            box_length = jnp.tile(self._box_length, (n_batch, 1, 1))

        if box_length.shape[1:] == (3, 3):
            # triclinic case
            batched_metric = jax.vmap(difference_pbc_triclinic)
        else:
            # isotropic case
            batched_metric = jax.vmap(difference_pbc)

        x_abs = jax.vmap(wrap)(x + lattice, -box_length / 2, box_length / 2)

        # Indexing helper: select node vectors by index
        def index_nodes(x, idx):
            return x[idx]  # x: [num_nodes, node_dim], idx: [num_edges]

        batched_index_nodes = jax.vmap(index_nodes, in_axes=(0, 0))

        if self._recompute_neighbors == True:
            senders, receivers = jax.vmap(
                lambda p, l: get_top_n_connections_vectorized(
                    positions=p, n=self._n_local, box=l
                )
            )(x_abs, box_length.reshape(n_batch, 3))

        else:
            senders = jnp.tile(self._senders, (n_batch, 1))
            receivers = jnp.tile(self._receivers, (n_batch, 1))

        batched_index_nodes = jax.vmap(index_nodes, in_axes=(0, 0))

        # Get vectors for senders and receivers
        sender_vecs = batched_index_nodes(x_abs, senders)
        receiver_vecs = batched_index_nodes(x_abs, receivers)
        # vectors = self._metric(receiver_vecs, sender_vecs)
        vectors = batched_metric(receiver_vecs, sender_vecs, box_length)

        h1 = self._encoding_fn_diffs(x)

        # always encode w.r.t. the equilibrium lattice
        h2 = self._encoding_fn_pos(jnp.tile(self._lattice, (n_batch, 1, 1)))
        h = jnp.concatenate((h1, h2), axis=-1)

        if context is not None:
            # Reshape context to make its shape compatible with the positional embeddings
            # Assumes that the context is of shape (n_batch, 1, dim_encode)
            context = jnp.tile(context, (1, n_nodes, 1))
            h = jnp.concatenate((h, context), axis=-1)

        if self._encode_diffs:
            vectors = self._encoding_fn_diffs(vectors)

        h = hk.Linear(self._embedding_size)(h)
        edge_features = hk.Linear(self._embedding_size)(vectors)

        for i in range(self._num_layers):
            h_norm = self._maybe_layer_norm(h, name=f"h{i}_ln_1")

            if self._use_interaction:
                h_update = GNNBlock(
                    w_init=self._w_init,
                    w_init_final=self._w_init_final,
                    n_local=self._n_local,
                    name=f"h{i}_gnn",
                )(
                    included_positions=self._included_positions,
                    node_features=h_norm,
                    edge_features=edge_features,
                    senders=senders,
                    receivers=receivers,
                )
            else:
                h_update = h_norm

            h = h + h_update
            h_norm = self._maybe_layer_norm(h, name=f"h{i}_ln_2")
            h_dense = _DenseBlock(
                w_init=self._w_init,
                w_init_final=self._w_init_final,
                name=f"h{i}_mlp",
            )(h_norm)
            # h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense

        h_norm = self._maybe_layer_norm(h, name="ln_f")
        h_out = h_norm
        return h_out
