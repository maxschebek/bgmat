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


"""Config file for monatomic water in the cubic or hexagonal ice phases."""

from bgmat.experiments import utils
from bgmat.models.gnn_conditioner import GNN
from bgmat.models import augmented_coupling_flows
from bgmat.models import particle_models
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

# Density, temperature and system shapes and sizes below chosen for comparison
# with the paper by Quigley (https://doi.org/10.1063/1.4896376, see Table 1).
from bgmat.systems.lennard_jones import LennardJonesEnergy

LATTICE_MODELS = {
    "fcc": particle_models.AugmentedFaceCentredCubicLattice,
    "hcp": particle_models.AugmentedHexagonalClosePackedLattice,
}
BOX_FUNS = {
    "hcp": utils.get_hexagonal_box_lengths,
    "fcc": utils.get_cubic_box_lengths,
}

NUM_ENCODE = {
    "fcc": 4,
    "hcp": 4,
}
DENSITY = 1.28


def get_config(num_particles: int, lattice: str) -> config_dict.ConfigDict:
    """Returns the config."""
    num_particles_encode = NUM_ENCODE[lattice]
    box_fun = BOX_FUNS[lattice]
    lattice_model = LATTICE_MODELS[lattice]
    box_lengths = box_fun(num_particles, density=DENSITY, dim=3)
    box_lengths_encode = box_fun(num_particles_encode, density=DENSITY, dim=3)
    box_lengths_108 = box_fun(108, density=DENSITY, dim=3)

    num_frequencies = 8

    remove_com = True
    use_com_shift = True
    recompute_neighbors = True

    config = config_dict.ConfigDict()
    config.state = dict(
        num_particles=num_particles,
        beta=0.5,
    )

    def create_base():
        return lattice_model(
            num_particles=num_particles,
            lower=-box_lengths / 2.0,
            upper=box_lengths / 2.0,
            noise_scale=0.05,
            noise_scale_aux=0.05,
            remove_com=remove_com,
        )

    base = create_base()

    conditioner = dict(
        constructor=augmented_coupling_flows.make_equivariant_conditioner,
        kwargs=dict(
            conditioner_constructor=GNN,
            conditioner_kwargs=dict(
                embedding_size=96,
                encode_diffs=True,
                recompute_neighbors=recompute_neighbors,
                included_positions="self_and_neighbor",
                n_local=12,
                lower_encode_pos=-box_lengths/ 2,
                upper_encode_pos=-box_lengths/ 2 + box_lengths_encode,
                lower_encode_diffs=-np.max(box_lengths_108) / 2,
                upper_encode_diffs=+np.max(box_lengths_108) / 2,
                lower=-box_lengths / 2,
                upper=box_lengths / 2,
                num_frequencies=num_frequencies,
                num_layers=2,
                lattice=base.lattice,
                use_layernorm=False,
                w_init_final=jnp.zeros,
            ),
        ),
    )

    config.model = dict(
        kwargs=dict(
            bijector=dict(
                constructor=augmented_coupling_flows.make_split_coupling_flow,
                kwargs=dict(
                    num_layers=8,
                    num_blocks_per_layer=1,
                    num_bins=16,
                    conditioner=conditioner,
                    split_axis=-2,
                    lower_spline=-np.max(box_lengths_108) / 2,
                    upper_spline=+np.max(box_lengths_108) / 2,
                    boundary_slopes="identity",
                    use_com_shift=use_com_shift,
                    prng=42,
                ),
            ),
            base=dict(
                constructor=lattice_model,
                kwargs=dict(
                    noise_scale=0.05,
                    noise_scale_aux=0.05,
                    lower=-box_lengths / 2.0,
                    upper=box_lengths / 2.0,
                    wrap=False,
                    remove_com=remove_com,
                ),
            ),
        ),
    )
    shared_kwargs = dict(
        cutoff=2.7,
        box_length=box_lengths,
        lambda_lj=1.,
        shift_energy=True)
    config.train_energy = dict(
        constructor=LennardJonesEnergy,
        kwargs=dict(min_distance=0.01, linearize_below=0.8, **shared_kwargs)
    )
    config.test_energy = dict(
        constructor=LennardJonesEnergy,
        kwargs=dict(min_distance=0., **shared_kwargs)
    )
    config.train = dict(
        batch_size=128,
        num_iterations=1000000,
        learning_rate=7e-5,
        learning_rate_decay_steps=[250000, 500000],
        learning_rate_decay_factor=0.1,
        seed=42,
        multi_gpu=True,
        every_k_schedule=1,
        save_dir="params",
        max_gradient_norm=10000.0,
    )
    config.test = dict(
        log_results=False,
        print_results=False,
        plot_results=True,
        test_every=1000,
        batch_size=1000,
    )
    return config
