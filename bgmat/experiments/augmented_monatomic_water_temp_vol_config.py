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

from bgmat.experiments import utils
from bgmat.models.gnn_conditioner import GNN
from bgmat.models import augmented_coupling_flows
from bgmat.models import particle_models
from bgmat.systems.monatomic_water import MonatomicWaterEnergy
import jax.numpy as jnp
import numpy as np
import distrax
from ml_collections import config_dict

# Density, temperature and system shapes and sizes below chosen for comparison
# with the paper by Quigley (https://doi.org/10.1063/1.4896376, see Table 1).
BOLTZMANN_CONSTANT = 0.0019872067  # in units of kcal/mol K
QUIGLEY_DENSITY = 0.033567184  # inverse cubic Angstrom
QUIGLEY_TEMPERATURE = 200.0  # Kelvin

BOX_FUNS = {
    "hex": utils.get_hexagonal_box_lengths,
    "cubic": utils.get_cubic_box_lengths,
}

LATTICE_MODELS = {
    "hex": particle_models.AugmentedHexagonalIceLattice,
    "cubic": particle_models.AugmentedDiamondCubicLattice,
}

FREQUENCIES = {
    8: 8,
    64: 8,
    216: 8,
    512: 8,
    1000: 8,
}
BOX_FUNS_CONDITIONAL = {
    'cubic': utils.get_orthorhombic_box_lengths_jax,
    'betatin': utils.get_beta_tin_box_lengths_jax,
}


def get_config(num_particles: int, lattice: str) -> config_dict.ConfigDict:
    """Returns the config."""
    num_particles_encode = 8
    box_fun = BOX_FUNS[lattice]
    box_fun_conditional = BOX_FUNS_CONDITIONAL[lattice]

    lattice_model = LATTICE_MODELS[lattice]
    box_lengths = box_fun(num_particles, density=QUIGLEY_DENSITY, dim=3)
    box_lengths_encode = box_fun(num_particles_encode, density=QUIGLEY_DENSITY, dim=3)
    box_lengths_64 = box_fun(64, density=QUIGLEY_DENSITY, dim=3)

    num_frequencies = 8

    remove_com = True
    use_com_shift = True
    recompute_neighbors = False

    # Temperature and density ranges for conditional training
    density_low = 0.050
    density_high = 0.060
    eval_density = 0.058

    temp_low = 350
    temp_high = 1050
    eval_temp = 600
    repeats_state = None

    config = config_dict.ConfigDict()
    config.state = dict(
        num_particles=num_particles,
        kb=BOLTZMANN_CONSTANT,
        eval_density=eval_density,
        eval_temp=eval_temp,
        box_fn=box_fun_conditional,
        repeats=repeats_state,
        ratio_params={"min_ba": 0.99, "max_ba": 1.01, "min_ca": 0.99, "max_ca": 1.01},
        has_defect=False,
    )

    def create_base():
        return lattice_model(
            num_particles=num_particles,
            lower=-box_lengths / 2.0,
            upper=box_lengths / 2.0,
            noise_scale=0.2,
            noise_scale_aux=0.2,
            remove_com=remove_com,
        )

    base = create_base()

    conditioner = dict(
        constructor=augmented_coupling_flows.make_equivariant_conditioner,
        kwargs=dict(
            conditioner_constructor=GNN,
            conditioner_kwargs=dict(
                embedding_size=32,
                encode_diffs=True,
                recompute_neighbors=recompute_neighbors,
                included_positions="self_and_neighbor",
                #    n_local=16,
                lower_encode_pos=-box_lengths / 2,
                upper_encode_pos=-box_lengths / 2 + box_lengths_encode,
                lower_encode_diffs=-np.max(box_lengths_64) / 2,
                upper_encode_diffs=+np.max(box_lengths_64) / 2,
                lower=-box_lengths / 2,
                upper=box_lengths / 2,
                num_frequencies=num_frequencies,
                num_layers=2,
                lattice=base.lattice,
                use_layernorm=False,
                w_init_final=jnp.zeros,
            ),
        ),
        context_kwargs=dict(
            lower_encode=temp_low,
            upper_encode=temp_high,
            num_frequencies=num_frequencies,
        ),
    )

    config.model = dict(
        kwargs=dict(
            bijector=dict(
                constructor=augmented_coupling_flows.make_split_coupling_flow,
                kwargs=dict(
                    num_layers=4,
                    num_blocks_per_layer=1,
                    num_bins=16,
                    conditioner=conditioner,
                    split_axis=-2,
                    lower_spline=-np.max(box_lengths_64) / 2,
                    upper_spline=+np.max(box_lengths_64) / 2,
                    boundary_slopes="identity",
                    use_com_shift=use_com_shift,
                    prng=42,
                ),
            ),
            base=dict(
                constructor=lattice_model,
                kwargs=dict(
                    noise_scale=0.2,
                    noise_scale_aux=0.2,
                    lower=-box_lengths / 2.0,
                    upper=box_lengths / 2.0,
                    wrap=False,
                    remove_com=remove_com,
                ),
            ),
            base_vol=dict(
                constructor=distrax.Uniform,
                kwargs=dict(low=density_low, high=density_high),
            ),
            base_temp=dict(
                constructor=distrax.Uniform,
                kwargs=dict(low=temp_low, high=temp_high),
            ),
        ),
    )
    shared_kwargs = dict(box_length=box_lengths)
    config.train_energy = dict(
        constructor=MonatomicWaterEnergy,
        kwargs=dict(min_distance=0.01, linearize_below=1.2, **shared_kwargs),
    )
    config.test_energy = dict(
        constructor=MonatomicWaterEnergy, kwargs=dict(**shared_kwargs)
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
