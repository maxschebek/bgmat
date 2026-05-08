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
# Modified: 2026-05-08 - Adapted and extended for bgmat project. Further modifications should be documented here.

"""Config file for monatomic water in the cubic or hexagonal ice phases."""

from bgmat.experiments import utils
from bgmat.models.gnn_conditioner import GNN
from bgmat.models import augmented_coupling_flows
from bgmat.models import particle_models
from bgmat.systems.stillinger_weber import (
    StillingerWeberEnergy,
    MW_LAMBDA,
    MW_EPSILON,
    MW_SIGMA,
)
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict
import distrax


BOLTZMANN_CONSTANT = 0.0019872067  # in units of kcal/mol K
TEMPERATURE = 75
BOX_FUNS = {
    "hex": utils.get_hexagonal_box_lengths,
    "cubic": utils.get_cubic_box_lengths,
    "bcc": utils.get_cubic_box_lengths,
    "betatin": utils.get_beta_tin_box_lengths,
}


LATTICE_MODELS = {
    "hex": particle_models.AugmentedHexagonalIceLattice,
    "cubic": particle_models.AugmentedDiamondCubicLattice,
    "bcc": particle_models.AugmentedBodyCenteredCubicLattice,
    "betatin": particle_models.AugmentedBetaTinLattice,
}
OPT_DENSITIES = {"cubic": 33.484 / 1000, "betatin": 41.480 / 1000, "bcc": 42.263 / 1000}

NUM_LOCAL = {"cubic": 16, "hex": 17, "bcc": 14, "betatin": 18}
NUM_ENCODE = {"cubic": 8, "hex": 8, "bcc": 16, "betatin": 4}

NUM_ENCODE_DIFF = {"cubic": 64, "hex": 64, "bcc": 54, "betatin": 64}
FREQUENCIES = {
    8: 8,
    64: 8,
    216: 8,
    512: 8,
    1000: 8,
}

MAX_CA = {"cubic": 1.01, "bcc": 1.01, "betatin": 0.56}

MIN_CA = {"cubic": 0.99, "bcc": 0.99, "betatin": 0.53}


MIN_DENSITY = OPT_DENSITIES
MAX_DENSITY = OPT_DENSITIES


def get_config(
    num_particles: int, lattice: str, repeats=None
) -> config_dict.ConfigDict:
    """Returns the config."""

    n_local = NUM_LOCAL[lattice]
    box_fun = BOX_FUNS[lattice]
    lattice_model = LATTICE_MODELS[lattice]
    num_particles_encode = NUM_ENCODE[lattice]
    num_particles_diff = NUM_ENCODE_DIFF[lattice]
    density = OPT_DENSITIES[lattice]
    box_fun_conditional = utils.get_orthorhombic_box_lengths_jax

    box_lengths = box_fun(num_particles, density=density, dim=3, repeats=repeats)

    box_lengths_diff = np.max(
        box_fun(num_particles_diff, density=density, dim=3, repeats=[2, 2, 4])
    )
    box_lengths_encode = box_fun(
        num_particles_encode, density=density, dim=3, repeats=None
    )
    encoding_length = box_lengths_diff
    spline_length = box_lengths_diff

    num_frequencies = 8

    remove_com = True
    use_com_shift = True

    density_low = MIN_DENSITY[lattice]
    density_high = MAX_DENSITY[lattice]
    eval_density = (density_high - density_low) / 2 + density_low

    lambda_low = 21
    lambda_high = 21
    eval_lambda = (lambda_high - lambda_low) / 2 + lambda_low

    min_ca = MIN_CA[lattice]
    max_ca = MAX_CA[lattice]
    repeats_state = None if repeats is None else jnp.array(repeats)
    config = config_dict.ConfigDict()
    config.state = dict(
        num_particles=num_particles,
        lambda_low=lambda_low,
        lambda_high=lambda_high,
        kb=BOLTZMANN_CONSTANT,
        eval_density=eval_density,
        eval_lambda=eval_lambda,
        beta=1.0 / (TEMPERATURE * BOLTZMANN_CONSTANT),
        box_fn=box_fun_conditional,
        repeats=repeats_state,
        ratio_params={
            "min_ba": 1.01,
            "max_ba": 1.01,
            "min_ca": min_ca,
            "max_ca": max_ca,
        },
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
                included_positions="self_and_neighbor",
                # n_local=n_local,
                lower_encode_pos=-box_lengths / 2,
                upper_encode_pos=-box_lengths / 2 + box_lengths_encode,
                lower_encode_diffs=-encoding_length / 2,
                upper_encode_diffs=+encoding_length / 2,
                upper=box_lengths / 2,
                lower=-box_lengths / 2,
                num_frequencies=num_frequencies,
                num_layers=2,
                lattice=base.lattice,
                use_layernorm=False,
                w_init_final=jnp.zeros,
            ),
        ),
        context_kwargs=dict(
            lower_encode=lambda_low,
            upper_encode=lambda_high,
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
                    use_com_shift=use_com_shift,
                    lower_spline=-spline_length / 2,
                    upper_spline=+spline_length / 2,
                    boundary_slopes="identity",
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
                kwargs=dict(
                    low=density_low,
                    high=density_high,
                ),
            ),
            base_lambda=dict(
                constructor=distrax.Uniform,
                kwargs=dict(
                    low=lambda_low,
                    high=lambda_high,
                ),
            ),
        ),
    )

    shared_kwargs = dict(box_length=box_lengths, epsilon=MW_EPSILON, sigma=MW_SIGMA, lambda_three_body=MW_LAMBDA)
    config.train_energy = dict(
        constructor=StillingerWeberEnergy,
        kwargs=dict(min_distance=0.01, linearize_below=1.2, **shared_kwargs),
    )
    config.test_energy = dict(
        constructor=StillingerWeberEnergy, kwargs=dict(**shared_kwargs)
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
