#
# Copyright 2025 Maximilian Schebek, Freie Universität Berlin
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

from jax import numpy as jnp
from bgmat.models.center_of_mass import split_augmented
import jax


def marginal_sample(pos_phys, pos_aux, flow, base):
    """
    Compute the marginal log-probability for a given physical and auxiliary position using the flow and base distribution.
    Only implemented for displacements and unconditional setting.
    Args:
        pos_phys: Physical positions array.
        pos_aux: Auxiliary positions array.
        flow: Flow model object.
        base: Base distribution object.
    Returns:
        Marginal log-probability for the sample.
    """

    base_samples = jnp.concatenate((pos_phys, pos_aux), axis=-2)

    samples_mapped_inv, ldj = flow.inverse_and_log_det(base_samples)
    displacements_phys_inv, displacements_aux_inv = split_augmented(samples_mapped_inv)
    d = jnp.prod(jnp.array(displacements_phys_inv.shape))

    log_norm = -0.5 * d * jnp.log(2 * jnp.pi) - jnp.sum(
        jnp.log(base._noise_dist.distribution.scale)
    )

    diffs_aux = pos_aux - pos_phys
    diffs_aux_inv = displacements_aux_inv - displacements_phys_inv

    log_unnormalized_aux = base._noise_dist_aux.log_prob(diffs_aux)
    log_unnormalized_aux_inv = base._noise_dist_aux.log_prob(diffs_aux_inv)
    log_unnormalized_phys_inv = (
        base._noise_dist.log_prob(displacements_phys_inv) - log_norm
    )

    energies_aux = -log_unnormalized_aux
    energy_aux_inv = -log_unnormalized_aux_inv
    energy_phys_inv = -log_unnormalized_phys_inv

    u_prior = energy_phys_inv + energy_aux_inv
    u_conditional = energies_aux
    p_marginal = -u_prior + ldj.squeeze() + u_conditional
    return p_marginal


def log_marginal_weight_given_generated(
    input_tuple,
    n_marginal,
    flow,
    base,
    target_beta,
):
    """
    Computes the marginal log q for one sample of the generated augmented distribution according to App. B9 of the SE(3) equivariant coupling flow paper.
    Args:
        input_tuple: Tuple containing (displacements_phys, energy_phys, key).
        n_marginal: Number of auxiliary samples drawn for evaluation.
        flow: Flow model object.
        base: Base distribution object.
        target_beta: Inverse temperature parameter.
    Returns:
        Tuple of (log_p, log_q) for the sample.
    """
    displacements_phys, energy_phys, key = input_tuple

    noise_0 = base._noise_dist_aux.sample(sample_shape=n_marginal, seed=key)
    batch_pos_aux = noise_0 + displacements_phys

    log_p = -energy_phys * target_beta
    marginals = jax.vmap(
        lambda p: marginal_sample(
            pos_phys=displacements_phys.reshape(1, *displacements_phys.shape),
            pos_aux=p,
            flow=flow,
            base=base,
        )
    )(batch_pos_aux.reshape(batch_pos_aux.shape[0], 1, *batch_pos_aux.shape[1:]))
    log_q = -jnp.log(len(marginals)) + jax.scipy.special.logsumexp(marginals, axis=0)
    return log_p.flatten(), log_q.flatten()
