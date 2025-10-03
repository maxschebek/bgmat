import jax
from jax import numpy as jnp
import bgmat.utils.observable_utils as obs_utils
from bgmat.models.center_of_mass import split_augmented
import chex
import numpy as np
from typing import Callable, Dict, Tuple, Union
from bgmat.utils.observable_utils import difference_pbc
import haiku as hk
import pickle
from bgmat.experiments.utils import (
    reshape_key,
    select_one_device,
    plot_results,
    init_fn_single_devices,
    concatenate_pmap_outputs,
)
from functools import partial
import gc

Array = chex.Array
PRNGKey = Array
Numeric = Union[Array, float]
from bgmat.experiments import utils

dimensionless_logvolume_mw = -3 * np.log(2.3925)
log_factorial = lambda n: np.sum(np.log(np.arange(n) + 1))
from bgmat.models.utils import count_parameters
import optax


def _get_loss(
    base,
    flow,
    energy_fn: Callable[[Array], Array],
    num_samples: int,
    state,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Compute the loss and statistics for the current model state.
    Args:
        base: Base distribution object.
        flow: Flow model object.
        energy_fn: Callable to compute physical energies.
        num_samples: Number of samples to draw.
        state: State/configuration object.
    Returns:
        Tuple of (loss, stats, displacements_phys, samples_phys).
    """
    rng_key = hk.next_rng_key()

    rng_key, base_key, add_key, defect_key = jax.random.split(
        rng_key,
        4,
    )

    base_samples, base_log_prob = base._sample_n_and_log_prob(
        key=base_key, n=num_samples
    )

    samples_mapped, log_det = flow.forward_and_log_det(base_samples)
    lattice_add = base.lattice

    displacements_phys, displacements_aux = split_augmented(samples_mapped)

    samples_phys = base.wrap(displacements_phys + lattice_add)
    samples_aux = base.wrap(displacements_aux + lattice_add)


    pbc_diff_aux = difference_pbc(samples_aux, samples_phys, base.width)

    energies_aux = -base._noise_dist_aux.log_prob(pbc_diff_aux)

    log_prob = base_log_prob - log_det
    energies_phys = energy_fn(samples_phys)

    energy_loss = jnp.mean(state.beta * energies_phys + energies_aux + log_prob)

    loss = energy_loss
    stats = {
        "energy": energies_phys,
        "model_log_prob": log_prob,
        "target_log_prob": -state.beta * energies_phys - energies_aux,
    }
    return loss, stats, displacements_phys, samples_phys


def create_base(config):
    """
    Construct the base distribution from the config.
    Args:
        config: Configuration object.
    Returns:
        Base distribution object.
    """
    state = config.state
    return config.model.kwargs.base["constructor"](
        num_particles=state.num_particles,
        **config.model.kwargs.base.kwargs,
    )


def create_flow(config):
    """
    Construct the flow model from the config.
    Args:
        config: Configuration object.
    Returns:
        Flow model object.
    """
    return config.model.kwargs.bijector["constructor"](
        event_shape=(
            2 * (config.state.num_particles),
            # 2,
            3,
        ),
        **config.model.kwargs.bijector.kwargs,
    )


def call_eval_fn_pmap(params, rng_keys, num_particles, system, eval_fn: Callable):
    """
    Evaluate the model in parallel across devices using pmap.
    Args:
        params: Model parameters.
        rng_keys: Random number generator keys for each device.
        num_particles: Number of particles in the system.
        system: System identifier.
        eval_fn: Evaluation function to call.
    Returns:
        Dictionary of evaluation metrics.
    """
    metrics_pmap, stats_pmap = eval_fn(params, rng_keys)
    # stats_flat = concatenate_pmap_outputs(stats_pmap)
    stats_flat = concatenate_pmap_outputs(stats_pmap)
    loss = select_one_device(metrics_pmap["loss"], True)
    log_probs = {
        "model_log_probs": stats_flat["model_log_prob"],
        "target_log_probs": stats_flat["target_log_prob"],
    }

    beta_f = -obs_utils.compute_logz(**log_probs) / num_particles

    metrics = {
        "ess": obs_utils.compute_ess(**log_probs),
        "logz": obs_utils.compute_logz(**log_probs),
        "beta_f_per_particle": beta_f,
        "loss": loss,
        "beta_f_per_particle_marginal": None,
        "ess_marginal": None,
    }

    return metrics


def call_eval_fn(params, rng_keys, num_particles, system, eval_fn: Callable):
    """
    Evaluate the model on a single device.
    Args:
        params: Model parameters.
        rng_keys: Random number generator keys.
        num_particles: Number of particles in the system.
        system: System identifier.
        eval_fn: Evaluation function to call.
    Returns:
        Dictionary of evaluation metrics.
    """
    metrics, _ = eval_fn(params, rng_keys)
    return metrics


def train(config, system, job_id=None):
    """
    Main training loop for the flow model.
    Args:
        config: Configuration object.
        system: System identifier.
    """
    lr_schedule_fn = utils.get_lr_schedule(
        config.train.learning_rate,
        config.train.learning_rate_decay_steps,
        config.train.learning_rate_decay_factor,
    )
    state = config.state

    n_devices = jax.device_count()

    config.train.batch_size = config.train.batch_size // n_devices
    config.test.batch_size = config.test.batch_size // n_devices

    def update_fn(params, opt_state, rng_key):
        """
        Update function for a single device.
        Args:
            params: Model parameters.
            opt_state: Optimizer state.
            rng_key: Random number generator key.
        Returns:
            Tuple of (metrics, updated_params, updated_opt_state).
        """
        (_, metrics), g = jitted_loss(params, rng_key)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return metrics, params, opt_state

    @partial(jax.pmap, axis_name="num_devices")
    def update_fn_pmap(params, opt_state, rng_key):
        """
        Update function for multiple devices using pmap.
        Args:
            params: Model parameters.
            opt_state: Optimizer state.
            rng_key: Random number generator key for each device.
        Returns:
            Tuple of (metrics, updated_params, updated_opt_state).
        """
        (_, metrics), g = jitted_loss(params, rng_key)
        g = jax.lax.pmean(g, axis_name="num_devices")
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return metrics, params, opt_state

    deltaFs = jnp.empty(shape=(0,))
    efficiencies = jnp.empty(shape=(0,))
    losses = jnp.empty(shape=(0,))

    # Loss function setup
    energy_fn_train = config.train_energy.constructor(**config.train_energy.kwargs)
    energy_fn_test = config.test_energy.constructor(**config.test_energy.kwargs)

    optimizer = optax.chain(
        optax.scale_by_adam(), optax.scale_by_schedule(lr_schedule_fn), optax.scale(-1)
    )

    if config.train.max_gradient_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.train.max_gradient_norm), optimizer
        )
    if config.train.every_k_schedule > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=config.train.every_k_schedule
        )

    def loss_fn():
        """
        Loss function for training.
        Returns:
            Tuple of (loss, metrics).
        """
        base = create_base(config)
        flow = create_flow(config)

        loss, stats, _, _ = _get_loss(
            base=base,
            flow=flow,
            energy_fn=energy_fn_train,
            state=state,
            num_samples=config.train.batch_size // config.train.every_k_schedule,
        )

        metrics = {
            "loss": loss,
            "energy": jnp.mean(stats["energy"]),
            "model_entropy": -jnp.mean(stats["model_log_prob"]),
        }
        return loss, metrics

    def eval_fn():
        """
        Evaluation function for validation/testing.
        Returns:
            Tuple of (metrics, stats).
        """
        base = create_base(config)

        flow = create_flow(config)
        loss, stats, _, _ = _get_loss(
            base=base,
            flow=flow,
            energy_fn=energy_fn_test,
            state=state,
            num_samples=config.test.batch_size,
        )

        log_probs = {
            "model_log_probs": stats["model_log_prob"],
            "target_log_probs": stats["target_log_prob"],
        }

        # Do not subtract the log N! if we use the com shift
        beta_f = -obs_utils.compute_logz(**log_probs) / state.num_particles

        metrics = {
            "loss": loss / state.num_particles,
            "energy": jnp.mean(stats["energy"]),
            "model_entropy": -jnp.mean(stats["model_log_prob"]),
            "ess": obs_utils.compute_ess(**log_probs),
            "logz": obs_utils.compute_logz(**log_probs),
            "logz_per_particle": obs_utils.compute_logz(**log_probs)
            / state.num_particles,
            "beta_f_per_particle": beta_f,
            "ess_marginal": None,
        }
        return metrics, stats

    rng_key = jax.random.PRNGKey(config.train.seed)
    rng_key, init_key = jax.random.split(rng_key)

    init_fn, apply_fn = hk.transform(loss_fn)
    _, apply_eval_fn = hk.transform(eval_fn)

    print(f"Initialising system {system}")

    def _loss(params, rng):
        """
        Compute loss and metrics given parameters and RNG.
        Args:
            params: Model parameters.
            rng: Random number generator key.
        Returns:
            Tuple of (loss, metrics).
        """
        loss, metrics = apply_fn(params, rng)
        return loss, metrics

    jitted_loss = jax.jit(jax.value_and_grad(_loss, has_aux=True))
    jitted_eval = jax.jit(apply_eval_fn)

    @partial(jax.pmap, axis_name="num_devices")
    def eval_fn_pmap(
        params,
        rng_key,
    ):
        """
        Evaluate the model in parallel across devices using pmap.
        Args:
            params: Model parameters.
            rng_key: Random number generator key for each device.
        Returns:
            Tuple of (metrics, stats).
        """
        metrics, stats = jitted_eval(params, rng_key)

        # average intensive quantities
        loss_mean = jax.lax.pmean(metrics["loss"], axis_name="num_devices")
        metrics["loss"] = loss_mean

        energy_mean = jax.lax.pmean(metrics["energy"], axis_name="num_devices")
        metrics["energy"] = energy_mean

        entropy_mean = jax.lax.pmean(metrics["model_entropy"], axis_name="num_devices")
        metrics["model_entropy"] = entropy_mean

        return metrics, stats

    # Multi-gpu setup
    if config.train.multi_gpu:
        common_keys = jnp.repeat(init_key[None, ...], n_devices, axis=0)
        params, opt_state = jax.pmap(
            lambda k: init_fn_single_devices(init_fn, k, optimizer)
        )(common_keys)

        update_function = update_fn_pmap
        eval_caller = call_eval_fn_pmap
        eval_function = eval_fn_pmap

        print(f"Beginning of multi-device training on {n_devices} device(s).")

    else:
        update_function = jax.jit(update_fn)
        eval_caller = call_eval_fn
        eval_function = jitted_eval
        params, opt_state = init_fn_single_devices(init_fn, init_key, optimizer)

        print(f"Beginning of single-device training on {n_devices} device(s).")

    print(f"Number of model params:{count_parameters(params):,}")
    step = 0

    while step < config.train.num_iterations:
        # Training update.
        rng_key, loss_key = jax.random.split(rng_key)
        per_device_keys = jax.random.split(loss_key, n_devices)
        per_device_keys = reshape_key(per_device_keys, config.train.multi_gpu)

        metrics, params, opt_state = update_function(
            params,
            opt_state,
            per_device_keys,
        )

        # Evaluation
        if (step % config.test.test_every) == 0:

            rng_key, per_device_key_eval = jax.random.split(rng_key)
            per_device_keys_eval = jax.random.split(per_device_key_eval, n_devices)
            per_device_keys_eval = reshape_key(
                per_device_keys_eval, config.train.multi_gpu
            )

            metrics = eval_caller(
                params=params,
                rng_keys=per_device_keys_eval,
                num_particles=state.num_particles,
                eval_fn=eval_function,
                system=system,
            )

            if config.test.plot_results:
                losses = jnp.concatenate((losses, jnp.array([metrics["loss"]])))

                deltaFs = jnp.concatenate(
                    (deltaFs, jnp.array([metrics["beta_f_per_particle"]]))
                )
                efficiencies = jnp.concatenate(
                    (efficiencies, jnp.array([metrics["ess"]]))
                )
                plot_results(losses, efficiencies, deltaFs, config)

            if job_id is not None:
                params_1 = select_one_device(params, config.train.multi_gpu)
                # save dictionary to person_data.pkl file
                with open(f"{config.train.save_dir}/params-{job_id}.pkl", "wb") as fp:
                    pickle.dump(params_1, fp)

        step += 1
