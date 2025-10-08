import jax
from jax import numpy as jnp
import bgmat.utils.observable_utils as obs_utils
from bgmat.models.center_of_mass import split_augmented
import chex
import numpy as np
from typing import Union
from bgmat.utils.observable_utils import difference_pbc
import haiku as hk
import pickle
from bgmat.experiments.train_augmented_nvt import create_base, create_flow

Array = chex.Array
PRNGKey = Array
Numeric = Union[Array, float]
from bgmat.models.gnn_conditioner import wrap

import jax
import jax.numpy as jnp
import numpy as np


def select_one_device(pytree, idx=0):
    """
    Select the data for a single device from a pytree (e.g., after pmap).

    Args:
        pytree: A pytree where the leading axis indexes devices.
        idx: Index of the device to select (default 0).

    Returns:
        A pytree with the selected device's slices.
    """
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)


import jax
import jax.numpy as jnp
import haiku as hk
import pickle
from scipy.optimize import minimize


def flatten_jax_dict(input_dict):
    """
    Concatenate lists of JAX arrays stored in a dict along the first axis.

    This helper expects the values in `input_dict` to be lists or arrays
    with a leading device/batch axis and concatenates across that axis.

    Args:
        input_dict: Dictionary mapping keys to arrays or lists of arrays.

    Returns:
        A dictionary with same keys and concatenated arrays as values.
    """
    return jax.tree.map(lambda x: jnp.concatenate(x, axis=0), input_dict)


def postprocess_metrics(metrics: dict, num_particles: int):
    """
    Post-process evaluation metrics by aggregating per-device results and computing
    derived quantities like free-energy per particle and ESS.

    Args:
        metrics: Dictionary produced by evaluation containing per-split/device metrics.
        num_particles: Number of particles in the system (used to normalize free energies).

    Returns:
        A dictionary with aggregated metrics, including 'beta_f_per_particle' and 'ess'.
    """
    metrics_cat = flatten_jax_dict(metrics)
    log_probs = metrics_cat["log_probs"]

    metrics_out = {
        "beta_f_per_particle": -(obs_utils.compute_logz(**log_probs)) / num_particles,
        "ess": obs_utils.compute_ess(**log_probs),
    }
    return metrics_out


def build_eval_functions(config_flow, config_eval, context: float = None):
    """
    Build and return evaluation functions (Gibbs and Helmholtz free energy evaluators)
    for a given flow and evaluation configuration.

    The returned functions close over loaded model parameters and use the
    supplied configurations to compute free energies using importance sampling.

    Args:
        config_flow: Configuration object for the trained flow and system state.
        config_eval: Configuration object for evaluation parameters (filenames, sample sizes).
        context: Optional scalar context variable to pass to the flow when evaluating.

    Returns:
        Tuple (gibbs_free_energy, helmholtz_free_energy) where each function has the
        signature (box, num_samples, num_splits, rng_key) -> (free_energy, ...).
    """

    def eval_fn(new_box, num_samples):
        """
        Evaluate log-prob metrics for a given box size by sampling from the base
        distribution and mapping through the flow.

        Args:
            new_box: Array-like box lengths for each spatial dimension.
            num_samples: Number of base samples to draw for this evaluation.

        Returns:
            A metrics dictionary containing model and target log-probabilities.
        """
        state = config_flow.state
        base = create_base(config_flow)
        energy_fn = config_flow.test_energy.constructor(
            **config_flow.test_energy.kwargs
        )
        flow = create_flow(config_flow)

        rng_key = hk.next_rng_key()

        rng_key, base_key, add_key, defect_key = jax.random.split(rng_key, 4)
        base_samples, base_log_prob = base._sample_n_and_log_prob(
            key=base_key, n=num_samples
        )

        new_lattice = new_box * base.fractional_lattice
        new_lattice = jnp.tile(new_lattice, (num_samples, 1, 1))
        new_box_tiled = jnp.tile(new_box, (num_samples, 1))

        if context is not None:
            context_expanded = context * jnp.ones((num_samples, 1))
        else:
            context_expanded = None
        samples_mapped, log_det = flow.forward_and_log_det(
            base_samples,
            lattice=new_lattice,
            box_length=new_box_tiled,
            context=context_expanded,
        )

        displacements_phys, displacements_aux = split_augmented(samples_mapped)
        new_lower = -new_box_tiled / 2
        new_upper = new_box_tiled / 2

        samples_phys = jax.vmap(wrap)(
            displacements_phys + new_lattice, new_lower, new_upper
        )
        samples_aux = jax.vmap(wrap)(
            displacements_aux + new_lattice, new_lower, new_upper
        )

        pbc_diff_aux = jax.vmap(difference_pbc)(
            samples_aux, samples_phys, new_box_tiled
        )
        energies_aux = -base._noise_dist_aux.log_prob(pbc_diff_aux)

        log_prob = base_log_prob - log_det
        energies_phys = energy_fn(samples_phys, new_box)

        log_probs = {
            "model_log_probs": log_prob,
            "target_log_probs": -state.beta * energies_phys - energies_aux,
        }

        metrics = {
            "log_probs": log_probs,
        }
        return metrics

    _, apply_eval_fn = hk.transform(eval_fn)
    with open(config_eval.filename, "rb") as fp:
        params = pickle.load(fp)

    def helmholtz_free_energy(box, num_samples, num_splits, rng_key):
        """
        Compute the Helmholtz free energy per particle for a given box.

        Args:
            box: Box dimensions to evaluate.
            num_samples: Total number of samples to draw.
            num_splits: Number of splits to divide samples into for repeated estimation.
            rng_key: JAX PRNGKey for reproducibility.

        Returns:
            Tuple (beta_f_per_particle, ess) where beta_f_per_particle is the estimated
            Helmholtz free energy per particle and ess is the effective sample size.
        """

        # Always take same seed to make loss landscape more smootg
        # rng_key = jax.random.PRNGKey(config_eval.seed)
        test_keys = jax.random.split(rng_key, num_splits)

        num_samples_split = num_samples // num_splits
        metrics = jax.lax.map(
            lambda k: apply_eval_fn(params, k, box, num_samples_split),
            test_keys,
        )
        metrics_out = postprocess_metrics(metrics, config_flow.state.num_particles)
        beta_f_per_particle = metrics_out["beta_f_per_particle"]
        ess = metrics_out["ess"]
        return beta_f_per_particle, ess

    def gibbs_free_energy(box, num_samples, num_splits, rng_key):
        """
        Compute the Gibbs free energy per particle for a given box from the Helmholtz free energy
        by adding the PV term (pressure * volume / N) scaled by beta.

        Args:
            box: Box dimensions to evaluate.
            num_samples: Total number of samples to draw.
            num_splits: Number of splits for estimation.
            rng_key: JAX PRNGKey for reproducibility.

        Returns:
            Gibbs free energy per particle (scalar).
        """
        vol = jnp.prod(box)
        beta = config_flow.state.beta
        num_particles = config_eval.num_particles
        helmholtz_energy, _, _ = helmholtz_free_energy(
            box, num_samples, num_splits, rng_key
        )
        return helmholtz_energy + beta * config_eval.pressure * vol / num_particles

    return gibbs_free_energy, helmholtz_free_energy


def legendre_helmholtz_to_gibbs(
    config_flow,
    config_eval,
    h0,
    box_mode: str,
    pressures,
    context: float = None,
    print_res=False,
    max_iter: int = 200,
    tol: float = 1e-3,
):
    """
    Convert a set of Helmholtz free energy evaluations into Gibbs free energy minima
    over a range of pressures using Legendre-like optimization.

    This routine optimizes box dimensions for each pressure by minimizing the Gibbs
    free energy per particle using the trained flow to evaluate Helmholtz energies.

    Args:
        config_flow: Configuration for the trained flow and system state.
        config_eval: Configuration for evaluation parameters.
        h0: Initial box guess (array-like of box lengths or reduced parameters depending on box_mode).
        box_mode: Mode controlling box parametrization ('keep_ratio', 'a_eq_b', or 'full').
        pressures: Iterable of pressure values to evaluate.
        context: Optional scalar context to pass to the flow.
        print_res: If True, prints progress and results.
        max_iter: Maximum number of iterations for the optimizer.
        tol: Tolerance for optimization stopping criteria.

    Returns:
        A list of dictionaries containing optimization results for each pressure.
    """

    _, helmholtz_free_energy = build_eval_functions(config_flow, config_eval, context)
    _, helmholtz_free_energy_eval = build_eval_functions(
        config_flow, config_eval, context
    )

    jitted_helmholtz_free_energy = jax.jit(helmholtz_free_energy, static_argnums=(1, 2))
    jitted_helmholtz_free_energy_eval = jax.jit(
        helmholtz_free_energy_eval, static_argnums=(1, 2)
    )
    num_particles = config_eval.num_particles
    beta = config_flow.state.beta

    if box_mode == "keep_ratio":
        x0 = jnp.mean(h0)
        fixed_ratios = h0 / jnp.mean(h0)
        preprocess_box = lambda x: x * fixed_ratios
        postprocess_box = lambda x: x * fixed_ratios
        unprocess_box = lambda x: jnp.mean(x)

    elif box_mode == "a_eq_b":
        # optimize two variables: x[0] = a=b, x[1] = c
        x0 = jnp.array([h0[0], h0[2]])
        preprocess_box = lambda x: jnp.array([x[0], x[0], x[1]])  # a=b, c independent
        postprocess_box = lambda x: jnp.array([x[0], x[0], x[1]])
        unprocess_box = lambda box: jnp.array(
            [box[0], box[2]]
        )  # extract independent entries

    else:  # full flexible
        x0 = h0
        preprocess_box = lambda x: x
        postprocess_box = lambda x: x
        unprocess_box = lambda x: x

    results = []

    f_helmholtz_optimize = lambda b, k: jitted_helmholtz_free_energy(
        box=b,
        rng_key=k,
        num_samples=config_eval.test_size,
        num_splits=config_eval.num_splits,
    )
    f_helmholtz_evaluate = lambda b, k: jitted_helmholtz_free_energy_eval(
        box=b,
        rng_key=k,
        num_samples=config_eval.test_size_final,
        num_splits=config_eval.num_splits_final,
    )

    rng_key = jax.random.key(config_eval.seed)

    for press in pressures:
        history = {"last_val": None, "step": 0}

        test_key, rng_key = jax.random.split(rng_key)

        def wrapped_f_helmholtz_optimize(box):
            return f_helmholtz_optimize(box, test_key)

        def wrapped_f_helmholtz_evaluate(box):
            return f_helmholtz_evaluate(box, test_key)

        # Objective with tracking
        def fun_optimize(box):
            """
            Objective function (Gibbs free energy per particle) to minimize during optimization.

            Args:
                box: Current box parameterization in optimizer space.

            Returns:
                Gibbs free energy per particle evaluated for this box.
            """
            box_proc = preprocess_box(box)
            vol = np.prod(box_proc)

            helmholtz_energy, ess = wrapped_f_helmholtz_optimize(box_proc)
            gibbs_free_energy = helmholtz_energy + beta * press * vol / num_particles
            history["last_val"] = gibbs_free_energy
            return gibbs_free_energy

        def fun_eval(box):
            """
            Evaluate (without optimizing) the Gibbs and Helmholtz free energies and ESS
            for a given box parameterization.

            Args:
                box: Current box parameterization in optimizer space.

            Returns:
                Tuple (gibbs_free_energy, helmholtz_free_energy, ess).
            """
            box_proc = preprocess_box(box)
            vol = np.prod(box_proc)

            helmholtz_energy, ess = wrapped_f_helmholtz_evaluate(box_proc)
            gibbs_free_energy = helmholtz_energy + beta * press * vol / num_particles
            history["last_val"] = gibbs_free_energy
            return gibbs_free_energy, helmholtz_energy, ess

        # Callback function called once per iteration (Nelder-Mead supports callback)
        def callback(xk):
            """
            Callback called by the optimizer each iteration to optionally print progress.

            Args:
                xk: Current optimizer parameter vector.
            """
            fx = history["last_val"]
            step = history["step"]

            if print_res:
                xk_string = np.array2string(
                    xk, formatter={"float_kind": lambda x: f"{x:.10f}"}
                )
                print(f"Pressure {press} Step {step}: x = {xk_string}, f(x) = {fx}")
                history["step"] += 1

        res = minimize(
            fun_optimize,
            x0,
            method="Nelder-Mead",
            callback=callback,
            options={"maxiter": max_iter, "fatol": tol},
        )

        optimal_box = postprocess_box(res.x)
        optimal_gibbs_energy = res.fun
        optimal_helmholtz_energy, ess_opt = wrapped_f_helmholtz_optimize(optimal_box)

        ## Final evaluation on larger test size
        optimal_gibbs_energy_final, optimal_helmholtz_energy_final, ess_final = (
            fun_eval(res.x)
        )

        if print_res:

            print(f"\nPressure: {press}")
            print("Optimal box:", optimal_box)
            print("Optimal Gibbs free energy per particle:", optimal_gibbs_energy)
            print(
                "Optimal Helmholtz free energy per particle:", optimal_helmholtz_energy
            )
            print("Sampling efficiency:", ess_opt)

            print("\n:", optimal_gibbs_energy_final)

            print("Final Gibbs free energy per particle:", optimal_gibbs_energy_final)
            print(
                "Final Helmholtz free energy per particle:",
                optimal_helmholtz_energy_final,
            )

            results.append(
                {
                    "pressure": press,
                    "optimal_box": optimal_box,
                    "optimal_volume": np.prod(optimal_box),
                    "optimal_density": num_particles / np.prod(optimal_box),
                    "optimal_gibbs_energy": optimal_gibbs_energy,
                    "optimal_helmholtz_energy": optimal_helmholtz_energy,
                    "optimal_gibbs_energy (final)": optimal_gibbs_energy_final,
                    "optimal_helmholtz_energy (final)": optimal_helmholtz_energy_final,
                    "optimal_pv_term": beta
                    * press
                    * np.prod(optimal_box)
                    / num_particles,
                    "sampling_efficiency": ess_opt,
                    "sampling_efficiency (final)": ess_final,
                    "result": res,
                }
            )

        x0 = unprocess_box(optimal_box)

    return results
