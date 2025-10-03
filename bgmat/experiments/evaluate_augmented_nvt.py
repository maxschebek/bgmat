import jax
from jax import numpy as jnp
import bgmat.utils.observable_utils as obs_utils
from bgmat.models.center_of_mass import split_augmented
import chex
import numpy as np
from typing import Callable, Dict, Tuple, Union
from bgmat.utils.observable_utils import difference_pbc
import haiku as hk
from bgmat.utils.marginal import log_marginal_weight_given_generated
import pickle 
from bgmat.experiments.train_augmented_nvt import  create_base, create_flow, _get_loss
from bgmat.experiments.utils import print_results
Array = chex.Array
PRNGKey = Array
Numeric = Union[Array, float]

def flatten_jax_dict(input_dict):
    """
    Concatenate all arrays in a JAX pytree dictionary along the first axis.
    Args:
        input_dict: Dictionary of arrays or pytrees.
    Returns:
        Dictionary with arrays concatenated along axis 0.
    """
    return jax.tree.map(lambda x: jnp.concatenate(x, axis=0), input_dict)

def postprocess_metrics(metrics: dict, num_particles: int):
    """
    Post-process and aggregate evaluation metrics, including log-probs and ESS.
    Args:
        metrics: Dictionary of raw metrics from evaluation.
        num_particles: Number of particles in the system.
    Returns:
        Dictionary of processed and aggregated metrics.
    """
    metrics_cat = flatten_jax_dict(metrics)
    log_probs = metrics_cat["log_probs"]
    log_probs_marginal = metrics_cat["log_probs_marginal"]

    if log_probs_marginal["target_log_probs"] is not None:
        beta_f_marginal = -(
                obs_utils.compute_logz(**log_probs_marginal)
            )  / num_particles
        ess_marginal =  obs_utils.compute_ess(**log_probs_marginal)
    else:
        beta_f_marginal = None
        ess_marginal = None
    
    metrics_out = {
        "loss": jnp.mean(metrics_cat["loss"]),
        "samples_mapped": metrics_cat["samples_mapped"],
        "energies": metrics_cat["energy"].reshape(-1),
        "beta_f_per_particle": -(
            obs_utils.compute_logz(**log_probs)
        )
        / num_particles,
        "weights": obs_utils._compute_importance_weights(**log_probs),
        "ess": obs_utils.compute_ess(**log_probs),
        "ess_marginal": ess_marginal,
        "beta_f_per_particle_marginal": beta_f_marginal,
    }


    return metrics_out

def select_one_device(pytree, idx=0):
    """
    Select the data for a single device from a pytree (e.g., after pmap).
    Args:
        pytree: Pytree of arrays with device axis first.
        idx: Index of the device to select (default: 0).
    Returns:
        Pytree with only the selected device's data.
    """
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)

def evaluate(config_flow, config_eval):
    """
    Evaluate a trained flow model using the provided configuration.
    Args:
        config_flow: Configuration object for the flow.
        config_eval: Configuration object for evaluation.
    Returns:
        Dictionary of evaluation metrics and results.
    """

    def eval_fn():
        """
        Evaluation function for a single evaluation batch.
        Returns:
            Dictionary of metrics for the batch.
        """
        state=config_flow.state
        base = create_base(config_flow)
        energy_fn_test = config_flow.test_energy.constructor(**config_flow.test_energy.kwargs)

        flow = create_flow(config_flow)
        loss, stats, displacements_mapped, samples_mapped = _get_loss(
            base=base,
            flow=flow,
            energy_fn=energy_fn_test,
            num_samples=config_eval.test_size // config_eval.num_splits,
            state=state
        )

        log_probs = {
            "model_log_probs": stats["model_log_prob"],
            "target_log_probs": stats["target_log_prob"],
        }
        rng_key = hk.next_rng_key()

        energy_phys = stats["energy"]
        if config_eval.return_marginal:
            key_array = jax.random.split(rng_key, samples_mapped.shape[0] )
            
            input_tuple = (displacements_mapped, energy_phys, key_array)

            target_log_probs_marginal, model_log_probs_marginal = jax.lax.map(
            lambda p: log_marginal_weight_given_generated(
                n_marginal=config_eval.n_marginal,
                flow=flow,
                base=base,
                target_beta=state.beta,
                input_tuple=p,
            ),
            input_tuple,
            # batch_size=5,
        )
            log_probs_marginal = {
            "model_log_probs": model_log_probs_marginal,
            "target_log_probs": target_log_probs_marginal,
        }
            
        else:
            log_probs_marginal = {
            "model_log_probs": None,
            "target_log_probs": None,
        }


        metrics = {
            "loss": loss.reshape(-1, 1),
            "energy": stats["energy"].reshape(-1, 1),
            # "beta_f_per_particle": beta_f,
            "log_probs": log_probs,
            "log_probs_marginal": log_probs_marginal,
            "samples_mapped": samples_mapped,
            "displacements_mapped": displacements_mapped
        }

        return metrics

    rng_key = jax.random.key(config_eval.seed)
    _, apply_eval_fn = hk.transform(eval_fn)

    # Read dictionary pkmodel_samplesl file
    with open(config_eval.filename, "rb") as fp:
        params = pickle.load(fp)
        
    jitted_eval =  jax.jit(apply_eval_fn)
    
    for _ in range(config_eval.eval_steps):
        test_key, rng_key = jax.random.split(rng_key)
        test_keys = jax.random.split(test_key,config_eval.num_splits )


        metrics = jax.lax.map(
            lambda k: jitted_eval(params, k),
            test_keys,
        )

        metrics_out = postprocess_metrics(metrics, config_flow.state.num_particles)
        
        print_results(metrics_out)

    metrics_report ={
        "samples_mapped" :metrics_out["samples_mapped"], 
        "energies" :metrics_out["energies"], 
        "ess" :metrics_out["ess"], 
        "beta_f_per_particle" :metrics_out["beta_f_per_particle"], 
        "beta_f_per_particle_marginal" :metrics_out["beta_f_per_particle_marginal"], 
        "ess_marginal" :metrics_out["ess_marginal"], 
    } 
    return metrics_report


