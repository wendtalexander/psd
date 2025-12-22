import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxon.models.punit import PUnitParams
from jaxon.utils.output import SIMOutput


def simulate_vdend(
    key: ArrayLike, stimulus: ArrayLike, params: PUnitParams
) -> SIMOutput:
    """Simulate a P-unit.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params: PUnitParams
        Parameter class

    Returns
    -------
    SIMOutput: class
        Simulation output class which holds both binary spikes and membrane voltage

    """
    noise = jax.random.normal(key, shape=stimulus.shape)
    noise *= params.noise_strength / jnp.sqrt(params.deltat)

    stimulus = jnp.maximum(stimulus, 0.0)

    def step(carry, x):
        v_mem, v_dend, adapt, last_spike_time, time_index = carry
        stim_i, noise_i = x
        current_time = time_index * params.deltat

        # Update dendritic voltage
        v_dend_new = v_dend + (-v_dend + stim_i) / params.dend_tau * params.deltat

        dv_mem = (
            (
                params.v_base
                - v_mem
                + params.v_offset
                + (v_dend_new * params.input_scaling)
                - adapt
                + noise_i
            )
            / params.mem_tau
            * params.deltat
        )

        # Update membrane potential
        v_mem_new = v_mem + dv_mem

        adapt_new = adapt - (adapt / params.tau_a) * params.deltat

        # Check for refractory period and reset membrane potential if needed
        is_refractory = (last_spike_time >= 0) & (
            current_time - last_spike_time < params.ref_period + params.deltat / 2
        )
        v_mem_ref = jnp.where(is_refractory, params.v_base, v_mem_new)

        # Check for threshold crossing
        spike_fired = v_mem_ref > params.threshold

        # Apply spike effects conditionally using jnp.where
        final_v_mem = jnp.where(spike_fired, params.v_base, v_mem_ref)
        final_adapt = jnp.where(
            spike_fired, adapt_new + params.delta_a / params.tau_a, adapt_new
        )
        final_last_spike_time = jnp.where(spike_fired, current_time, last_spike_time)

        new_carry = (
            final_v_mem,
            v_dend_new,
            final_adapt,
            final_last_spike_time,
            time_index + 1,
        )
        # WARNING: adding v_dend_new to output
        return new_carry, (spike_fired, final_v_mem, v_dend_new * params.input_scaling)

    # Initial conditions for the scan
    initial_carry = (params.v_zero, stimulus[0], params.a_zero, -1.0, 0)

    # Run the simulation using jax.lax.scan
    _, (spikes, vmem, v_dend) = jax.lax.scan(step, initial_carry, (stimulus, noise))

    # WARNING: adding v_dend_new to output
    return SIMOutput(spikes.astype(jnp.int32), v_dend)


def simulate_without_adaptation(
    key: ArrayLike, stimulus: ArrayLike, params: PUnitParams
) -> SIMOutput:
    """Simulate a P-unit.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params: PUnitParams
        Parameter class

    Returns
    -------
    SIMOutput: class
        Simulation output class which holds both binary spikes and membrane voltage

    """
    noise = jax.random.normal(key, shape=stimulus.shape)
    noise *= params.noise_strength / jnp.sqrt(params.deltat)

    stimulus = jnp.maximum(stimulus, 0.0)

    def step(carry, x):
        v_mem, v_dend, adapt, last_spike_time, time_index = carry
        stim_i, noise_i = x
        current_time = time_index * params.deltat

        # Update dendritic voltage
        v_dend_new = v_dend + (-v_dend + stim_i) / params.dend_tau * params.deltat

        dv_mem = (
            (
                params.v_base
                - v_mem
                + params.v_offset
                + (v_dend_new * params.input_scaling)
                # WARNING: Dont substract the adaptation
                # - adapt
                + noise_i
            )
            / params.mem_tau
            * params.deltat
        )

        # Update membrane potential
        v_mem_new = v_mem + dv_mem
        # WARNING:remove adaptation
        # adapt_new = adapt - (adapt / params.tau_a) * params.deltat

        # Check for refractory period and reset membrane potential if needed
        is_refractory = (last_spike_time >= 0) & (
            current_time - last_spike_time < params.ref_period + params.deltat / 2
        )
        v_mem_ref = jnp.where(is_refractory, params.v_base, v_mem_new)

        # Check for threshold crossing
        spike_fired = v_mem_ref > params.threshold

        # Apply spike effects conditionally using jnp.where
        final_v_mem = jnp.where(spike_fired, params.v_base, v_mem_ref)
        # WARNING: remove adaptation
        # final_adapt = jnp.where(
        #     spike_fired, adapt_new + params.delta_a / params.tau_a, adapt_new
        # )
        final_last_spike_time = jnp.where(spike_fired, current_time, last_spike_time)

        new_carry = (
            final_v_mem,
            v_dend_new,
            0.0,  # final_adapt,#WARNING: remove adaptation
            final_last_spike_time,
            time_index + 1,
        )
        return new_carry, (spike_fired, final_v_mem)

    # Initial conditions for the scan
    initial_carry = (params.v_zero, stimulus[0], params.a_zero, -1.0, 0)

    # Run the simulation using jax.lax.scan
    _, (spikes, vmem) = jax.lax.scan(step, initial_carry, (stimulus, noise))

    return SIMOutput(spikes.astype(jnp.int32), vmem)


def simulate_without_ref_period(
    key: ArrayLike, stimulus: ArrayLike, params: PUnitParams
) -> SIMOutput:
    """Simulate a P-unit.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params: PUnitParams
        Parameter class

    Returns
    -------
    SIMOutput: class
        Simulation output class which holds both binary spikes and membrane voltage

    """
    noise = jax.random.normal(key, shape=stimulus.shape)
    noise *= params.noise_strength / jnp.sqrt(params.deltat)

    stimulus = jnp.maximum(stimulus, 0.0)

    def step(carry, x):
        v_mem, v_dend, adapt, last_spike_time, time_index = carry
        stim_i, noise_i = x
        current_time = time_index * params.deltat

        # Update dendritic voltage
        v_dend_new = v_dend + (-v_dend + stim_i) / params.dend_tau * params.deltat

        dv_mem = (
            (
                params.v_base
                - v_mem
                + params.v_offset
                + (v_dend_new * params.input_scaling)
                - adapt
                + noise_i
            )
            / params.mem_tau
            * params.deltat
        )

        # Update membrane potential
        v_mem_new = v_mem + dv_mem

        adapt_new = adapt - (adapt / params.tau_a) * params.deltat

        # WARNING: removing the ref period
        # Check for refractory period and reset membrane potential if needed
        # is_refractory = (last_spike_time >= 0) & (
        #     current_time - last_spike_time < params.ref_period + params.deltat / 2
        # )
        # v_mem_ref = jnp.where(is_refractory, params.v_base, v_mem_new)

        # Check for threshold crossing
        spike_fired = v_mem_new > params.threshold

        # Apply spike effects conditionally using jnp.where
        final_v_mem = jnp.where(spike_fired, params.v_base, v_dend_new)
        final_adapt = jnp.where(
            spike_fired, adapt_new + params.delta_a / params.tau_a, adapt_new
        )
        final_last_spike_time = jnp.where(spike_fired, current_time, last_spike_time)

        new_carry = (
            final_v_mem,
            v_dend_new,
            final_adapt,
            final_last_spike_time,
            time_index + 1,
        )
        return new_carry, (spike_fired, final_v_mem)

    # Initial conditions for the scan
    initial_carry = (params.v_zero, stimulus[0], params.a_zero, -1.0, 0)

    # Run the simulation using jax.lax.scan
    _, (spikes, vmem) = jax.lax.scan(step, initial_carry, (stimulus, noise))

    return SIMOutput(spikes.astype(jnp.int32), vmem)
