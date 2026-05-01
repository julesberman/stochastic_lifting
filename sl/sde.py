import random

import jax
import jax.numpy as jnp
from jax import jit, vmap


def euler_maruyama(drift, diffusion, y0, t_eval, key, dt=None):
    """Simulate one SDE path at the requested times."""
    t_eval = jnp.asarray(t_eval)
    y0 = jnp.asarray(y0)
    step_size = None if dt is None else jnp.asarray(dt, dtype=t_eval.dtype)

    def em_step(y, t, h, key):
        key, subkey = jax.random.split(key)
        # dW has variance h, so sample it as sqrt(h) * N(0, I).
        dW = jnp.sqrt(h) * jax.random.normal(subkey, shape=y.shape, dtype=y.dtype)
        y = y + drift(t, y) * h + diffusion(t, y) * dW
        return y, key

    def step(carry, t_next):
        y, t_prev, key = carry

        if step_size is None:
            y, key = em_step(y, t_prev, t_next - t_prev, key)
            return (y, t_next, key), y

        # Take fixed substeps, then one short final step to hit t_next exactly.
        def cond_fn(state):
            _, t, _ = state
            return t < t_next

        def body_fn(state):
            y, t, key = state
            h = jnp.minimum(step_size, t_next - t)
            y, key = em_step(y, t, h, key)
            return y, t + h, key

        y, _, key = jax.lax.while_loop(cond_fn, body_fn, (y, t_prev, key))
        return (y, t_next, key), y

    _, ys = jax.lax.scan(step, (y0, t_eval[0], key), t_eval[1:])
    return jnp.concatenate([y0[None], ys], axis=0)


def solve_sde(drift, diffusion, t_eval, get_ic, n_samples, dt=1e-2, key=None):
    """Simulate many independent SDE paths."""
    t_eval = jnp.asarray(t_eval)

    @jit
    def solve_single(key):
        ic_key, solve_key = jax.random.split(key)
        y0 = get_ic(ic_key)
        return euler_maruyama(drift, diffusion, y0, t_eval, solve_key, dt=dt)

    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 1_000_000))

    keys = jax.random.split(key, num=n_samples)
    return vmap(solve_single)(keys)


def solve_sde_ic(y0, key, t_eval, dt, drift, diffusion):
    """Simulate one SDE path from a fixed initial condition."""
    return euler_maruyama(drift, diffusion, y0, t_eval, key, dt=dt)
