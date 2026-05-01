import jax
import numpy as np
import optax
from jax import jit
from tqdm.auto import tqdm

str_to_opt = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'amsgrad': optax.amsgrad,
    'adabelief': optax.adabelief,
}


def _as_args(args):
    if args is None:
        return ()
    if isinstance(args, tuple):
        return args
    if isinstance(args, list):
        return tuple(args)
    return (args,)


def optimize(
    theta_init,
    loss_fn,
    args_fn,
    init_state=None,
    steps=1000,
    learning_rate=5e-4,
    scheduler=True,
    verbose=False,
    loss_tol=None,
    optimizer='adam',
    key=None,
):
    if optimizer not in str_to_opt:
        raise ValueError(f'unknown optimizer: {optimizer}')
    if steps <= 0:
        return theta_init, np.asarray([])

    # decay the learning rate over the requested run.
    if scheduler:
        learning_rate = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=steps,
            alpha=0.0,
        )

    opt = str_to_opt[optimizer](learning_rate=learning_rate)
    state = init_state if init_state is not None else opt.init(theta_init)

    @jit
    def step(params, state, args):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, *args)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return loss_value, params, state

    params = theta_init
    pbar = tqdm(range(steps), disable=not verbose)
    loss_history = []
    n_rec = max(steps // 1000, 1)

    for i in pbar:
        if callable(args_fn):
            if key is not None:
                key, skey = jax.random.split(key)
                args = _as_args(args_fn(skey))
            else:
                args = _as_args(args_fn())
        else:
            args = _as_args(args_fn)

        cur_loss, params_new, state_new = step(params, state, args)
        cur_loss = float(cur_loss)

        if verbose:
            pbar.set_postfix({'loss': f'{cur_loss:.3E}'})

        # keep history bounded for long runs.
        if i % n_rec == 0:
            loss_history.append(cur_loss)

        params = params_new
        state = state_new

        if loss_tol is not None and cur_loss < loss_tol:
            break

    loss_history = np.asarray(loss_history)
    return params, loss_history
