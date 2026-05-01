import numpy as np


def _wasserstein_2_1d(x, y):
    x = np.sort(np.ravel(np.asarray(x, dtype=float)))
    y = np.sort(np.ravel(np.asarray(y, dtype=float)))
    if x.size == 0 or y.size == 0:
        raise ValueError("Wasserstein samples must be non-empty.")
    if x.size == y.size:
        return float(np.sqrt(np.mean((x - y) ** 2)))

    q = np.unique(
        np.concatenate(
            [np.linspace(0.0, 1.0, x.size + 1), np.linspace(0.0, 1.0, y.size + 1)]
        )
    )
    mid = 0.5 * (q[:-1] + q[1:])
    xq = x[np.minimum((mid * x.size).astype(int), x.size - 1)]
    yq = y[np.minimum((mid * y.size).astype(int), y.size - 1)]
    return float(np.sqrt(np.sum(np.diff(q) * (xq - yq) ** 2)))


def sliced_wasserstein_2(
    x,
    y,
    n_projections=128,
    seed=0,
    return_squared=False,
):
    """Empirical sliced Wasserstein-2 distance between two sample clouds."""
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("`x` and `y` must have shape (n_samples, dim).")
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, x.shape[-1]))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    x_proj = np.sort(x @ directions.T, axis=0)
    y_proj = np.sort(y @ directions.T, axis=0)
    sw2_squared = np.mean((x_proj - y_proj) ** 2)

    if return_squared:
        return float(sw2_squared)
    return float(np.sqrt(sw2_squared))


def trajectory_sliced_wasserstein_2(
    true_paths,
    generated_paths,
    n_projections=128,
    seed=0,
):
    """Compare path distributions at each time step and return per-time and mean error."""
    true_paths = np.asarray(true_paths)
    generated_paths = np.asarray(generated_paths)

    if true_paths.shape != generated_paths.shape:
        raise ValueError("`true_paths` and `generated_paths` must have the same shape.")
    if true_paths.ndim != 3:
        raise ValueError("paths must have shape (n_samples, n_times, dim).")

    per_t = np.asarray(
        [
            sliced_wasserstein_2(
                true_paths[:, i],
                generated_paths[:, i],
                n_projections=n_projections,
                seed=seed + i,
            )
            for i in range(true_paths.shape[1])
        ]
    )
    return per_t, float(per_t.mean())


def mass_wasserstein_2(true_paths, generated_paths):
    """Mass W2 for trajectories shaped (n_samples, n_times, ...).

    Returns (true_mass, generated_mass, per_time_w2, mean_w2).  The mass is
    the mean absolute value over all state-grid entries, matching L = 1.
    """
    true_paths = np.asarray(true_paths)
    generated_paths = np.asarray(generated_paths)

    if true_paths.ndim < 3 or generated_paths.ndim != true_paths.ndim:
        raise ValueError("paths must have shape (n_samples, n_times, ...).")
    if true_paths.shape[1:] != generated_paths.shape[1:]:
        raise ValueError("paths must have the same time and state shape.")

    axes = tuple(range(2, true_paths.ndim))
    true_mass = np.mean(np.abs(true_paths), axis=axes)
    generated_mass = np.mean(np.abs(generated_paths), axis=axes)
    per_t = np.asarray(
        [
            _wasserstein_2_1d(true_mass[:, i], generated_mass[:, i])
            for i in range(true_mass.shape[1])
        ]
    )
    return true_mass, generated_mass, per_t, float(per_t.mean())


def crossing_time_wasserstein_2(
    true_paths,
    generated_paths,
    threshold=0.0,
    spatial_axes=(2, 3),
    boundary_mask=None,
    time_values=None,
    no_crossing_time=1.0,
):
    """W2 between first normalized times that trajectories hit the boundary.

    Returns (true_tau, generated_tau, w2).  A hit occurs when any selected
    boundary entry is greater than `threshold`; non-hits are set to
    `no_crossing_time`.
    """
    true_paths = np.asarray(true_paths)
    generated_paths = np.asarray(generated_paths)

    if true_paths.ndim < 4 or generated_paths.ndim != true_paths.ndim:
        raise ValueError("paths must have shape (n_samples, n_times, nx, ny, ...).")
    if true_paths.shape[1:] != generated_paths.shape[1:]:
        raise ValueError("paths must have the same time and state shape.")

    def selected_boundary(paths):
        axes = tuple(ax if ax >= 0 else paths.ndim + ax for ax in spatial_axes)
        if len(axes) != 2:
            raise ValueError("`spatial_axes` must identify two spatial axes.")
        field = np.moveaxis(paths, axes, (2, 3))
        if boundary_mask is None:
            mask = np.zeros(field.shape[2:4], dtype=bool)
            mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        else:
            mask = np.asarray(boundary_mask, dtype=bool)
            if mask.shape != field.shape[2:4]:
                raise ValueError("`boundary_mask` must match the spatial grid shape.")
        return field[:, :, mask, ...]

    def first_hit_time(paths):
        hit_values = selected_boundary(paths) > threshold
        hit = np.any(hit_values, axis=tuple(range(2, hit_values.ndim)))
        first = np.argmax(hit, axis=1)
        did_hit = np.any(hit, axis=1)
        if time_values is None:
            times = np.linspace(0.0, 1.0, paths.shape[1])
        else:
            times = np.asarray(time_values, dtype=float)
            if times.shape != (paths.shape[1],):
                raise ValueError("`time_values` must have shape (n_times,).")
            times = (times - times[0]) / (times[-1] - times[0])
        return np.where(did_hit, times[first], no_crossing_time)

    true_tau = first_hit_time(true_paths)
    generated_tau = first_hit_time(generated_paths)
    return true_tau, generated_tau, _wasserstein_2_1d(true_tau, generated_tau)
