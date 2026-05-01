import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import numpy as np
from jax import jit, vmap
from jax.numpy.fft import fft2, fftfreq, ifft2


def solve_wave_equation(Tend, dt, N, ic_fn, speed):
    """Solve the 2D wave equation on [0, 2*pi)^2 with a spectral Laplacian."""
    L = 2 * jnp.pi
    dx = L / N
    x = jnp.linspace(0, L - dx, N)
    y = jnp.linspace(0, L - dx, N)
    grid = jnp.meshgrid(x, y, indexing="ij")
    x_pts = jnp.asarray([axis.flatten() for axis in grid]).T

    u0 = ic_fn(x_pts).reshape(N, N)
    c2 = speed**2

    # Spectral wave numbers for the periodic Laplacian.
    kx = fftfreq(N, d=dx) * 2 * jnp.pi
    ky = fftfreq(N, d=dx) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    K_squared = KX**2 + KY**2

    U0_hat = fft2(u0)
    Laplacian_u0_hat = -K_squared * U0_hat
    Laplacian_u0 = ifft2(Laplacian_u0_hat).real

    u1 = u0 + 0.5 * dt**2 * c2 * Laplacian_u0
    Nt = int(Tend / dt)

    def time_step(u_nm1, u_n):
        U_hat = fft2(u_n)
        Laplacian_u_hat = -K_squared * U_hat
        Laplacian_u = ifft2(Laplacian_u_hat).real

        u_np1 = 2 * u_n - u_nm1 + dt**2 * c2 * Laplacian_u
        return u_n, u_np1

    def scan_fn(carry, _):
        u_nm1, u_n = carry
        carry = time_step(u_nm1, u_n)
        return carry, carry[1]

    # u1 is computed explicitly; scan returns u2 through u_Nt.
    num_steps = Nt - 1
    _, tail = jax.lax.scan(scan_fn, (u0, u1), None, length=num_steps)
    sol = jnp.concatenate([u0[None], u1[None], tail], axis=0)
    return sol


def get_wave_random_media(
    n_samples,
    t_pts,
    x_pts,
    key,
    batch_size=32,
    sigma=None,
    Tend=8.0,
    dt=4e-3,
):
    """Generate wave trajectories in random media before spatial subsampling."""
    keys = jax.random.split(key, num=n_samples)
    grid = grids.Grid(
        (x_pts, x_pts),
        domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
    )
    max_velocity = 1
    peak_wavenumber = 4 if sigma is None else sigma

    def get_speed_field(key):
        v0 = cfd.initial_conditions.filtered_velocity_field(
            key, grid, max_velocity, peak_wavenumber
        )
        v0 = v0[0].data + v0[1].data

        # Keep the original [0, 1] min-max scaling for each speed field.
        v0_min = v0.min()
        v0_max = v0.max()
        return (v0 - v0_min) / (v0_max - v0_min)

    s_fields = vmap(get_speed_field)(keys)

    if sigma == 0:
        s_fields = np.ones_like(s_fields) * 0.72

    @vmap
    def ic_fn(x):
        x = x - jnp.pi
        return jnp.exp(-jnp.sum(x**2) * 30)

    @jit
    def solve(s_f):
        sol = solve_wave_equation(Tend, dt, x_pts, ic_fn, s_f)
        t_idx = jnp.linspace(0, len(sol) - 1, t_pts, dtype=jnp.int32)
        return sol[t_idx]

    sols = jax.lax.map(solve, s_fields, batch_size=batch_size)
    sols = np.asarray(sols)
    return sols


def get_wave_data(
    key=jax.random.PRNGKey(1),
    n_samples=1024,
    n_x=256,
    n_t=64,
    sigma=1,
    sub_x=4,
    sub_t=1,
    batch_size=32,
    Tend=8.0,
    dt=4e-3,
):
    """Generate the default wave dataset with channel-last output."""
    sols = get_wave_random_media(
        n_samples,
        n_t,
        n_x,
        key,
        batch_size=batch_size,
        sigma=sigma,
        Tend=Tend,
        dt=dt,
    )
    sols = sols[:, ::sub_t, ::sub_x, ::sub_x, None]
    return sols
