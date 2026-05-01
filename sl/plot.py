import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import imageio  # pip install "imageio[ffmpeg]"
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from einops import rearrange
from IPython.display import HTML
from matplotlib import animation, colors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import seaborn


def imshow_movie(
    sol: np.ndarray,
    *,
    frames: Optional[int] = 50,
    t: Optional[Sequence[float]] = None,
    interval: int = 100,
    title: str = "",
    cmap: str = "viridis",
    aspect: str = "equal",
    interpolation: str = "nearest",
    figsize: Optional[Tuple[float, float]] = None,
    show_colorbar: bool = True,
    live_cbar: bool = True,
    tight: bool = True,
    c_norm: Optional[Tuple[float, float]] = None,
    t_txt: bool = True,
    gif_hq: bool = False,
    #
    label: Optional[Union[str, Sequence[str]]] = None,      # ← NEW
    label_font_size: int = 10,                               # ← NEW
    label_color: str = "white",                              # ← NEW
    #
    save_to: Optional[Union[str, Path]] = None,
    save_format: Literal["gif", "mp4"] = "gif",
    fps: int = 10,
    dpi: Optional[int] = None,
    #
    show_inline: bool = True,
) -> Optional[HTML]:
    """
    Animate a stack of 2-D images (T × H × W).

    Parameters
    ----------
    sol : ndarray
        Array of frames ordered in time, shape (T, H, W).
    frames : int | None
        Number of frames to *display* (sub-samples if needed, None ⇒ all).
    t : sequence[float] | None
        Time-stamps for each frame (defaults to np.arange(T)).
    interval : int
        Delay between frames in ms for the JS/HTML player.
    figsize : (w, h) in inches | None
        Figure size passed to `plt.subplots`; None uses Matplotlib default.
    show_colorbar : bool
        Draw a colour-bar (independent of live updates).
    live_cbar : bool
        If True, rescale colour limits every frame.
    tight : bool
        Remove all extra padding/margins around the axes.
    c_norm : (vmin, vmax) | None
        Fixed colour limits (overrides data-driven limits).
    t_txt : bool
        Show time/frame information in the axes title.
    label : str | sequence[str] | None
        Optional label(s) rendered at the bottom-right of the axes.
        - If a single string (or a non-list scalar), it is shown for every frame.
        - If a sequence, it must have length T (the original number of frames),
          and the label shown is sub-sampled consistently with `frames`.
    label_font_size : int
        Font size for the label text.
    label_color : str
        Color for the label text.
    save_to : str | Path | None
        Persist animation to disk if not None (extension added).
    save_format : {"gif", "mp4"}
        Container used when saving.
    fps : int
        Frames-per-second for the saved video.
    dpi : int | None
        DPI for MP4 writer; ignored for GIF.
    show_inline : bool
        Return an IPython HTML widget (JS) for inline display.

    Returns
    -------
    IPython.display.HTML | None
        Inline HTML widget if `show_inline` else None.
    """
    sol = np.asarray(sol)
    T = sol.shape[0]

    # Validate/build time vector
    if t is None:
        t = np.arange(T)
    else:
        t = np.asarray(t)
        if t.shape[0] != T:
            raise ValueError("`t` must have the same length as sol.shape[0]")

    # Validate/build label vector (over original T)
    label_is_seq = False
    labels_full: Optional[np.ndarray] = None
    label_static: Optional[str] = None
    if label is not None:
        # Treat plain strings as a scalar label (not a sequence of chars)
        if isinstance(label, str):
            label_static = label
        else:
            # Consider it a sequence of per-frame labels
            labels_full = np.asarray(list(label), dtype=object)
            label_is_seq = True
            if labels_full.shape[0] != T:
                raise ValueError(
                    "If `label` is a sequence, it must have length T (= sol.shape[0]).")

    # Frame sub-sampling
    step = 1 if frames is None else max(T // frames, 1)
    sol_frames, t_frames = sol[::step], t[::step]
    labels_frames: Optional[np.ndarray] = None
    if label_is_seq and labels_full is not None:
        labels_frames = labels_full[::step]

    # Figure & axes
    fig, ax = plt.subplots(figsize=figsize)
    cax = None
    if show_colorbar:
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad="3%")

    norm = (
        colors.Normalize(*c_norm)
        if c_norm is not None
        else colors.Normalize(vmin=float(sol.min()), vmax=float(sol.max()))
    )
    im = ax.imshow(
        sol_frames[0],
        cmap=cmap,
        aspect=aspect,
        interpolation=interpolation,
        norm=norm,
    )
    if show_colorbar:
        _ = fig.colorbar(im, cax=cax)  # noqa: F841

    ax.set_xticks([])
    ax.set_yticks([])
    tx = ax.set_title("") if t_txt else None

    # Label artist (bottom-right in axes coords)
    lbl_artist = None
    if label is not None:
        initial_label = label_static if label_static is not None else (
            labels_frames[0] if labels_frames is not None else "")
        lbl_artist = ax.text(
            0.99,
            0.01,
            str(initial_label),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=label_font_size,
            color=label_color,
        )

    # Remove padding if tight=True
    if tight:
        ax.margins(0)
        fig.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Animation callback
    def _animate(idx: int):
        arr = sol_frames[idx]
        im.set_data(arr)
        if live_cbar and show_colorbar:
            im.set_clim(float(arr.min()), float(arr.max()))
        if t_txt:
            tx.set_text(f"{title} t={t_frames[idx]:.3g} (frame {idx})")
        if lbl_artist is not None:
            if label_static is not None:
                lbl_artist.set_text(str(label_static))
            elif labels_frames is not None:
                lbl_artist.set_text(str(labels_frames[idx]))

    ani = animation.FuncAnimation(
        fig,
        _animate,
        frames=len(sol_frames),
        interval=interval,
        blit=False,
    )

    # Save to disk
    if save_to is not None:
        path = Path(save_to).with_suffix(f".{save_format}")
        if save_format == "gif":
            if gif_hq:
                _save_gif_high_quality(ani, path, fps=fps, dpi=dpi or 100)
            else:
                writer = animation.PillowWriter(fps=fps)
                ani.save(path, writer=writer)
        else:  # mp4
            writer = animation.FFMpegWriter(fps=fps, codec="libx264")
            ani.save(path, writer=writer, dpi=dpi)
        print(f"animation saved → {path.resolve()}")

    plt.close(fig)
    return HTML(ani.to_jshtml()) if show_inline else None


def _save_gif_high_quality(ani, path: Path, fps: int, dpi: int | None):
    """Write *path* as a palette-optimised GIF via ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_mp4 = Path(tmp, "tmp.mp4")
        palette = Path(tmp, "palette.png")

        # 1) MP4 (true colour, no palette limits)
        writer = animation.FFMpegWriter(fps=fps, codec="libx264")
        ani.save(tmp_mp4, writer=writer, dpi=dpi)

        # 2) Generate optimal palette
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-i", tmp_mp4, "-vf", "palettegen=stats_mode=diff", palette],
            check=True,
        )
        # 3) Apply palette with Floyd–Steinberg dithering
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-i", tmp_mp4, "-i", palette,
             "-lavfi", "paletteuse=dither=floyd_steinberg",
             "-loop", "0", path],
            check=True,
        )


def scatter_movie(
    pts,
    c="r",
    n_samples=None,
    size=None,
    xlim=None,
    ylim=None,
    alpha=1,
    frames=60,
    t=None,
    title="",
    interval=100,
    save_to=None,
    show=True,
    fps=10,
    no_title=False,
    stroke_color=None,
    stroke_width=0.5,
    figsize=None,
    xticks=None,
    yticks=None,
    grid=False,
    grid_kwargs=None,
):
    pts = np.asarray(pts)

    if len(pts.shape) == 4:
        g, _, n, _ = pts.shape
        colors = ["r", "b", "g", "m", "k"]
        c_list = []
        for i in range(g):
            c_list.extend([colors[i]] * n)
        c = c_list
        pts = rearrange(pts, "g t n d -> t (g n) d")

    pts = rearrange(pts, "t n d -> t d n")

    if n_samples is not None:
        sample_idx = np.random.choice(
            pts.shape[-1] - 1, size=n_samples, replace=False)
        sample_idx = np.asarray(sample_idx, dtype=np.int32)
        pts = pts[:, :, sample_idx]
        if isinstance(c, (list, tuple, np.ndarray)) and not isinstance(c, str):
            c = np.asarray(c)[sample_idx]

    pts_full = pts
    time = len(pts_full)
    if t is None:
        t = np.arange(time)

    inc = max(time // frames, 1)
    t_frames = t[::inc]
    pts_anim = pts_full[::inc]

    fig, ax = plt.subplots(figsize=figsize)

    # Ensure grid is drawn beneath artists (including scatter points)
    ax.set_axisbelow(True)

    mm = pts_full.min(axis=(0, 2))
    mx = pts_full.max(axis=(0, 2))

    if xlim is None:
        xlim = [mm[0], mx[0]]
    if ylim is None:
        ylim = [mm[1], mx[1]]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    if xticks is False:
        ax.set_xticks([])
        ax.set_xticklabels([])
    elif xticks is not None:
        ax.set_xticks(xticks)

    if yticks is False:
        ax.set_yticks([])
        ax.set_yticklabels([])
    elif yticks is not None:
        ax.set_yticks(yticks)

    if grid:
        ax.grid(True, **(grid_kwargs or {}))
    else:
        ax.grid(False)

    scatter_kwargs = dict(alpha=alpha, s=size, c=c)
    if stroke_color is not None:
        scatter_kwargs.update(edgecolors=stroke_color, linewidths=stroke_width)

    sct = ax.scatter(x=pts_anim[0, 0], y=pts_anim[0, 1], **scatter_kwargs)

    tx = None
    if not no_title:
        tx = ax.set_title(f"{title} t={t_frames[0]:.2f}")

    def animate(frame):
        scatter, tt = frame
        sct.set_offsets(scatter.T)
        if tx is not None:
            tx.set_text(f"{title} t={tt:.2f}")

    frames_iter = list(zip(pts_anim, t_frames, strict=False))
    ani = FuncAnimation(fig, animate, frames=frames_iter, interval=interval)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def scatter_movie_grid(
    pts,
    t=None,
    titles_x=None,
    titles_y=None,
    suptitle=None,
    suptitle_y=None,
    grid_height=None,
    grid_width=None,
    fig_size=(8, 8),
    xticks_on=False,
    yticks_on=False,
    xticks=None,
    yticks=None,
    xlabel=None,
    ylabel=None,
    space=0.1,
    interval=200,
    save_to=None,
    show=True,
    seconds=5,
    writer="pillow",
    frames=64,
    c="r",
    n_samples=None,
    n_traj=None,
    size=None,
    xlim=None,
    ylim=None,
    alpha=1,
    title="",
    no_title=False,
    stroke_color=None,
    stroke_width=0.5,
    grid=False,
    grid_kwargs=None,
    plot_trajectories=False,
    trajectory_length=None,
    trajectory_fade=False,
    trajectory_alpha=None,
    trajectory_width=1.0,
):
    """
    Animate a grid of scatter plots.

    Parameters
    ----------
    pts : array-like, shape ``(N, T, P, 2)`` or ``(N, G, T, P, 2)``
        Stack of *N* scatter movies. Grouped input uses one colour per group,
        matching the grouped behaviour of ``scatter_movie``.
    n_samples : int, 1-D array, or None
        Number of particles to display. A scalar samples one fixed subset for
        all frames. A 1-D array can control the displayed count per panel, per
        group, or per animation frame.
    n_traj : int | None
        Number of particle trajectories to draw, sampled randomly from the
        displayed points. ``None`` draws traces for all displayed points.
    plot_trajectories : bool
        If True, draw a short line trace behind each particle.
    trajectory_length : int | None
        Number of sampled frames to keep in each trace. ``None`` uses the full
        history available in the animation.
    trajectory_fade : bool
        If True, older parts of each trace are rendered with lower alpha.
    trajectory_alpha : float | None
        Opacity for trajectory traces. Defaults to ``alpha`` if omitted.
    trajectory_width : float
        Line width for trajectory traces.
    The remaining grid-related arguments mirror ``plot_grid_movie`` while the
    scatter-specific arguments mirror ``scatter_movie``.
    """
    pts = np.asarray(pts)

    if pts.ndim not in (4, 5):
        raise ValueError(
            "`pts` must have shape (N, T, P, 2) or (N, G, T, P, 2)."
        )

    if pts.shape[-1] != 2:
        raise ValueError(
            "`pts` must have shape (N, T, P, 2) or (N, G, T, P, 2)."
        )

    grouped = pts.ndim == 5
    time = pts.shape[2] if grouped else pts.shape[1]
    if t is None:
        t = np.arange(time)
    else:
        t = np.asarray(t)
        if t.shape[0] != time:
            raise ValueError(
                "`t` must have length equal to the time axis of `pts`."
            )

    if frames is None:
        t_idx = np.arange(time, dtype=np.int32)
    else:
        n_frames = min(frames, time)
        t_idx = np.linspace(0, time - 1, n_frames, dtype=np.int32)
    t_frames = t[t_idx]
    n_frames = len(t_idx)

    n_samples_counts = None
    n_samples_group_counts = None
    n_samples_panel_counts = None
    n_samples_scalar = n_samples
    if n_samples is not None:
        n_samples_arr = np.asarray(n_samples)
        if n_samples_arr.ndim == 0:
            if not np.equal(n_samples_arr, np.floor(n_samples_arr)):
                raise ValueError("`n_samples` must be an integer count.")
            n_samples_scalar = int(n_samples_arr)
            if n_samples_scalar < 0:
                raise ValueError("`n_samples` must be non-negative.")
        else:
            if n_samples_arr.ndim != 1:
                raise ValueError("`n_samples` must be a scalar or 1-D array.")
            if n_samples_arr.shape[0] == pts.shape[0]:
                n_samples_panel_counts = n_samples_arr
            elif grouped and n_samples_arr.shape[0] == pts.shape[1]:
                n_samples_group_counts = n_samples_arr
            elif n_samples_arr.shape[0] == time:
                n_samples_counts = n_samples_arr[t_idx]
            elif n_samples_arr.shape[0] == n_frames:
                n_samples_counts = n_samples_arr
            else:
                raise ValueError(
                    "`n_samples` array must have length equal to the number of "
                    "groups, panels, original time steps, or displayed frames."
                )
            n_samples_active = (
                n_samples_counts
                if n_samples_counts is not None
                else n_samples_group_counts
                if n_samples_group_counts is not None
                else n_samples_panel_counts
            )
            if not np.all(np.equal(n_samples_active, np.floor(n_samples_active))):
                raise ValueError(
                    "`n_samples` array must contain integer counts.")
            n_samples_active = n_samples_active.astype(np.int32)
            if np.any(n_samples_active < 0):
                raise ValueError("`n_samples` array must be non-negative.")
            if n_samples_active.max(initial=0) < 1:
                raise ValueError(
                    "`n_samples` array must show at least one point.")
            if n_samples_counts is not None:
                n_samples_counts = n_samples_active
            elif n_samples_group_counts is not None:
                n_samples_group_counts = n_samples_active
            else:
                n_samples_panel_counts = n_samples_active
            n_samples_scalar = None

    color_data = c
    point_visible_mask = None
    point_visible_mask_kind = None
    if pts.ndim == 5:
        n_panels, n_groups, _, n_points_per_group, _ = pts.shape
        n_points_per_group_total = n_points_per_group
        sample_idx = None
        if n_samples_group_counts is not None:
            max_samples = int(n_samples_group_counts.max())
            if max_samples > n_points_per_group:
                raise ValueError(
                    "`n_samples` cannot exceed the number of points per group."
                )
            sample_idx = np.stack(
                [
                    np.random.choice(
                        n_points_per_group, size=max_samples, replace=False
                    )
                    for _ in range(n_groups)
                ]
            ).astype(np.int32)
            pts = np.take_along_axis(
                pts,
                sample_idx[None, :, None, :, None],
                axis=3,
            )
            n_points_per_group = max_samples
            point_visible_mask = np.concatenate(
                [
                    np.arange(n_points_per_group) < group_count
                    for group_count in n_samples_group_counts
                ]
            )
            point_visible_mask_kind = "point"
        elif n_samples_panel_counts is not None:
            max_samples = int(n_samples_panel_counts.max())
            if max_samples > n_points_per_group:
                raise ValueError(
                    "`n_samples` cannot exceed the number of points per group."
                )
            sample_idx = np.stack(
                [
                    np.random.choice(
                        n_points_per_group, size=max_samples, replace=False
                    )
                    for _ in range(n_panels)
                ]
            ).astype(np.int32)
            pts = np.take_along_axis(
                pts,
                sample_idx[:, None, None, :, None],
                axis=3,
            )
            n_points_per_group = max_samples
            point_visible_mask = (
                np.arange(n_points_per_group)[None, None, :]
                < n_samples_panel_counts[:, None, None]
            )
            point_visible_mask = np.broadcast_to(
                point_visible_mask,
                (n_panels, n_groups, n_points_per_group),
            ).reshape(n_panels, n_groups * n_points_per_group)
            point_visible_mask_kind = "panel"
        elif n_samples_counts is not None:
            max_samples = int(n_samples_counts.max())
            if max_samples > n_points_per_group:
                raise ValueError(
                    "`n_samples` cannot exceed the number of points per group."
                )
            sample_idx = np.random.choice(
                n_points_per_group, size=max_samples, replace=False
            )
            sample_idx = np.asarray(sample_idx, dtype=np.int32)
            pts = pts[:, :, :, sample_idx]
            n_points_per_group = max_samples
            point_visible_mask = np.concatenate(
                [
                    np.arange(n_points_per_group)[None, :]
                    < n_samples_counts[:, None]
                    for _ in range(n_groups)
                ],
                axis=1,
            )
            point_visible_mask_kind = "frame"
        elif n_samples_scalar is not None:
            if n_samples_scalar > n_points_per_group:
                raise ValueError(
                    "`n_samples` cannot exceed the number of points per group."
                )
            sample_idx = np.random.choice(
                n_points_per_group, size=n_samples_scalar, replace=False
            )
            sample_idx = np.asarray(sample_idx, dtype=np.int32)
            pts = pts[:, :, :, sample_idx]
            n_points_per_group = n_samples_scalar

        if isinstance(c, str):
            colors_default = ["r", "b", "g", "m", "k"]
            color_data = [
                colors_default[i % len(colors_default)]
                for i in range(n_groups)
                for _ in range(n_points_per_group)
            ]
        elif (
            isinstance(c, (list, tuple, np.ndarray))
            and not mcolors.is_color_like(c)
        ):
            color_arr = np.asarray(c)
            if color_arr.shape[0] == n_groups:
                color_data = [
                    color_arr[i]
                    for i in range(n_groups)
                    for _ in range(n_points_per_group)
                ]
            elif (
                sample_idx is not None
                and color_arr.shape[0] == n_groups * n_points_per_group_total
            ):
                if sample_idx.ndim == 1:
                    color_data = color_arr.reshape(
                        n_groups, n_points_per_group_total, *
                        color_arr.shape[1:]
                    )[:, sample_idx].reshape(
                        n_groups * n_points_per_group, *color_arr.shape[1:]
                    )
        pts = rearrange(pts, "n g t p d -> n t (g p) d")
    else:
        n_panels = pts.shape[0]

    if grid_width is not None and grid_height is not None:
        n_keep = grid_height * grid_width
        pts = pts[:n_keep]
        n_panels = pts.shape[0]

    pts = rearrange(pts, "n t p d -> n t d p")

    if n_samples_counts is not None and not grouped:
        max_samples = int(n_samples_counts.max())
        if max_samples > pts.shape[-1]:
            raise ValueError("`n_samples` cannot exceed the number of points.")
        sample_idx = np.random.choice(
            pts.shape[-1], size=max_samples, replace=False
        )
        sample_idx = np.asarray(sample_idx, dtype=np.int32)
        pts = pts[:, :, :, sample_idx]
        point_visible_mask = (
            np.arange(max_samples)[None, :] < n_samples_counts[:, None]
        )
        point_visible_mask_kind = "frame"
        if isinstance(color_data, (list, tuple, np.ndarray)) and not isinstance(
            color_data, str
        ):
            color_data = np.asarray(color_data)[sample_idx]
    elif n_samples_scalar is not None and not grouped:
        if n_samples_scalar > pts.shape[-1]:
            raise ValueError("`n_samples` cannot exceed the number of points.")
        sample_idx = np.random.choice(
            pts.shape[-1], size=n_samples_scalar, replace=False
        )
        sample_idx = np.asarray(sample_idx, dtype=np.int32)
        pts = pts[:, :, :, sample_idx]
        if isinstance(color_data, (list, tuple, np.ndarray)) and not isinstance(
            color_data, str
        ):
            color_data = np.asarray(color_data)[sample_idx]

    pts = pts[:, t_idx]
    _, n_frames = pts.shape[:2]

    if trajectory_length is not None and trajectory_length < 1:
        raise ValueError("`trajectory_length` must be at least 1.")
    if n_traj is not None:
        if n_traj < 1:
            raise ValueError("`n_traj` must be at least 1.")

    if grid_height is None and grid_width is None:
        grid_width = int(np.ceil(np.sqrt(n_panels)))
        grid_height = int(np.ceil(n_panels / grid_width))
    elif grid_height is None:
        grid_height = int(np.ceil(n_panels / grid_width))
    elif grid_width is None:
        grid_width = int(np.ceil(n_panels / grid_height))

    mm = pts.min(axis=(0, 1, 3))
    mx = pts.max(axis=(0, 1, 3))

    def _resolve_limits(lims, default_min, default_max):
        if lims is None:
            return np.tile(np.array([default_min, default_max]), (n_panels, 1))
        arr = np.asarray(lims)
        if arr.shape == (2,):
            return np.tile(arr, (n_panels, 1))
        if arr.shape == (n_panels, 2):
            return arr
        raise ValueError(
            "Axis limits must be None, a single pair, or one pair per panel."
        )

    xlims = _resolve_limits(xlim, mm[0], mx[0])
    ylims = _resolve_limits(ylim, mm[1], mx[1])

    fig = plt.figure(figsize=fig_size)
    if suptitle is not None:
        fig.suptitle(suptitle, y=suptitle_y)

    grid_axes = ImageGrid(
        fig,
        111,
        nrows_ncols=(grid_height, grid_width),
        axes_pad=space,
        share_all=False,
        aspect=True,
    )

    scatter_kwargs = dict(alpha=alpha, s=size)
    if mcolors.is_color_like(color_data) and not isinstance(color_data, str):
        scatter_kwargs["color"] = color_data
    else:
        scatter_kwargs["c"] = color_data
    if stroke_color is not None:
        scatter_kwargs.update(edgecolors=stroke_color, linewidths=stroke_width)

    scatters = []
    trace_collections = []
    title_artists = []

    n_points = pts.shape[-1]
    if n_traj is not None:
        n_traj = min(n_traj, n_points)
    if trajectory_alpha is None:
        trajectory_alpha = alpha

    traj_idx = None
    if plot_trajectories:
        if n_traj is None:
            traj_idx = np.arange(n_points, dtype=np.int32)
        else:
            traj_idx = np.random.choice(n_points, size=n_traj, replace=False)
            traj_idx = np.asarray(traj_idx, dtype=np.int32)

    def _resolve_trace_colors(color_spec):
        if isinstance(color_spec, str):
            return np.repeat(
                np.asarray(mcolors.to_rgba(color_spec))[
                    None, :], n_points, axis=0
            )

        color_arr = np.asarray(color_spec)
        if color_arr.ndim == 1 and color_arr.shape[0] == n_points:
            try:
                return mcolors.to_rgba_array(color_arr)
            except (ValueError, TypeError):
                norm = mcolors.Normalize(
                    vmin=float(np.min(color_arr)), vmax=float(np.max(color_arr))
                )
                cmap = plt.get_cmap("viridis")
                return cmap(norm(color_arr))

        rgba = mcolors.to_rgba_array(color_spec)
        if rgba.shape[0] == 1:
            return np.repeat(rgba, n_points, axis=0)
        if rgba.shape[0] != n_points:
            raise ValueError(
                "Trajectory colors must resolve to one color or one color per point."
            )
        return rgba

    trace_colors = _resolve_trace_colors(
        color_data) if plot_trajectories else None
    if trace_colors is not None and traj_idx is not None:
        trace_colors = trace_colors[traj_idx]

    def _visible_mask(frame_idx, panel_idx):
        if point_visible_mask is None:
            return None
        if point_visible_mask_kind == "frame":
            return point_visible_mask[frame_idx]
        if point_visible_mask_kind == "panel":
            return point_visible_mask[panel_idx]
        return point_visible_mask

    def _visible_points(panel_pts, frame_idx, panel_idx):
        cur = panel_pts[frame_idx].copy()
        visible_mask = _visible_mask(frame_idx, panel_idx)
        if visible_mask is not None:
            cur[:, ~visible_mask] = np.nan
        return cur

    def _build_trace_segments(panel_pts, frame_idx, panel_idx):
        start_idx = 0
        if trajectory_length is not None:
            start_idx = max(frame_idx - trajectory_length + 1, 0)

        history = np.transpose(panel_pts[start_idx:frame_idx + 1], (2, 0, 1))
        history = history[traj_idx]
        if point_visible_mask is None:
            visible_history = np.ones(history.shape[:2], dtype=bool)
        elif point_visible_mask_kind == "frame":
            visible_history = point_visible_mask[
                start_idx:frame_idx + 1, traj_idx
            ].T
        else:
            visible_mask = _visible_mask(frame_idx, panel_idx)[traj_idx]
            visible_history = np.repeat(
                visible_mask[:, None], history.shape[1], axis=1
            )
        if history.shape[1] < 2:
            return [], None

        segments = []
        segment_colors = []
        tail_len = history.shape[1] - 1

        for point_idx, (point_history, point_visible) in enumerate(
            zip(history, visible_history, strict=False)
        ):
            point_segments = np.stack(
                [point_history[:-1], point_history[1:]],
                axis=1,
            )
            visible_segments = point_visible[:-1] & point_visible[1:]
            point_segments = point_segments[visible_segments]
            segments.extend(point_segments)

            if trajectory_fade:
                fade = np.linspace(0.15, 1.0, tail_len)
                point_colors = np.repeat(
                    trace_colors[point_idx][None, :], tail_len, axis=0
                )
                point_colors[:, 3] = trajectory_alpha * fade
                point_colors = point_colors[visible_segments]
                segment_colors.extend(point_colors)

        if trajectory_fade:
            return segments, np.asarray(segment_colors)

        return segments, None

    for i, ax in enumerate(grid_axes):
        if i >= n_panels:
            ax.set_visible(False)
            continue

        ax.set_axisbelow(True)
        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])

        if not xticks_on:
            ax.set_xticks([])
        else:
            ax.set_xticks(
                xticks if xticks is not None
                else np.linspace(xlims[i][0], xlims[i][1], 3)
            )
        if not yticks_on:
            ax.set_yticks([])
        else:
            ax.set_yticks(
                yticks if yticks is not None
                else np.linspace(ylims[i][0], ylims[i][1], 3)
            )

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if grid:
            ax.grid(True, **(grid_kwargs or {}))
        else:
            ax.grid(False)

        initial_pts = _visible_points(pts[i], 0, i)
        scatter = ax.scatter(
            x=initial_pts[0], y=initial_pts[1], **scatter_kwargs)
        scatters.append(scatter)

        if plot_trajectories:
            trace_collection = LineCollection(
                [],
                colors=trace_colors,
                linewidths=trajectory_width,
                alpha=trajectory_alpha,
                zorder=scatter.get_zorder() - 1,
            )
            ax.add_collection(trace_collection)
            trace_collections.append(trace_collection)
        else:
            trace_collections.append(None)

        if titles_x is not None and i < grid_width:
            ax.set_title(titles_x[i])

        if titles_y is not None and i % grid_width == 0:
            row = i // grid_width
            if row < len(titles_y):
                ax.set_ylabel(titles_y[row])

        if not no_title:
            title_artists.append(
                ax.text(
                    0.5,
                    1.02,
                    f"{title} t={t_frames[0]:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                )
            )
        else:
            title_artists.append(None)

    def update(frame_idx):
        for i, scatter in enumerate(scatters):
            cur = _visible_points(pts[i], frame_idx, i)
            scatter.set_offsets(cur.T)

            trace_collection = trace_collections[i]
            if trace_collection is not None:
                segments, segment_colors = _build_trace_segments(
                    pts[i], frame_idx, i)
                trace_collection.set_segments(segments)
                if trajectory_fade and segment_colors is not None:
                    trace_collection.set_color(segment_colors)
                else:
                    trace_collection.set_color(trace_colors)

            title_artist = title_artists[i]
            if title_artist is not None:
                title_artist.set_text(f"{title} t={t_frames[frame_idx]:.2f}")
        return scatters + [tc for tc in trace_collections if tc is not None]

    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=interval, blit=False)

    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        fps = max(n_frames // seconds, 1)
        ani.save(p, writer=writer, fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def trajectory_movie(
    y,
    frames=50,
    title="",
    ylabel="",
    xlabel="Time",
    legend=[],
    x=None,
    interval=100,
    ylim=None,
    save_to=None,
):

    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))

    fig, ax = plt.subplots()
    total = len(x)
    inc = max(total // frames, 1)
    x = x[::inc]
    y = y[::inc]
    if ylim is None:
        ylim = np.array([y.min(), y.max()])
    xlim = [x.min(), x.max()]

    def animate(i):
        ax.cla()
        ax.plot(x[:i], y[:i], marker="o", markevery=[-1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(legend, loc="lower right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} t={x[i]:.2f}")

    ani = FuncAnimation(fig, animate, frames=len(x), interval=interval)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=30)

    return HTML(ani.to_jshtml())


def plot_grid(
    A,
    *,
    fig=None,                 # NEW: existing Figure
    colorbar=True,
    colorbar_mode="single",
    grid_height=None,
    grid_width=None,
    fig_size=(8, 8),
    cmap="viridis",
    xticks_on=False,
    yticks_on=False,
    aspect="auto",
    space=0.1,
    save_to=None,
    titles_x=None,
    titles_y=None,
    c_norm=None,
    to_int=False,
    cbar_ticks=None,
    cbar_tick_fmt=None,
    imagegrid_kwargs=None,
    imshow_kwargs=None,
    show=True,
    title_size=None,                # font size for titles_x and titles_y
    x_titles_position="top",        # "top" or "bottom"
    y_titles_position="left",       # "left" or "right"
):
    """
    Display *N* images in a tidy rectangular grid.

    (Docstring omitted here for brevity; add entries for the three new params.)
    """
    if to_int:
        A = np.clip(A, a_min=0, a_max=255)
        A = np.asarray(A, dtype=np.uint16)

    # ---- resolve grid shape -------------------------------------------------
    N = A.shape[0]
    if grid_height is None and grid_width is None:
        grid_width = int(np.ceil(np.sqrt(N)))
        grid_height = int(np.ceil(N / grid_width))
    elif grid_height is None:
        grid_height = int(np.ceil(N / grid_width))
    elif grid_width is None:
        grid_width = int(np.ceil(N / grid_height))

    # ---- validate titles ----------------------------------------------------
    if titles_x is not None and len(titles_x) != grid_width:
        raise ValueError(
            f"Expected {grid_width} column titles, got {len(titles_x)}.")
    if titles_y is not None and len(titles_y) != grid_height:
        raise ValueError(
            f"Expected {grid_height} row titles, got {len(titles_y)}.")

    # ---- validate new options ----------------------------------------------
    if x_titles_position not in {"top", "bottom"}:
        raise ValueError("x_titles_position must be 'top' or 'bottom'.")
    if y_titles_position not in {"left", "right"}:
        raise ValueError("y_titles_position must be 'left' or 'right'.")

    # ---- prepare extra kwargs ----------------------------------------------
    imagegrid_kwargs = imagegrid_kwargs or {}
    imshow_kwargs = imshow_kwargs or {}

    # ---- create or reuse Figure ---------------------------------------------
    if fig is None:
        fig = plt.figure(figsize=fig_size)

    # ---- create ImageGrid ---------------------------------------------------
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(grid_height, grid_width),
        axes_pad=space,
        share_all=True,
        cbar_mode=(colorbar_mode if colorbar else None),
        aspect=aspect,
        **imagegrid_kwargs,
    )

    # ---- plot each image ----------------------------------------------------
    for i in range(N):
        ax = grid[i]

        norm = (colors.Normalize(vmin=c_norm[0], vmax=c_norm[1])
                if c_norm is not None
                else colors.Normalize(vmin=np.min(A[i]), vmax=np.max(A[i])))

        im = ax.imshow(
            A[i],
            cmap=cmap,
            aspect="auto",
            norm=norm,
            **imshow_kwargs,
        )

        if not xticks_on:
            ax.set_xticks([])
        if not yticks_on:
            ax.set_yticks([])

        if colorbar and colorbar_mode == "each":
            cax = ax.cax
            cax.colorbar(im)
            cax.tick_params(labelleft=True)

    # ---- shared colorbar ----------------------------------------------------
    if colorbar and colorbar_mode == "single":
        cax = grid.cbar_axes[0]
        if cbar_ticks is not None:
            cbar = cax.colorbar(im, ticks=cbar_ticks)
        else:
            cbar = cax.colorbar(im)
        cax.tick_params()
        if cbar_tick_fmt is not None:
            if type(cbar_tick_fmt) != "str":
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(cbar_tick_fmt))
            else:
                cbar.ax.yaxis.set_major_formatter(
                    FormatStrFormatter(cbar_tick_fmt))

    # ---- column titles ------------------------------------------------------
    if titles_x is not None:
        if x_titles_position == "top":
            for col in range(grid_width):
                grid[col].set_title(titles_x[col], fontsize=title_size)
        else:
            # Put titles on bottom as x-axis labels on the last row
            bottom_row_start = (grid_height - 1) * grid_width
            for col in range(grid_width):
                ax = grid[bottom_row_start + col]
                ax.set_xlabel(titles_x[col], fontsize=title_size)
                ax.xaxis.set_label_position("bottom")

    # ---- row titles ---------------------------------------------------------
    if titles_y is not None:
        for row in range(grid_height):
            idx = row * grid_width
            ax = grid[idx]

            ax.set_ylabel(
                titles_y[row],
                rotation=90,
                ha=("right" if y_titles_position == "left" else "left"),
                va="center",
                labelpad=4,
                fontsize=title_size,
            )

            if y_titles_position == "right":
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()

    # ---- save or show -------------------------------------------------------
    if save_to is not None:
        plt.savefig(Path(save_to))

    if show:
        plt.show()
        return
    else:
        return grid


def plot_grid_movie(

    A,
    t=None,
    titles_x=None,
    titles_y=None,
    suptitle=None,
    suptitle_y=None,
    colorbar=True,
    colorbar_mode="single",
    grid_height=None,
    grid_width=None,
    fig_size=(8, 8),
    cmap="viridis",
    xticks_on=False,
    yticks_on=False,
    aspect="auto",
    space=0.1,
    interval=200,
    save_to=None,
    show=True,
    live_cbar=True,
    c_norm=None,
    seconds=5,
    writer="pillow",
    frames=64,
    to_int=False,
):
    """
    Animate a grid of *N* miniature movies and return an embeddable HTML
    representation (or save as a GIF).

    Parameters
    ----------
    A : array-like, shape ``(N, T, H, W)`` or ``(N, T, H, W, C)``
        Stack of *N* movies, each with *T* time-steps.
        When *to_int* is ``True`` the data are clipped to ``[0, 255]``
        and cast to ``uint16`` before display.
    t : 1-D array-like or ``None``, optional
        Time stamps for each frame (currently unused – reserved).
    titles_x : Sequence[str], optional
        Column titles – length **must equal** *grid_width* if supplied.
    titles_y : Sequence[str], optional
        Row titles – length **must equal** *grid_height* if supplied.
    suptitle : str, optional
        Figure-level title.
    suptitle_y : float, optional
        Vertical position (in figure fraction) for *suptitle*.
    colorbar : bool, default ``True``
        Draw colourbar(s).
    colorbar_mode : {'single', 'each'}, default ``'single'``
        One shared colourbar or one per panel.
    grid_height, grid_width : int or ``None``, optional
        Fix grid size; any ``None`` side is inferred so that
        ``grid_height × grid_width ≥ N``.  
        If **both** are given, only the first
        ``grid_height × grid_width`` movies are drawn.
    fig_size : (float, float), default ``(8, 8)``
        Figure size in inches.
    cmap : str or :class:`matplotlib.colors.Colormap`, default ``'viridis'``
    xticks_on, yticks_on : bool, default ``False``
        Toggle axis tick visibility.
    aspect : {'auto', 'equal'} or float, default ``'auto'``
        Aspect ratio argument passed to the internal ``ImageGrid``.
    space : float, default ``0.1``
        Normalised padding between grid cells (``axes_pad``).
    interval : int, default ``200``
        Delay between frames *in milliseconds* for the animation.
    save_to : str or :class:`pathlib.Path`, optional
        Base name to save the animation as ``<name>.gif`` using *writer*.
        Ignored if ``None``.
    show : bool, default ``True``
        Return an embeddable :class:`IPython.display.HTML` object.  
        If ``False`` the function performs the save (if requested) and
        returns ``None``.
    live_cbar : bool, default ``True``
        Update colour limits every frame to reflect current data range.
    c_norm : (float, float) or ``None``, optional
        ``(vmin, vmax)`` applied to **all** frames. ``None`` ⇒ auto‐scale.
    seconds : int, default ``5``
        Target animation length in seconds (used to set FPS).
    writer : str, default ``'pillow'``
        Animation writer passed to Matplotlib (e.g. ``'imagemagick'``).
    frames : int, default ``64``
        Number of frames uniformly sampled from the input *T* steps.
    to_int : bool, default ``False``
        Clip data to 0–255 and convert to ``uint16`` before plotting.
    """
    A = np.asarray(A)
    if grid_width is not None and grid_height is not None:
        A = A[: grid_height * grid_width]
    # A is expected to be an array of movies with shape (n, t, h, w)
    t_idx = np.linspace(0, A.shape[1] - 1, frames, dtype=np.int32)
    A = A[:, t_idx]
    n, t = A.shape[:2]

    # Calculate grid dimensions if not provided
    if grid_height is None and grid_width is None:
        grid_width = int(np.ceil(np.sqrt(n)))
        grid_height = int(np.ceil(n / grid_width))
    elif grid_height is None:
        grid_height = int(np.ceil(n / grid_width))
    elif grid_width is None:
        grid_width = int(np.ceil(n / grid_height))

    # Compute vmin and vmax for consistent color scales
    vmin = A.min()
    vmax = A.max()

    # Create figure
    fig = plt.figure(figsize=fig_size)
    if suptitle is not None:
        fig.suptitle(suptitle, y=suptitle_y)

    # Set up image grid with specified aspect ratio, colorbar mode, and spacing
    cbar_mode = colorbar_mode if colorbar else None
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(grid_height, grid_width),
        axes_pad=space,
        share_all=True,
        cbar_mode=cbar_mode,
        aspect=aspect == "auto",
    )

    images = []

    if to_int:
        A = np.clip(A, a_min=0, a_max=255)
        A = np.asarray(A, dtype=np.uint16)

    # Plot initial images
    for i in range(n):
        ax = grid[i]
        if c_norm is not None:
            norm = colors.Normalize(vmin=c_norm[0], vmax=c_norm[1])
        else:
            norm = colors.Normalize(vmin=np.min(A[i, 0]), vmax=np.max(A[i, 0]))
        im = ax.imshow(A[i, 0], cmap=cmap, aspect="auto", norm=norm)
        images.append(im)
        if not xticks_on:
            ax.set_xticks([])
        if not yticks_on:
            ax.set_yticks([])

        # Add colorbar for each image if needed
        if colorbar and colorbar_mode == "each":
            cbar = ax.cax.colorbar(im)
            ax.cax.tick_params(labelleft=True)

        if titles_x is not None and i < grid_width:
            ax.set_title(titles_x[i])

        if titles_y is not None and i % grid_width == 0:
            ax.set_ylabel(titles_y.pop())

    # Add single colorbar if needed
    if colorbar and colorbar_mode == "single":
        cbar = grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[0].tick_params()

    # Define update function
    def update(frame):
        for i, im in enumerate(images):
            cur = A[i, frame]
            im.set_data(cur)
            if live_cbar:
                vmax = np.max(cur)
                vmin = np.min(cur)
                im.set_clim(vmin, vmax)
        return images

    # Create animation
    ani = FuncAnimation(fig, update, frames=t, interval=interval, blit=False)

    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        fps = t // seconds
        ani.save(
            p,
            writer=writer,
            fps=fps,
        )

    if show:
        return HTML(ani.to_jshtml())


def save_tensor_to_mp4(
    video: np.ndarray,
    out_path: Union[str, Path],
    *,
    fps: int = 30,
    seconds: float | None = None,                 # overrides fps if provided
    codec: str = "libx264",
    crf: int | None = 18,                    # constant-rate factor (0-51)
    bitrate: str | None = None,              # mutually exclusive with *crf*
    pix_fmt: str = "yuv420p",
    resize_to: Tuple[int, int] | None = None,
    progress: bool = True,
    cmap: str | mcolors.Colormap | None = None,   # NEW
    c_norm: Tuple[float, float] | mcolors.Normalize | None = None  # NEW
) -> None:
    """
    Encode a video tensor as an **H.264 MP4** file.

    Parameters
    ----------
    video : np.ndarray, shape ``(T, H, W, C)`` *or* ``(T, H, W)``
        Input frames.  
        *C* (channels) must be 1, 3, or 4.  
        *dtype* may be ``uint8`` (0–255) **or** ``float32/64``  
        in either **[0, 1]** (auto-scaled to 0–255) or **[0, 255]**.
    out_path : str or pathlib.Path
        Destination filename – the ``.mp4`` suffix is appended if missing.
    fps : int, default ``30``
        Frames per second.  Ignored when *seconds* is supplied.
    seconds : float, optional
        Target clip length in seconds.  Sets  
        ``fps = max(round(T / seconds), 1)``.
    codec : str, default ``'libx264'``
        FFmpeg codec name (any encoder supported by your FFmpeg build).
    crf : int, default ``18``
        Constant–rate factor **0 (best) – 51 (worst)**.  
        *Ignored if* *bitrate* is given.
    bitrate : str, optional
        Fixed bitrate such as ``'4M'``.  Mutually exclusive with *crf*.
    pix_fmt : str, default ``'yuv420p'``
        Pixel format handed to FFmpeg (use ``'yuv444p'`` for lossless RGBA).
    resize_to : (int, int), optional
        ``(height, width)`` to isotropically downsample every frame
        via OpenCV *INTER_AREA* before encoding.
    progress : bool, default ``True``
        Show textual progress bars using *tqdm*.
    cmap : str or matplotlib.colors.Colormap, optional
        Apply a Matplotlib colormap **before** encoding.  
        Requires **single-channel** input; results in 3-channel RGB output.
    c_norm : (vmin, vmax) tuple or matplotlib.colors.Normalize, optional
        Normalisation used together with *cmap*.  
        ``None`` ⇒ identity (no scaling).
    """

    # ---------- validation --------------------------------------------------
    if video.ndim not in (3, 4):
        raise ValueError("video must have shape (T, H, W[, C])")

    # Auto-expand (T, H, W) → (T, H, W, 1)
    if video.ndim == 3:
        video = video[..., None]

    if video.shape[-1] not in (1, 3, 4):
        raise ValueError("channel count must be 1, 3, or 4")

    if cmap is not None and video.shape[-1] != 1:
        raise ValueError("cmap can only be used with single-channel input")

    if not (np.issubdtype(video.dtype, np.floating) or video.dtype == np.uint8):
        raise TypeError("dtype must be float32/64 or uint8")

    T = video.shape[0]
    if seconds is not None:
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        fps = max(int(round(T / seconds)), 1)

    # ---------- optional resize --------------------------------------------
    if resize_to is not None:
        import cv2  # pip install opencv-python-headless
        h_new, w_new = resize_to
        iterable = tqdm.tqdm(video, desc="Resizing") if progress else video
        video = np.stack(
            [cv2.resize(f, (w_new, h_new), interpolation=cv2.INTER_AREA)
             for f in iterable],
            axis=0,
        )

    # ---------- apply colormap (if requested) ------------------------------
    if cmap is not None:
        # Resolve cmap → Colormap instance
        cmap_obj = mcm.get_cmap(cmap) if isinstance(cmap, str) else cmap

        # Resolve c_norm → Normalize instance or identity
        if c_norm is None:
            def norm(x): return x
        elif isinstance(c_norm, tuple) or isinstance(c_norm, list):
            norm = mcolors.Normalize(vmin=c_norm[0], vmax=c_norm[1])
        else:
            norm = c_norm  # already a Normalize instance

        iterable = tqdm.tqdm(
            video, desc="Applying cmap") if progress else video
        recoloured = []
        for frame in iterable:
            frame_2d = frame[..., 0].astype(np.float32)
            # (H, W, 4), float in [0,1]
            rgba = cmap_obj(norm(frame_2d))
            rgb = (rgba[..., :3] * 255 + 0.5).astype(np.uint8)
            recoloured.append(rgb)
        video = np.stack(recoloured, axis=0)

    # ---------- float → uint8 *if needed* -----------------------------------
    if np.issubdtype(video.dtype, np.floating):
        # Detect if data are already 0–255: treat values > 1 as “already scaled”
        needs_scaling = video.max() <= 1.0
        if needs_scaling:
            if progress:
                print("Converting float32/64 in [0, 1] → uint8 …")
            video = (np.clip(video, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
        else:
            if progress:
                print("Casting float32/64 in [0, 255] → uint8 (no rescale) …")
            video = video.round().astype(np.uint8)

    # ---------- assemble writer kwargs -------------------------------------
    writer_kwargs: dict[str, Any] = {
        "format": "ffmpeg",
        "mode": "I",
        "fps": fps,
        "codec": codec,
        "pixelformat": pix_fmt,
        "ffmpeg_log_level": "error",
    }

    # bitrate OR crf (mutually exclusive)
    if bitrate is not None:
        writer_kwargs["bitrate"] = bitrate
    elif crf is not None:
        writer_kwargs["ffmpeg_params"] = ["-crf", str(crf)]

    out_path = Path(out_path).with_suffix(".mp4")

    # ---------- encode ------------------------------------------------------
    with imageio.get_writer(out_path, **writer_kwargs) as writer:
        iterable = tqdm.tqdm(video, desc="Encoding",
                             unit="frame") if progress else video
        for frame in iterable:
            writer.append_data(frame)

    if progress:
        print(
            f"✓ Saved {len(video)} frames at {fps} fps → {out_path.resolve()}")
