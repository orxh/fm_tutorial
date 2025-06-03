import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
from IPython.display import HTML, display
from matplotlib.lines import Line2D
import torch


def sample_gaussian_mixture(n_samples, means_list, covs_list, weights_list):
    """Samples from a Mixture of Gaussians."""
    n_components = len(means_list)

    if n_components == 1:
        return multivariate_normal.rvs(
            mean=means_list[0],
            cov=covs_list[0],
            size=n_samples
        )

    normalized_weights = np.array(weights_list) / np.sum(weights_list)
    counts = np.random.multinomial(n_samples, normalized_weights)

    sampled_points = []
    for i in range(n_components):
        if counts[i] > 0:
            component_samples = multivariate_normal.rvs(mean=means_list[i], cov=covs_list[i], size=counts[i])
            if counts[i] == 1 and component_samples.ndim == 1:
                component_samples = component_samples.reshape(1, -1)
            sampled_points.append(component_samples)

    return np.vstack(sampled_points)


def gaussian_mixture_pdf(x, means_list, covs_list, weights_list):
    """Calculates the PDF of a Mixture of Gaussians."""
    n_components = len(means_list)

    if n_components == 1:
        return multivariate_normal(mean=means_list[0], cov=covs_list[0]).pdf(x)

    pdf_val = np.zeros_like(x[..., 0])
    normalized_weights = np.array(weights_list) / np.sum(weights_list)
    for i in range(n_components):
        pdf_val += normalized_weights[i] * multivariate_normal(mean=means_list[i], cov=covs_list[i]).pdf(x)
    return pdf_val


def generate_flow_animation(
    source_dist_params,
    target_dist_params,
    config,
    animation_title_prefix="",
):
    N_POINTS = config.get('N_POINTS', 1000)
    N_STEPS = config.get('N_STEPS', 50)
    T_MAX = config.get('T_MAX', 1.0)
    DT = T_MAX / N_STEPS
    INTERVAL_MS = config.get('INTERVAL_MS', 100)
    CONTOUR_LEVELS = config.get('CONTOUR_LEVELS', 5)


    source_samples = sample_gaussian_mixture(
        N_POINTS,
        source_dist_params['means'],
        source_dist_params['covs'],
        source_dist_params['weights']
    )

    target_samples_all = sample_gaussian_mixture(
        N_POINTS,
        target_dist_params['means'],
        target_dist_params['covs'],
        target_dist_params['weights']
    )

    xlims = (-7, 7)
    ylims = (-7, 7)

    contour_xx, contour_yy = np.meshgrid(np.linspace(xlims[0], xlims[1], 100), np.linspace(ylims[0], ylims[1], 100))
    contour_pos = np.dstack((contour_xx, contour_yy))

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=100)

    plot_elements = {}

    def setup_plot_ax(ax_handle, initial_source_samples, all_target_samples, paired_target_samples_for_flow):
        current_x_samples = initial_source_samples.copy()
        sample_velocities = (paired_target_samples_for_flow - initial_source_samples) / T_MAX

        scatter_plot = ax_handle.scatter(current_x_samples[:, 0], current_x_samples[:, 1], c='black', s=10, alpha=0.7, label='Flowing Samples')
        target_display_scatter = ax_handle.scatter(all_target_samples[:, 0], all_target_samples[:, 1], c='magenta', marker='x', s=10, alpha=0.5, label='Target Points')

        lines = []
        for i in range(N_POINTS):
            line, = ax_handle.plot([initial_source_samples[i, 0], paired_target_samples_for_flow[i, 0]],
                                   [initial_source_samples[i, 1], paired_target_samples_for_flow[i, 1]],
                                   'gray', alpha=0.2, linewidth=0.5)
            lines.append(line)

        source_pdf_values = gaussian_mixture_pdf(
            contour_pos, source_dist_params['means'], source_dist_params['covs'], source_dist_params['weights']
        )
        ax_handle.contour(contour_xx, contour_yy, source_pdf_values, levels=CONTOUR_LEVELS, colors='blue', alpha=0.5, linestyles='dashed')

        target_pdf_values = gaussian_mixture_pdf(
            contour_pos, target_dist_params['means'], target_dist_params['covs'], target_dist_params['weights']
        )
        ax_handle.contour(contour_xx, contour_yy, target_pdf_values, levels=CONTOUR_LEVELS, colors='red', alpha=0.5, linestyles='dashed')

        ax_handle.set_xlim(xlims)
        ax_handle.set_ylim(ylims)
        ax_handle.set_aspect('equal', adjustable='box')
        ax_handle.grid(True, linestyle=':', alpha=0.4)
        title_artist = ax_handle.set_title(f"{animation_title_prefix} (t = 0.00)")

        legend_handles = [
            scatter_plot,
            target_display_scatter,
            Line2D([0], [0], color='gray', lw=1, alpha=0.3),
            Line2D([0], [0], color='blue', lw=2, linestyle='dashed', alpha=0.5),
            Line2D([0], [0], color='red', lw=2, linestyle='dashed', alpha=0.5),
        ]
        legend_labels = ['Samples', 'Target Points', 'Source-Target Paths', 'Source (t=0)', 'Target (t=1)']
        ax_handle.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

        plot_elements['current_samples'] = current_x_samples
        plot_elements['initial_source_samples'] = initial_source_samples.copy()
        plot_elements['velocities'] = sample_velocities
        plot_elements['scatter_plot_artist'] = scatter_plot
        plot_elements['title_artist'] = title_artist

    random_indices = np.random.permutation(N_POINTS)
    paired_target_samples = target_samples_all[random_indices]

    setup_plot_ax(ax, source_samples, target_samples_all, paired_target_samples)

    def update_animation(frame):
        artists = []
        current_t = frame * DT

        if frame == 0:
            plot_elements['current_samples'] = plot_elements['initial_source_samples'].copy()
        else:
            time_factor = min(current_t, T_MAX)
            plot_elements['current_samples'] = plot_elements['initial_source_samples'] + \
                                             plot_elements['velocities'] * time_factor

        plot_elements['scatter_plot_artist'].set_offsets(plot_elements['current_samples'])
        artists.append(plot_elements['scatter_plot_artist'])

        plot_elements['title_artist'].set_text(f"{animation_title_prefix} (t = {current_t:.2f})")
        artists.append(plot_elements['title_artist'])

        return artists

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update_animation, frames=N_STEPS + 1,
                                 interval=INTERVAL_MS, blit=True, repeat=True)

    html_output = HTML(ani.to_jshtml())
    display(html_output)

    plt.close(fig)
    return ani


def animate_flow_with_arrows(
    model,
    source_dist_params,
    num_samples: int = 1000,
    num_steps: int = 50,
    interval_ms: int = 100,
    grid_size: int = 20,
    arrow_scale: float = 30.0,
):
    model.eval()
    device = next(model.parameters()).device
    title = 'Learned velocity field'

    src_np = sample_gaussian_mixture(
        num_samples,
        source_dist_params['means'],
        source_dist_params['covs'],
        source_dist_params['weights']
    )
    src = torch.from_numpy(src_np).float().to(device)

    path = model.sample_path(src, num_steps=num_steps).cpu().numpy()

    xs = np.linspace(-8, 8, grid_size)
    ys = np.linspace(-8, 8, grid_size)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_points = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)

    grid_torch = torch.from_numpy(grid_points).float().to(device)

    ts = torch.linspace(0, 1, num_steps + 1, device=device).unsqueeze(-1)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    scat = ax.scatter([], [], s=8, c="black", alpha=0.8, label="particles")

    U0 = np.zeros_like(Xg)
    V0 = np.zeros_like(Yg)
    quiv = ax.quiver(
        Xg, Yg, U0, V0,
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        color="red",
        alpha=0.7,
        width=0.004,
        label="velocity field",
    )

    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.set_title(f"{title} (t=0.00)")
    ax.legend(loc="upper right")

    fig.tight_layout()

    def init():
        scat.set_offsets(np.empty((0, 2)))
        quiv.set_UVC(np.zeros_like(U0), np.zeros_like(V0))
        return scat, quiv

    def update(frame_idx: int):
        t_scalar = ts[frame_idx : frame_idx + 1]  
        current_positions = path[frame_idx]
        scat.set_offsets(current_positions)

        t_repeat = t_scalar.repeat(grid_torch.size(0), 1)
        with torch.no_grad():
            v_torch = model(t_repeat, grid_torch)
        v_np = v_torch.cpu().numpy()

        U = v_np[:, 0].reshape(grid_size, grid_size)
        V = v_np[:, 1].reshape(grid_size, grid_size)

        quiv.set_UVC(U, V)

        ax.set_title(f"{title} (t={frame_idx/num_steps:.2f})")
        return scat, quiv

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_steps + 1,
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    html_output = HTML(ani.to_jshtml())
    display(html_output)

    plt.close()


def visualize_flow_path(trained_flow, source_dist_params, num_samples=250, num_steps=64):
    trained_flow.eval()
    device = next(trained_flow.parameters()).device

    src_np = sample_gaussian_mixture(
        num_samples,
        source_dist_params['means'],
        source_dist_params['covs'],
        source_dist_params['weights']
    )

    src = torch.from_numpy(src_np).float().to(device)
    path = trained_flow.sample_path(src, num_steps=num_steps).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

    ax.scatter(path[0, :, 0], path[0, :, 1], s=15, alpha=1, color='red', label="Start Points", marker='x')
    ax.scatter(path[:, :, 0], path[:, :, 1], s=2, alpha=0.3, color='olive', label="FM Path")
    ax.scatter(path[-1, :, 0], path[-1, :, 1], s=15, alpha=1, color='blue', label="End Points", marker='x')

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.grid(True, linestyle='--', alpha=0.5)
    legend = ax.legend(loc="upper left", fontsize=10)
    legend.legend_handles[1].set_sizes([15])

    plt.tight_layout()
    plt.show()

    plt.close(fig)
