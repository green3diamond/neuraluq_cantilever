"""NeuralUQ for 1-D Poisson equation (forward), from B-PINN paper."""


# See also this paper for reference:
# B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import neuraluq as neuq
import neuraluq.variables as neuq_vars
from neuraluq.config import tf

import sys
import numpy as np
import numpy.core
# Monkey-patch: allow unpickling files saved with NumPy 2.x on NumPy 1.x
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import matplotlib.pyplot as plt


# ── Physical parameters ──────────────────────────────────────────────────────
L = 1.0          # beam length
EI = 1.0         # flexural rigidity
mu = 1.0         # mass per unit length
c = 0.0          # damping coefficient
P = 0          # load amplitude
f = 2.4          # forcing frequency


def load_data(data_file, select_every_nth=1, number_of_modes=1):
    """Load cantilever beam data and compute held-out test set."""
    data = np.load(data_file, allow_pickle=True).item()

    t_full = data['t'].astype(np.float32)
    w_d_full = data['U'].astype(np.float32)
    x_full = data['x'].astype(np.float32)

    # Subsample time
    if select_every_nth > 1:
        indexes_t = list(range(0, len(t_full), select_every_nth))
    else:
        indexes_t = list(range(len(t_full)))

    # Subsample space
    if number_of_modes == 1:
        indexes_x = [0, 50, -1]
    elif number_of_modes == 2:
        indexes_x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    elif number_of_modes == 3:
        indexes_x = [0, 6, 13, 19, 26, 32, 39, 45, 52, 58, 65, 71, 78, 84, 91, 99]
    else:
        indexes_x = list(range(len(x_full)))

    t = t_full[indexes_t]
    x = x_full[indexes_x]
    w_d = w_d_full[indexes_x][:, indexes_t]

    # Resolve negative indices for test set computation
    resolved_indexes_x = [i if i >= 0 else len(x_full) + i for i in indexes_x]

    # Test indices = all indices minus training indices minus IC (t=0)
    indexes_x_test = [i for i in range(len(x_full)) if i not in resolved_indexes_x]
    indexes_t_test = [i for i in range(len(t_full)) if i not in indexes_t and i != 0]

    x_test = x_full[indexes_x_test]
    t_test = t_full[indexes_t_test]
    w_d_test = w_d_full[indexes_x_test][:, indexes_t_test]

    return x, t, w_d, x_test, t_test, w_d_test


# ── PDE residual: EI * w_xxxx + mu * w_tt + c * w_t - p(x,t) = 0 ────────────
def pde_fn(xt, w):
    """Euler-Bernoulli beam PDE residual.

    Input xt has columns [x, t]. Output w is displacement.
    """
    w_xt = tf.gradients(w, xt)[0]
    w_x, w_t = w_xt[..., 0:1], w_xt[..., 1:2]

    w_xx = tf.gradients(w_x, xt)[0][..., 0:1]
    w_xxx = tf.gradients(w_xx, xt)[0][..., 0:1]
    w_xxxx = tf.gradients(w_xxx, xt)[0][..., 0:1]
    w_tt = tf.gradients(w_t, xt)[0][..., 1:2]

    x_coord = xt[..., 0:1]
    t_coord = xt[..., 1:2]
    p = -P * tf.sin(np.pi * x_coord / L) * tf.cos(2 * np.pi * f * t_coord)

    residual = EI * w_xxxx + mu * w_tt + c * w_t - p
    return residual


# ── BC/IC PDE helpers (return the quantity that should be zero) ───────────────
def bc_slope_fn(xt, w):
    """dw/dx — used for clamped end slope BC."""
    w_x = tf.gradients(w, xt)[0][..., 0:1]
    return w_x


def bc_moment_fn(xt, w):
    """d^2 w/dx^2 — used for free end moment BC."""
    w_x = tf.gradients(w, xt)[0][..., 0:1]
    w_xx = tf.gradients(w_x, xt)[0][..., 0:1]
    return w_xx


def bc_shear_fn(xt, w):
    """d^3 w/dx^3 — used for free end shear BC."""
    w_x = tf.gradients(w, xt)[0][..., 0:1]
    w_xx = tf.gradients(w_x, xt)[0][..., 0:1]
    w_xxx = tf.gradients(w_xx, xt)[0][..., 0:1]
    return w_xxx


def ic_velocity_fn(xt, w):
    """dw/dt — used for zero initial velocity IC."""
    w_t = tf.gradients(w, xt)[0][..., 1:2]
    return w_t


# ── Collocation / BC / IC point generation ────────────────────────────────────
def generate_collocation_points(n_pde, n_bc, n_ic, t_start, t_end):
    """Generate collocation points for PDE, BCs, and ICs within [t_start, t_end]."""
    # Interior PDE points
    x_pde = np.random.uniform(0, L, (n_pde, 1)).astype(np.float32)
    t_pde = np.random.uniform(t_start, t_end, (n_pde, 1)).astype(np.float32)
    xt_pde = np.concatenate([x_pde, t_pde], axis=-1)

    # Left BC: x=0, clamped (w=0, dw/dx=0)
    t_bc = np.random.uniform(t_start, t_end, (n_bc, 1)).astype(np.float32)
    x_left = np.zeros((n_bc, 1), dtype=np.float32)
    xt_left = np.concatenate([x_left, t_bc], axis=-1)

    # Right BC: x=L, free (d^2w/dx^2=0, d^3w/dx^3=0)
    t_bc_r = np.random.uniform(t_start, t_end, (n_bc, 1)).astype(np.float32)
    x_right = np.full((n_bc, 1), L, dtype=np.float32)
    xt_right = np.concatenate([x_right, t_bc_r], axis=-1)

    # IC: t=t_start
    x_ic = np.random.uniform(0, L, (n_ic, 1)).astype(np.float32)
    t_ic = np.full((n_ic, 1), t_start, dtype=np.float32)
    xt_ic = np.concatenate([x_ic, t_ic], axis=-1)

    return xt_pde, xt_left, xt_right, xt_ic


def train_interval_pinn(x, t_interval, w_d_interval, x_ic_pts, w_ic_targets,
                        layers, n_pde, n_bc, n_ic, sigma_data, sigma_pde,
                        sigma_bc, sigma_ic, num_samples, num_burnin, seed):
    """Train a B-PINN on a single time interval.

    Args:
        x: spatial training locations, shape [n_x]
        t_interval: time points in this interval, shape [n_t_int]
        w_d_interval: data in this interval, shape [n_x, n_t_int]
        x_ic_pts: spatial points for IC, shape [n_ic_pts]
        w_ic_targets: IC displacement targets, shape [n_ic_pts, 1]
        Returns: (model, process_w, samples, results)
    """
    t_start, t_end = t_interval.min(), t_interval.max()

    # Flatten data
    xx, tt = np.meshgrid(x, t_interval, indexing='ij')
    xt_data = np.stack([xx.ravel(), tt.ravel()], axis=-1).astype(np.float32)
    w_data = w_d_interval.ravel()[:, None].astype(np.float32)

    # Collocation points
    xt_pde, xt_left, xt_right, xt_ic = generate_collocation_points(
        n_pde, n_bc, n_ic, t_start, t_end)
    zeros_pde = np.zeros((n_pde, 1), dtype=np.float32)
    zeros_bc = np.zeros((n_bc, 1), dtype=np.float32)
    zeros_ic = np.zeros((n_ic, 1), dtype=np.float32)

    # IC data points
    x_ic_data = x_ic_pts[:, None] if x_ic_pts.ndim == 1 else x_ic_pts
    t_ic_data = np.full_like(x_ic_data, t_start, dtype=np.float32)
    xt_ic_data = np.concatenate([x_ic_data.astype(np.float32), t_ic_data], axis=-1)

    # Build surrogate and process
    process_w = neuq.process.Process(
        surrogate=neuq.surrogates.FNN(layers=layers),
        prior=neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1),
    )

    # Likelihoods
    likelihoods = [
        neuq.likelihoods.Normal(inputs=xt_data, targets=w_data,
                                processes=[process_w], pde=None, sigma=sigma_data),
        neuq.likelihoods.Normal(inputs=xt_pde, targets=zeros_pde,
                                processes=[process_w], pde=pde_fn, sigma=sigma_pde),
        neuq.likelihoods.Normal(inputs=xt_left, targets=zeros_bc,
                                processes=[process_w], pde=None, sigma=sigma_bc),
        neuq.likelihoods.Normal(inputs=xt_left, targets=zeros_bc,
                                processes=[process_w], pde=bc_slope_fn, sigma=sigma_bc),
        neuq.likelihoods.Normal(inputs=xt_right, targets=zeros_bc,
                                processes=[process_w], pde=bc_moment_fn, sigma=sigma_bc),
        neuq.likelihoods.Normal(inputs=xt_right, targets=zeros_bc,
                                processes=[process_w], pde=bc_shear_fn, sigma=sigma_bc),
        neuq.likelihoods.Normal(inputs=xt_ic_data, targets=w_ic_targets,
                                processes=[process_w], pde=None, sigma=sigma_ic),
        neuq.likelihoods.Normal(inputs=xt_ic, targets=zeros_ic,
                                processes=[process_w], pde=ic_velocity_fn, sigma=sigma_ic),
    ]

    model = neuq.models.Model(processes=[process_w], likelihoods=likelihoods)
    method = neuq.inferences.HMC(
        num_samples=num_samples, num_burnin=num_burnin,
        init_time_step=0.1, leapfrog_step=5, seed=seed,
    )
    model.compile(method)
    samples, results = model.run()

    return model, process_w, samples, results


if __name__ == "__main__":
    # ── Hyperparameters ───────────────────────────────────────────────────────
    layers = [2, 50, 50, 50, 1]   # input: (x, t), output: w
    n_pde = 5000       # collocation points for PDE
    n_bc = 300         # boundary points per edge
    n_ic = 100         # initial condition points
    sigma_data = 0.05  # noise std for data likelihood
    sigma_pde = 0.01   # noise std for PDE residual likelihood
    sigma_bc = 0.01    # noise std for BC likelihoods
    sigma_ic = 0.01    # noise std for IC likelihoods
    num_samples = 100
    num_burnin = 0
    n_intervals = 3

    # ── Load data ─────────────────────────────────────────────────────────────
    data_file = "test_TC01_M2_L1_EI1_noise001.npy"
    x, t, w_d, x_test, t_test, w_d_test = load_data(data_file, select_every_nth=2, number_of_modes=2)

    n_x_test = len(x_test)
    n_t_test = len(t_test)

    # ── Split time into 3 intervals ──────────────────────────────────────────
    t_boundaries = np.linspace(t.min(), t.max(), n_intervals + 1)
    time_intervals = [(t_boundaries[i], t_boundaries[i + 1]) for i in range(n_intervals)]
    print(f"Time intervals: {time_intervals}")

    # IC for first interval comes from data at t=0
    w_ic_next = w_d[:, 0:1].astype(np.float32)

    # Collect predictions across all intervals
    w_mean_full = np.zeros((n_x_test, n_t_test))
    w_std_full = np.zeros((n_x_test, n_t_test))
    models_info = []

    for iv, (t_start, t_end) in enumerate(time_intervals):
        print(f"\n{'='*60}")
        print(f"Interval {iv+1}/{n_intervals}: t ∈ [{t_start:.4f}, {t_end:.4f}]")
        print(f"{'='*60}")

        # Select training data in this interval
        t_mask = (t >= t_start) & (t <= t_end)
        t_interval = t[t_mask]
        w_d_interval = w_d[:, t_mask]

        # Train B-PINN for this interval
        model, process_w, samples, results = train_interval_pinn(
            x=x, t_interval=t_interval, w_d_interval=w_d_interval,
            x_ic_pts=x, w_ic_targets=w_ic_next,
            layers=layers, n_pde=n_pde, n_bc=n_bc, n_ic=n_ic,
            sigma_data=sigma_data, sigma_pde=sigma_pde,
            sigma_bc=sigma_bc, sigma_ic=sigma_ic,
            num_samples=num_samples, num_burnin=num_burnin,
            seed=7777 + iv,
        )
        print(f"Interval {iv+1} acceptance rate: {np.mean(results):.3f}")

        # Predict on test points within this interval
        t_test_mask = (t_test >= t_start) & (t_test <= t_end)
        t_test_interval = t_test[t_test_mask]

        if len(t_test_interval) > 0:
            xx_ti, tt_ti = np.meshgrid(x_test, t_test_interval, indexing='ij')
            xt_test_iv = np.stack([xx_ti.ravel(), tt_ti.ravel()], axis=-1).astype(np.float32)

            (w_pred_iv,) = model.predict(xt_test_iv, samples, processes=[process_w])
            n_t_iv = len(t_test_interval)
            w_pred_reshape = w_pred_iv.reshape([-1, n_x_test, n_t_iv])
            w_mean_iv = np.mean(w_pred_reshape, axis=0)
            w_std_iv = np.std(w_pred_reshape, axis=0)

            # Place into full arrays
            t_test_indices = np.where(t_test_mask)[0]
            w_mean_full[:, t_test_indices] = w_mean_iv
            w_std_full[:, t_test_indices] = w_std_iv

        # Get IC for next interval: predict at t_end for training x locations
        x_ic_next = x[:, None] if x.ndim == 1 else x
        t_ic_next = np.full_like(x_ic_next, t_end, dtype=np.float32)
        xt_boundary = np.concatenate([x_ic_next.astype(np.float32), t_ic_next], axis=-1)
        (w_boundary,) = model.predict(xt_boundary, samples, processes=[process_w])
        w_ic_next = np.mean(w_boundary, axis=0).reshape(-1, 1).astype(np.float32)

        models_info.append({
            'model': model, 'process': process_w, 'samples': samples,
            't_start': t_start, 't_end': t_end,
        })
        break

    # ── Postprocessing ────────────────────────────────────────────────────────
    w_mean = w_mean_full
    w_std = w_std_full

    # ── Plot 1: Spacetime heatmaps (mean, uncertainty, reference) ────────────
    w_upper = w_mean + 2 * w_std
    w_lower = w_mean - 2 * w_std
    vmin = min(w_mean.min(), w_d_test.min())
    vmax = max(w_mean.max(), w_d_test.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(w_mean, aspect='auto', origin='lower',
                         extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()],
                         cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im0, ax=axes[0], label='Displacement')
    axes[0].set_xlabel('Time t', labelpad=5)
    axes[0].set_ylabel('Position x', labelpad=5)
    axes[0].set_title('Mean Prediction', pad=15)

    im1 = axes[1].imshow(2 * w_std, aspect='auto', origin='lower',
                         extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()],
                         cmap='Oranges', vmin=0)
    plt.colorbar(im1, ax=axes[1], label='Width (2σ)')
    axes[1].set_xlabel('Time t', labelpad=5)
    axes[1].set_ylabel('Position x', labelpad=5)
    axes[1].set_title('Uncertainty (95% CI Width)', pad=15)

    im2 = axes[2].imshow(w_d_test, aspect='auto', origin='lower',
                         extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()],
                         cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=axes[2], label='Displacement')
    axes[2].set_xlabel('Time t', labelpad=5)
    axes[2].set_ylabel('Position x', labelpad=5)
    axes[2].set_title('Reference Data', pad=15)

    # Show interval boundaries
    for ax in axes:
        for tb in t_boundaries[1:-1]:
            ax.axvline(x=tb, color='white', linestyle='--', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig("results_hmc/cantilever_bpinn_heatmap.png", dpi=150)
    # plt.savefig("results_hmc/cantilever_bpinn_heatmap.pdf")
    plt.show()

    # ── Plot 2: Interval comparison at specific time slices ────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    time_points = np.linspace(t_test.min(), t_test.max(), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, t_val in enumerate(time_points):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Find nearest test time index
        t_idx = np.argmin(np.abs(t_test - t_val))

        # Plot uncertainty interval
        ax.fill_between(x_test, w_lower[:, t_idx], w_upper[:, t_idx],
                        alpha=0.3, color='blue', label='95% CI')
        ax.plot(x_test, w_mean[:, t_idx], 'k-', linewidth=2, label='Mean')
        ax.plot(x_test, w_lower[:, t_idx], 'k--', linewidth=1, alpha=0.7)
        ax.plot(x_test, w_upper[:, t_idx], 'k--', linewidth=1, alpha=0.7)

        # Reference data
        ax.plot(x_test, w_d_test[:, t_idx], 'r.', markersize=6, label='Reference')

        # Training data at nearest time
        t_train_idx = np.argmin(np.abs(t - t_test[t_idx]))
        if np.abs(t[t_train_idx] - t_test[t_idx]) < (t[1] - t[0]):
            ax.scatter(x, w_d[:, t_train_idx], c='#00ff55', s=55,
                       marker='s', edgecolors='k', linewidths=0.3, label='Train', zorder=5)

        # Determine which PINN covers this time
        pinn_idx = n_intervals - 1
        for j, (ts, te) in enumerate(time_intervals):
            if ts <= t_test[t_idx] <= te:
                pinn_idx = j
                break

        ax.set_xlabel('Position x')
        ax.set_ylabel('Displacement w')
        ax.set_title(f't = {t_test[t_idx]:.3f} (PINN #{pinn_idx+1})')
        ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color='black', linewidth=2, label='Mean'),
        Line2D([0], [0], color='black', linewidth=1, linestyle='--', label='Bounds'),
        Patch(facecolor='blue', alpha=0.3, label='95% CI'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='red',
               markersize=8, label='Reference'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ff55',
               markersize=8, markeredgecolor='black', markeredgewidth=0.3, label='Train data'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("results_hmc/cantilever_bpinn_intervals.png", dpi=150, bbox_inches='tight')
    # plt.savefig("results_hmc/cantilever_bpinn_intervals.pdf", bbox_inches='tight')
    plt.show()
