"""NeuralUQ for Euler-Bernoulli cantilever beam (B-PINN)."""

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
def generate_collocation_points(n_pde, n_bc, n_ic, t_max):
    """Generate collocation points for PDE, BCs, and ICs."""
    # Interior PDE points
    x_pde = np.random.uniform(0, L, (n_pde, 1)).astype(np.float32)
    t_pde = np.random.uniform(0, t_max, (n_pde, 1)).astype(np.float32)
    xt_pde = np.concatenate([x_pde, t_pde], axis=-1)

    # Left BC: x=0, clamped (w=0, dw/dx=0)
    t_bc = np.random.uniform(0, t_max, (n_bc, 1)).astype(np.float32)
    x_left = np.zeros((n_bc, 1), dtype=np.float32)
    xt_left = np.concatenate([x_left, t_bc], axis=-1)

    # Right BC: x=L, free (d^2w/dx^2=0, d^3w/dx^3=0)
    t_bc_r = np.random.uniform(0, t_max, (n_bc, 1)).astype(np.float32)
    x_right = np.full((n_bc, 1), L, dtype=np.float32)
    xt_right = np.concatenate([x_right, t_bc_r], axis=-1)

    # IC: t=0 (w=0, dw/dt=0 assuming rest start)
    x_ic = np.random.uniform(0, L, (n_ic, 1)).astype(np.float32)
    t_ic = np.zeros((n_ic, 1), dtype=np.float32)
    xt_ic = np.concatenate([x_ic, t_ic], axis=-1)

    return xt_pde, xt_left, xt_right, xt_ic


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

    # ── Load data ─────────────────────────────────────────────────────────────
    data_file = "test_TC01_M2_L1_EI1_noise001.npy"
    x, t, w_d, x_test, t_test, w_d_test = load_data(data_file, select_every_nth=2, number_of_modes=1)

    # Flatten data into (N, 2) input arrays and (N, 1) targets
    # w_d has shape [n_x, n_t], build meshgrid
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_data = np.stack([xx.ravel(), tt.ravel()], axis=-1).astype(np.float32)
    w_data = w_d.ravel()[:, None].astype(np.float32)

    t_max = t.max()

    # ── Generate collocation points ───────────────────────────────────────────
    xt_pde, xt_left, xt_right, xt_ic = generate_collocation_points(n_pde, n_bc, n_ic, t_max)
    zeros_pde = np.zeros((n_pde, 1), dtype=np.float32)
    zeros_bc = np.zeros((n_bc, 1), dtype=np.float32)
    zeros_ic = np.zeros((n_ic, 1), dtype=np.float32)

    # IC displacement targets: assume starting from rest (w=0 at t=0)
    # If data contains IC, use w_d[:, 0] instead
    w_ic = w_d[:, 0:1].astype(np.float32)   # shape [n_x, 1]
    x_ic_data = x[:, None] if x.ndim == 1 else x
    t_ic_data = np.zeros_like(x_ic_data, dtype=np.float32)
    xt_ic_data = np.concatenate([x_ic_data.astype(np.float32), t_ic_data], axis=-1)

    # ── Build surrogate and process ───────────────────────────────────────────
    process_w = neuq.process.Process(
        surrogate=neuq.surrogates.FNN(layers=layers),
        prior=neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1),
    )

    # ── Likelihoods ───────────────────────────────────────────────────────────
    # 1) Data likelihood: w(x_i, t_j) ~ N(w_data, sigma_data)
    likelihood_data = neuq.likelihoods.Normal(
        inputs=xt_data,
        targets=w_data,
        processes=[process_w],
        pde=None,
        sigma=sigma_data,
    )

    # 2) PDE residual: EI*w_xxxx + mu*w_tt + c*w_t - p = 0
    likelihood_pde = neuq.likelihoods.Normal(
        inputs=xt_pde,
        targets=zeros_pde,
        processes=[process_w],
        pde=pde_fn,
        sigma=sigma_pde,
    )

    # 3) Left BC — displacement: w(0, t) = 0
    likelihood_left_disp = neuq.likelihoods.Normal(
        inputs=xt_left,
        targets=zeros_bc,
        processes=[process_w],
        pde=None,
        sigma=sigma_bc,
    )

    # 4) Left BC — slope: dw/dx(0, t) = 0
    likelihood_left_slope = neuq.likelihoods.Normal(
        inputs=xt_left,
        targets=zeros_bc,
        processes=[process_w],
        pde=bc_slope_fn,
        sigma=sigma_bc,
    )

    # 5) Right BC — moment: d^2w/dx^2(L, t) = 0
    likelihood_right_moment = neuq.likelihoods.Normal(
        inputs=xt_right,
        targets=zeros_bc,
        processes=[process_w],
        pde=bc_moment_fn,
        sigma=sigma_bc,
    )

    # 6) Right BC — shear: d^3w/dx^3(L, t) = 0
    likelihood_right_shear = neuq.likelihoods.Normal(
        inputs=xt_right,
        targets=zeros_bc,
        processes=[process_w],
        pde=bc_shear_fn,
        sigma=sigma_bc,
    )

    # 7) IC — displacement: w(x, 0) = w_ic(x)
    likelihood_ic_disp = neuq.likelihoods.Normal(
        inputs=xt_ic_data,
        targets=w_ic,
        processes=[process_w],
        pde=None,
        sigma=sigma_ic,
    )

    # 8) IC — velocity: dw/dt(x, 0) = 0
    likelihood_ic_vel = neuq.likelihoods.Normal(
        inputs=xt_ic,
        targets=zeros_ic,
        processes=[process_w],
        pde=ic_velocity_fn,
        sigma=sigma_ic,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = neuq.models.Model(
        processes=[process_w],
        likelihoods=[
            likelihood_data,
            likelihood_pde,
            likelihood_left_disp,
            likelihood_left_slope,
            likelihood_right_moment,
            likelihood_right_shear,
            likelihood_ic_disp,
            likelihood_ic_vel,
        ],
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    method = neuq.inferences.HMC(
        num_samples=500,
        num_burnin=1000,
        init_time_step=0.01,
        leapfrog_step=50,
        seed=7777,
    )
    model.compile(method)
    samples, results = model.run()
    print("Acceptance rate: %.3f" % np.mean(results))

    # ── Predictions ───────────────────────────────────────────────────────────
    xx_test, tt_test = np.meshgrid(x_test, t_test, indexing='ij')
    xt_test = np.stack([xx_test.ravel(), tt_test.ravel()], axis=-1).astype(np.float32)

    (w_pred,) = model.predict(xt_test, samples, processes=[process_w])

    # ── Postprocessing ────────────────────────────────────────────────────────
    n_x_test = len(x_test)
    n_t_test = len(t_test)
    w_pred_reshape = w_pred.reshape([-1, n_x_test, n_t_test])
    w_mean = np.mean(w_pred_reshape, axis=0)
    w_std = np.std(w_pred_reshape, axis=0)

    # Plot at a few spatial locations
    fig, axes = plt.subplots(1, min(3, n_x_test), figsize=(15, 4))
    if n_x_test < 3:
        axes = [axes] if n_x_test == 1 else axes
    plot_indices = np.linspace(0, n_x_test - 1, min(3, n_x_test), dtype=int)
    for ax, ix in zip(axes, plot_indices):
        ax.plot(t_test, w_d_test[ix, :], 'k-', label='Reference')
        ax.plot(t_test, w_mean[ix, :], 'r--', label='Mean')
        ax.fill_between(
            t_test.ravel(),
            (w_mean[ix, :] + 2 * w_std[ix, :]).ravel(),
            (w_mean[ix, :] - 2 * w_std[ix, :]).ravel(),
            alpha=0.3, label='95% CI',
        )
        ax.set_xlabel('t')
        ax.set_ylabel('w(x, t)')
        ax.set_title(f'x = {x_test[ix]:.3f}')
        ax.legend()
    plt.tight_layout()
    plt.savefig("cantilever_bpinn_results.png", dpi=150)
    plt.show()
