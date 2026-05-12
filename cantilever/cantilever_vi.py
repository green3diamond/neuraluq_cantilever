"""NeuralUQ for Euler-Bernoulli cantilever beam — Variational Inference version.

The neural network outputs two channels: (w, w_xx). The w_xx channel is
trained against finite-difference second spatial derivatives of the data,
so the 4th-order spatial term in the PDE residual is obtained by taking
two autodiff derivatives of the NN's w_xx output instead of four of w.
At prediction time, only the displacement channel w is used.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import neuraluq as neuq
from neuraluq.config import tf

import re
import sys
import numpy as np
import numpy.core
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size':        17,
    'axes.labelsize':   17,
    'xtick.labelsize':  17,
    'ytick.labelsize':  17,
    'legend.fontsize':  17,
    'axes.titlesize':   19,
    'figure.titlesize': 19,
})


# ── Output directory ─────────────────────────────────────────────────────────
RESULTS_DIR = sys.argv[2]
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Physical parameters (overwritten from beam_params at runtime) ─────────────
L = 1.0
EI = 1.0
mu = 1.0
c = 0.0
P = 0
f = 2.4


def finite_diff_xx(w, x):
    """Central-difference second spatial derivative along axis 0."""
    w_x = np.gradient(w, x, axis=0)
    w_xx = np.gradient(w_x, x, axis=0)
    return w_xx


def load_data(data_file, select_every_nth=1, number_of_modes=1):
    """Load cantilever beam data and compute held-out test set + FD w_xx."""
    data = np.load(data_file, allow_pickle=True).item()

    t_full = data['t'].astype(np.float32)
    w_d_full = data['U'].astype(np.float32)
    x_full = data['x'].astype(np.float32)
    beam_params = data['beam_params']

    # FD second derivative on the full dense spatial grid, then subsample
    w_xx_full = finite_diff_xx(w_d_full, x_full).astype(np.float32)

    if select_every_nth > 1:
        indexes_t = list(range(0, len(t_full), select_every_nth))
    else:
        indexes_t = list(range(len(t_full)))

    if number_of_modes == 1:
        indexes_x = [0, 50, -1]
    elif number_of_modes == 2:
        indexes_x = [0, 20, 40, 60, 80, 99]
    elif number_of_modes == 3:
        indexes_x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    else:
        indexes_x = list(range(len(x_full)))

    t_train = t_full[indexes_t]
    x_train = x_full[indexes_x]
    w_d_train = w_d_full[indexes_x][:, indexes_t]
    w_xx_train = w_xx_full[indexes_x][:, indexes_t]

    resolved_indexes_x = [i if i >= 0 else len(x_full) + i for i in indexes_x]
    indexes_x_test = [i for i in range(len(x_full)) if i not in resolved_indexes_x]
    indexes_t_test = [i for i in range(len(t_full)) if i not in indexes_t and i != 0]

    x_test = x_full[indexes_x_test]
    t_test = t_full[indexes_t_test]
    w_d_test = w_d_full[indexes_x_test][:, indexes_t_test]

    noise_amplitude = float(data.get('noise_amplitude', 0))
    noise_type = data.get('noise_type', 'unknown')

    return (x_full, t_full, w_d_full, w_xx_full,
            x_train, t_train, w_d_train, w_xx_train,
            x_test, t_test, w_d_test,
            beam_params, noise_amplitude, noise_type)


def compute_sigmas(L, EI, mu, F_tip):
    """Compute per-term sigma valcues from physical loss coefficients.

    In NeuralUQ, Normal log-likelihood ~ -1/(2*sigma^2) * ||residual||^2,
    so sigma = 1/sqrt(2*coeff) maps loss coefficients to sigma values.
    """
    Y_ref = F_tip * L**3 / EI
    time_concern = (mu * L**4 / EI) ** 0.5

    loss_BC_coeff = (1 / Y_ref**2) * np.array([1e2, L**2, L**4, L**6])
    loss_PDE_coeff = (1e-2 / Y_ref**2) * (L**8 / EI**2)
    loss_IC_coeff = (1 / Y_ref**2) * np.array([1., 0.1, time_concern**2])
    loss_D_coeff = (1 / Y_ref**2) * 10
    # w_xx has units of Y_ref / L^2; normalize residual by that scale.
    loss_D_wxx_coeff = (L**4 / Y_ref**2) * 10

    sigma_data = float(1.0 / np.sqrt(2 * loss_D_coeff))
    sigma_data_wxx = float(1.0 / np.sqrt(2 * loss_D_wxx_coeff))
    sigma_pde = float(1.0 / np.sqrt(2 * loss_PDE_coeff))
    sigma_bc_left_disp = float(1.0 / np.sqrt(2 * loss_BC_coeff[0]))
    sigma_bc_left_slope = float(1.0 / np.sqrt(2 * loss_BC_coeff[1]))
    sigma_bc_right_moment = float(1.0 / np.sqrt(2 * loss_BC_coeff[2]))
    sigma_bc_right_shear = float(1.0 / np.sqrt(2 * loss_BC_coeff[3]))
    sigma_ic_disp = float(1.0 / np.sqrt(2 * loss_IC_coeff[0]))
    sigma_ic_vel = float(1.0 / np.sqrt(2 * loss_IC_coeff[2]))

    return {
        'data': sigma_data,
        'data_wxx': sigma_data_wxx,
        'pde': sigma_pde,
        'bc_left_disp': sigma_bc_left_disp,
        'bc_left_slope': sigma_bc_left_slope,
        'bc_right_moment': sigma_bc_right_moment,
        'bc_right_shear': sigma_bc_right_shear,
        'ic_disp': sigma_ic_disp,
        'ic_vel': sigma_ic_vel,
    }


# ── Output-channel extractors ────────────────────────────────────────────────
# NN output is concatenated as [w, w_xx] along the last axis.
def w_disp_fn(xt, w):
    return w[..., 0:1]


def w_xx_fn(xt, w):
    return w[..., 1:2]


# ── PDE residual: EI * w_xxxx + mu * w_tt + c * w_t - p(x,t) = 0 ────────────
# w_xxxx is taken as the 2nd autodiff x-derivative of the NN's w_xx channel.
def pde_fn(xt, w):
    w_disp = w[..., 0:1]
    w_xx_pred = w[..., 1:2]

    w_disp_xt = tf.gradients(w_disp, xt)[0]
    w_t = w_disp_xt[..., 1:2]
    w_tt = tf.gradients(w_t, xt)[0][..., 1:2]

    w_xx_xt = tf.gradients(w_xx_pred, xt)[0]
    w_xxx = w_xx_xt[..., 0:1]
    w_xxxx = tf.gradients(w_xxx, xt)[0][..., 0:1]

    return EI * w_xxxx + mu * w_tt


def bc_slope_fn(xt, w):
    return tf.gradients(w[..., 0:1], xt)[0][..., 0:1]


def bc_moment_fn(xt, w):
    # w_xx is a direct NN output — no autodiff needed.
    return w[..., 1:2]


def bc_shear_fn(xt, w):
    # 3rd x-derivative = 1st x-derivative of the w_xx output channel.
    return tf.gradients(w[..., 1:2], xt)[0][..., 0:1]


def ic_velocity_fn(xt, w):
    return tf.gradients(w[..., 0:1], xt)[0][..., 1:2]


# ── Collocation / BC / IC point generation ────────────────────────────────────
def generate_collocation_points(n_pde, n_bc, n_ic, t_start, t_end):
    x_pde = np.random.uniform(0, L, (n_pde, 1)).astype(np.float32)
    t_pde = np.random.uniform(t_start, t_end, (n_pde, 1)).astype(np.float32)
    xt_pde = np.concatenate([x_pde, t_pde], axis=-1)

    t_bc = np.random.uniform(t_start, t_end, (n_bc, 1)).astype(np.float32)
    x_left = np.zeros((n_bc, 1), dtype=np.float32)
    xt_left = np.concatenate([x_left, t_bc], axis=-1)

    t_bc_r = np.random.uniform(t_start, t_end, (n_bc, 1)).astype(np.float32)
    x_right = np.full((n_bc, 1), L, dtype=np.float32)
    xt_right = np.concatenate([x_right, t_bc_r], axis=-1)

    x_ic = np.random.uniform(0, L, (n_ic, 1)).astype(np.float32)
    t_ic = np.full((n_ic, 1), t_start, dtype=np.float32)
    xt_ic = np.concatenate([x_ic, t_ic], axis=-1)

    return xt_pde, xt_left, xt_right, xt_ic


def train_interval_vi(x, t_interval, w_d_interval, w_xx_d_interval,
                      x_ic_pts, w_ic_targets,
                      layers, n_pde, n_bc, n_ic, sigmas,
                      vi_batch_size, num_samples, num_iterations,
                      ic_sigma=None):
    """Train a VI B-PINN on a single time interval.

    The NN has 2 outputs (w, w_xx). Data likelihoods are placed on both
    channels; PDE/BC residuals consume the appropriate channels.
    """
    t_start, t_end = t_interval.min(), t_interval.max()

    # Flatten data
    xx, tt = np.meshgrid(x, t_interval, indexing='ij')
    xt_data = np.stack([xx.ravel(), tt.ravel()], axis=-1).astype(np.float32)
    w_data = w_d_interval.ravel()[:, None].astype(np.float32)
    w_xx_data = w_xx_d_interval.ravel()[:, None].astype(np.float32)

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

    # VI requires both prior and variational posterior
    prior = neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1)
    posterior = neuq.variables.fnn.Variational(
        layers=layers, mean=0, sigma=0.1, trainable=True,
    )

    process_w = neuq.process.Process(
        surrogate=neuq.surrogates.FNN(layers=layers),
        prior=prior,
        posterior=posterior,
    )

    sigma_ic_disp = ic_sigma if ic_sigma is not None else sigmas['ic_disp']
    print(f"IC data sigma: {sigma_ic_disp:.6f}")

    likelihoods = [
        # Data: displacement (first NN output channel)
        neuq.likelihoods.Normal(inputs=xt_data, targets=w_data,
                                processes=[process_w], pde=w_disp_fn,
                                sigma=sigmas['data']),
        # Data: FD second derivative (second NN output channel)
        neuq.likelihoods.Normal(inputs=xt_data, targets=w_xx_data,
                                processes=[process_w], pde=w_xx_fn,
                                sigma=sigmas['data_wxx']),
        # PDE: uses both channels
        neuq.likelihoods.Normal(inputs=xt_pde, targets=zeros_pde,
                                processes=[process_w], pde=pde_fn,
                                sigma=sigmas['pde']),
        # BCs
        neuq.likelihoods.Normal(inputs=xt_left, targets=zeros_bc,
                                processes=[process_w], pde=w_disp_fn,
                                sigma=sigmas['bc_left_disp']),
        neuq.likelihoods.Normal(inputs=xt_left, targets=zeros_bc,
                                processes=[process_w], pde=bc_slope_fn,
                                sigma=sigmas['bc_left_slope']),
        neuq.likelihoods.Normal(inputs=xt_right, targets=zeros_bc,
                                processes=[process_w], pde=bc_moment_fn,
                                sigma=sigmas['bc_right_moment']),
        neuq.likelihoods.Normal(inputs=xt_right, targets=zeros_bc,
                                processes=[process_w], pde=bc_shear_fn,
                                sigma=sigmas['bc_right_shear']),
        # ICs
        neuq.likelihoods.Normal(inputs=xt_ic_data, targets=w_ic_targets,
                                processes=[process_w], pde=w_disp_fn,
                                sigma=sigma_ic_disp),
        neuq.likelihoods.Normal(inputs=xt_ic, targets=zeros_ic,
                                processes=[process_w], pde=ic_velocity_fn,
                                sigma=sigmas['ic_vel']),
    ]

    model = neuq.models.Model(processes=[process_w], likelihoods=likelihoods)

    method = neuq.inferences.VI(
        batch_size=vi_batch_size,
        num_samples=num_samples,
        num_iterations=num_iterations,
        optimizer=tf.train.AdamOptimizer(1e-3),
    )
    model.compile(method)
    samples = model.run()

    return model, process_w, samples


def predict_w(model, samples, process_w, xt_pts):
    """Predict displacement only (drops w_xx output channel)."""
    (w_pred,) = model.predict(xt_pts, samples, processes=[process_w])
    return w_pred[..., 0:1]  # (n_samples, n_points, 1)


if __name__ == "__main__":
    # ── Hyperparameters ───────────────────────────────────────────────────────
    layers = [2, 50, 50, 50, 2]   # 2 inputs (x, t), 2 outputs (w, w_xx)
    n_pde = 1000 #5000 
    n_bc = 100 #300
    n_ic = 100
    vi_batch_size = 64
    num_samples = 200
    num_iterations = 30000
    n_intervals = 3
    F_tip = -0.013
    test_first_interval_only = True

    # ── Load data ─────────────────────────────────────────────────────────────
    data_file = sys.argv[1]
    data_tag = os.path.splitext(os.path.basename(data_file))[0]
    mode_match = re.search(r'_M(\d+)_', data_tag)
    if mode_match is None:
        raise ValueError(f"Could not parse mode count (_M<N>_) from {data_tag}")
    number_of_modes = int(mode_match.group(1))

    (x_full, t_full, w_d_full, w_xx_full,
     x_train, t_train, w_d_train, w_xx_train,
     x_test, t_test, w_d_test,
     beam_params, noise_amplitude, noise_type) = load_data(
        data_file, select_every_nth=20, number_of_modes=number_of_modes)

    L = float(beam_params['L'])
    EI = float(beam_params['EI'])
    mu = float(beam_params['mu'])
    c = 0.0
    P = 0
    f = 2.4

    print(f"Using L={L}, EI={EI}, mu={mu}")

    Y_ref = F_tip * L**3 / EI
    time_concern = (mu * L**4 / EI) ** 0.5
    print(f"Y_ref={Y_ref:.6f}, time_concern={time_concern:.6f}")

    # ── Compute physically-motivated sigmas ──────────────────────────────────
    sigmas = compute_sigmas(L, EI, mu, F_tip)
    print("\nComputed sigma values:")
    for k, v in sigmas.items():
        print(f"  {k:20s}: {v:.6f}")

    print(f"\nw_xx_train stats: min={w_xx_train.min():.4f} max={w_xx_train.max():.4f} "
          f"mean_abs={np.mean(np.abs(w_xx_train)):.4f}")

    n_x_test = len(x_test)
    n_t_test = len(t_test)

    # ── Split time into intervals ─────────────────────────────────────────────
    t_boundaries = np.linspace(t_full.min(), t_full.max(), n_intervals + 1)
    time_intervals = [(t_boundaries[i], t_boundaries[i + 1]) for i in range(n_intervals)]
    print(f"\nTime intervals: {time_intervals}")
    models_info = []

    # IC for first interval comes from data at t=0
    w_ic_next = w_d_full[:, 0:1].astype(np.float32)
    x_ic = x_full.copy()
    ic_sigma_next = None  # first interval uses sigmas['ic_disp']

    for iv, (t_start, t_end) in enumerate(time_intervals):
        print(f"\n{'='*60}")
        print(f"Interval {iv+1}/{n_intervals}: t in [{t_start:.4f}, {t_end:.4f}]")
        print(f"{'='*60}")

        t_mask = (t_train >= t_start) & (t_train <= t_end)
        t_interval = t_train[t_mask]
        w_d_interval = w_d_train[:, t_mask]
        w_xx_d_interval = w_xx_train[:, t_mask]

        model, process_w, samples = train_interval_vi(
            x=x_train, t_interval=t_interval,
            w_d_interval=w_d_interval, w_xx_d_interval=w_xx_d_interval,
            x_ic_pts=x_ic, w_ic_targets=w_ic_next,
            layers=layers, n_pde=n_pde, n_bc=n_bc, n_ic=n_ic,
            sigmas=sigmas,
            vi_batch_size=vi_batch_size, num_samples=num_samples,
            num_iterations=num_iterations,
            ic_sigma=ic_sigma_next,
        )
        models_info.append({
            'model': model, 'process': process_w, 'samples': samples,
            't_start': t_start, 't_end': t_end,
        })

        # IC for next interval: predict w only at t_end
        x_ic_col = x_ic[:, None] if x_ic.ndim == 1 else x_ic
        t_ic_col = np.full_like(x_ic_col, t_end, dtype=np.float32)
        xt_boundary = np.concatenate([x_ic_col.astype(np.float32), t_ic_col], axis=-1)
        w_boundary = predict_w(model, samples, process_w, xt_boundary)
        w_ic_next = np.mean(w_boundary, axis=0).reshape(-1, 1).astype(np.float32)
        ic_sigma_next = float(np.mean(np.std(w_boundary, axis=0)))
        print(f"IC for next interval: mean range [{w_ic_next.min():.6f}, {w_ic_next.max():.6f}], sigma={ic_sigma_next:.6f}")

        if test_first_interval_only:
            print("test_first_interval_only=True — stopping after first interval.")
            break

    # ── Log trained models ────────────────────────────────────────────────────
    model_log = {
        'layers': layers,
        'n_intervals': n_intervals,
        'num_iterations': num_iterations,
        'num_samples': num_samples,
        'vi_batch_size': vi_batch_size,
        'data_file': data_file,
        'time_intervals': time_intervals,
        'intervals': [],
    }
    for iv, mi in enumerate(models_info):
        model_log['intervals'].append({
            'interval': iv,
            't_start': mi['t_start'],
            't_end': mi['t_end'],
            'samples': mi['samples'],
        })
    model_path = os.path.join(RESULTS_DIR, f"{data_tag}_vi_model.npy")
    np.save(model_path, model_log, allow_pickle=True)
    print(f"Model logged to {model_path}")

    # ── Metrics (on test data only) ─────────────────────────────────────────
    w_mean_test = np.zeros((n_x_test, n_t_test))
    w_std_test = np.zeros((n_x_test, n_t_test))

    for mi in models_info:
        t_mask = (t_test >= mi['t_start']) & (t_test <= mi['t_end'])
        t_test_iv = t_test[t_mask]
        if len(t_test_iv) == 0:
            continue
        xx_ti, tt_ti = np.meshgrid(x_test, t_test_iv, indexing='ij')
        xt_pts = np.stack([xx_ti.ravel(), tt_ti.ravel()], axis=-1).astype(np.float32)
        w_pred = predict_w(mi['model'], mi['samples'], mi['process'], xt_pts)
        w_pred_reshape = w_pred[..., 0].reshape([-1, n_x_test, len(t_test_iv)])
        t_indices = np.where(t_mask)[0]
        w_mean_test[:, t_indices] = np.mean(w_pred_reshape, axis=0)
        w_std_test[:, t_indices] = np.std(w_pred_reshape, axis=0)

    target = w_d_test
    width = 3 * w_std_test
    abs_diff = np.abs(target - w_mean_test)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean((target - w_mean_test) ** 2)))
    distance_to_boundary = float(np.mean(np.maximum(abs_diff - width, 0.0)))
    distance_from_boundary = float(np.mean(np.maximum(width - abs_diff, 0.0)))
    boundary_width = float(np.mean(2 * width))
    width_vs_noise_ratio = boundary_width / noise_amplitude if noise_amplitude > 0 else float('inf')
    fraction_outside = float(np.mean(abs_diff > width))

    metrics_txt = (
        f"data_file:              {data_file}\n"
        f"noise_type:             {noise_type}\n"
        f"noise_amplitude:        {noise_amplitude}\n"
        f"n_intervals:            {n_intervals}\n"
        f"num_iterations:         {num_iterations}\n"
        f"num_samples:            {num_samples}\n"
        f"vi_batch_size:          {vi_batch_size}\n"
        f"layers:                 {layers}\n"
        f"\n"
        f"mae:                    {mae:.6f}\n"
        f"rmse:                   {rmse:.6f}\n"
        f"distance_to_boundary:   {distance_to_boundary:.6f}\n"
        f"distance_from_boundary: {distance_from_boundary:.6f}\n"
        f"boundary_width:         {boundary_width:.6f}\n"
        f"width_vs_noise_ratio:   {width_vs_noise_ratio:.3f}\n"
        f"fraction_outside:       {fraction_outside:.4f}\n"
    )
    print(f"\n{'='*60}")
    print("METRICS")
    print(f"{'='*60}")
    print(metrics_txt)

    metrics_path = os.path.join(RESULTS_DIR, f"{data_tag}_vi_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(metrics_txt)
    print(f"Metrics saved to {metrics_path}")

    # ── Plot 1: Spacetime heatmaps on dense regular grid ────────────────────
    t_max = time_intervals[-1][1]
    n_x_hm, n_t_hm = 100, 200
    x_hm = np.linspace(0, L, n_x_hm)
    t_hm = np.linspace(0, t_max, n_t_hm)

    w_mean_hm = np.zeros((n_x_hm, n_t_hm))
    w_std_hm = np.zeros((n_x_hm, n_t_hm))
    for mi in models_info:
        t_mask = (t_hm >= mi['t_start']) & (t_hm <= mi['t_end'])
        t_hm_iv = t_hm[t_mask]
        if len(t_hm_iv) == 0:
            continue
        xx_ti, tt_ti = np.meshgrid(x_hm, t_hm_iv, indexing='ij')
        xt_pts = np.stack([xx_ti.ravel(), tt_ti.ravel()], axis=-1).astype(np.float32)
        w_pred = predict_w(mi['model'], mi['samples'], mi['process'], xt_pts)
        w_pred_reshape = w_pred[..., 0].reshape([-1, n_x_hm, len(t_hm_iv)])
        t_indices = np.where(t_mask)[0]
        w_mean_hm[:, t_indices] = np.mean(w_pred_reshape, axis=0)
        w_std_hm[:, t_indices] = np.std(w_pred_reshape, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(w_mean_hm, aspect='auto', origin='lower',
                         extent=[0, t_max, 0, L], cmap='RdBu_r',
                         vmin=-0.25, vmax=0.25)
    for i, (t_start, t_end) in enumerate(time_intervals):
        if i > 0:
            axes[0].axvline(x=t_start, color='white', linestyle='--', linewidth=2, alpha=0.7)
    plt.colorbar(im0, ax=axes[0], label='Center')
    axes[0].set_xlabel('Time t', labelpad=5)
    axes[0].set_ylabel('Position x', labelpad=5)
    axes[0].set_xticks([0.00, 0.05, 0.10])
    axes[0].set_title('VI Mean Prediction', pad=15)

    im1 = axes[1].imshow(3 * w_std_hm, aspect='auto', origin='lower',
                         extent=[0, t_max, 0, L], cmap='Oranges',
                         vmin=0, vmax=0.25)
    for i, (t_start, t_end) in enumerate(time_intervals):
        if i > 0:
            axes[1].axvline(x=t_start, color='white', linestyle='--', linewidth=2, alpha=0.7)
    plt.colorbar(im1, ax=axes[1], label='Width')
    axes[1].set_xlabel('Time t', labelpad=5)
    axes[1].set_ylabel('Position x', labelpad=5)
    axes[1].set_xticks([0.00, 0.05, 0.10])
    axes[1].set_title('VI Uncertainty', pad=15)

    im2 = axes[2].imshow(w_d_train, aspect='auto', origin='lower',
                         extent=[0, t_max, 0, L], cmap='RdBu_r',
                         vmin=-0.25, vmax=0.25)
    plt.colorbar(im2, ax=axes[2], label='Displacement')
    axes[2].set_xlabel('Time t', labelpad=5)
    axes[2].set_ylabel('Position x', labelpad=5)
    axes[2].set_xticks([0.00, 0.05, 0.10])
    axes[2].set_title('Original Noisy Data', pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{data_tag}_vi_heatmap.pdf"))
    plt.show()

    # ── Plot 2: Comparison at time slices ─────────────────────────────────────
    time_points = np.stack([
        t_train[0],
        (t_train[1] + t_train[2]) / 2,
        t_train[4],
        (t_train[5] + t_train[6]) / 2,
        t_train[8],
        (t_train[9] + t_train[10]) / 2,
    ])

    n_x_plot = len(x_full)

    # Batch-predict for all time_points: group by interval
    tp_mean = np.zeros((n_x_plot, len(time_points)))
    tp_std = np.zeros((n_x_plot, len(time_points)))
    tp_model_idx = np.full(len(time_points), len(models_info) - 1, dtype=int)

    for j, mi in enumerate(models_info):
        tp_mask = (time_points >= mi['t_start']) & (time_points <= mi['t_end'])
        tp_iv = time_points[tp_mask]
        if len(tp_iv) == 0:
            continue
        tp_model_idx[tp_mask] = j
        xx_ti, tt_ti = np.meshgrid(x_full, tp_iv, indexing='ij')
        xt_pts = np.stack([xx_ti.ravel(), tt_ti.ravel()], axis=-1).astype(np.float32)
        w_pred = predict_w(mi['model'], mi['samples'], mi['process'], xt_pts)
        w_pred_reshape = w_pred[..., 0].reshape([-1, n_x_plot, len(tp_iv)])
        tp_indices = np.where(tp_mask)[0]
        tp_mean[:, tp_indices] = np.mean(w_pred_reshape, axis=0)
        tp_std[:, tp_indices] = np.std(w_pred_reshape, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, t_val in enumerate(time_points):
        if idx >= len(axes):
            break
        ax = axes[idx]

        w_pred_mean = tp_mean[:, idx]
        w_pred_std = tp_std[:, idx]
        w_pred_upper = w_pred_mean + 3 * w_pred_std
        w_pred_lower = w_pred_mean - 3 * w_pred_std

        # Plot prediction with CI
        ax.fill_between(x_full, w_pred_lower, w_pred_upper,
                        alpha=0.3, color='blue', label='95% CI')
        ax.plot(x_full, w_pred_mean, 'k-', linewidth=2, label='Mean')
        ax.plot(x_full, w_pred_lower, 'k--', linewidth=1, alpha=0.7)
        ax.plot(x_full, w_pred_upper, 'k--', linewidth=1, alpha=0.7)

        # Plot full ground truth (all x)
        try:
            t_full_idx = np.where(t_full == t_val)[0][0]
        except IndexError:
            t_full_idx = np.argmin(np.abs(t_full - t_val))
            print(f"Warning: t_val={t_val:.6f} not in t_full, using closest t={t_full[t_full_idx]:.6f}")
        ax.scatter(x_full, w_d_full[:, t_full_idx], c='#ff4500', s=20,
                   edgecolors='k', linewidths=0.2, label='Test', zorder=4)

        # Plot IC data at t=0
        if t_val == t_full[0]:
            ax.scatter(x_full, w_d_full[:, 0], c='blue', s=25,
                       edgecolors='k', linewidths=0.2, label='IC', zorder=5)

        # Plot training data points
        if t_val in t_train:
            t_train_idx = np.where(t_train == t_val)[0][0]
            ax.scatter(x_train, w_d_train[:, t_train_idx], c='#00ff55', s=55,
                        marker='s', edgecolors='k', linewidths=0.3, label='Train', zorder=5)

        ax.set_xlabel('Position x')
        ax.set_ylabel('Displacement w')
        ax.set_title(f't = {t_val:.3f} (VI #{tp_model_idx[idx]+1})')
        ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color='black', linewidth=2, label='Mean'),
        Line2D([0], [0], color='black', linewidth=1, linestyle='--', label='Bounds'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=8, markeredgecolor='black', markeredgewidth=0.2, label='IC data'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4500',
               markersize=8, markeredgecolor='black', markeredgewidth=0.2, label='Test data'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ff55',
               markersize=8, markeredgecolor='black', markeredgewidth=0.3, label='Train data'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(os.path.join(RESULTS_DIR, f"{data_tag}_vi_intervals.pdf"),
                bbox_inches='tight')
    plt.show()
