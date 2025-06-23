
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve
from scipy.stats import norm
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from scipy.interpolate import interp1d


def plot_pde_bc_loss (pde_hist, bc_hist):
    plt.figure(figsize=(12,8))
    plt.plot(pde_hist, label='PDE Loss', color='blue')
    plt.plot(bc_hist, label='BC Loss', color='orange')
    plt.yscale('log')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_loss_comparison(results, metric="loss", case_name="Case 3"):
    """
    results: list of dicts with keys:
      - 'name': str
      - 'include_boundaries': bool
      - 'adaptive': bool
      - 'loss_hist': list
      - 'pde_loss_hist': list
      - 'bc_loss_hist': list
      - 'l2_hist': list
    metric: "loss", "pde", "bc", or "l2"
    """
    plt.figure(figsize=(10, 6))
    for result in results:
        if metric == "loss":
            y = result['loss_hist']
            label = f"{result['name']} | BC {'✓' if result['include_boundaries'] else '✗'}"
        elif metric == "pde":
            y = result['pde_loss_hist']
            label = f"{result['name']} | PDE loss | BC {'✓' if result['include_boundaries'] else '✗'}"
        elif metric == "bc":
            y = result['bc_loss_hist']
            label = f"{result['name']} | BC loss | BC {'✓' if result['include_boundaries'] else '✗'}"
        elif metric == "l2":
            y = result['l2_hist']
            label = f"{result['name']} | L2 error | BC {'✓' if result['include_boundaries'] else '✗'}"
        else:
            continue
        plt.plot(y, label=label)
    
    titles = {
        "loss": "Total Loss (PDE + BC)",
        "pde": "PDE Residual Loss",
        "bc": "Boundary Condition Loss",
        "l2": "L2 Relative Error on Test Set"
    }
    plt.title(f"{titles.get(metric, 'Loss')} | {case_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_solution_snapshots_compare(
    x_test: np.ndarray,
    u_test: np.ndarray,
    pred_hist_adapt: list[np.ndarray],
    pred_hist_nonadapt: list[np.ndarray],
    ep_hist_pred: list[int],
    n_cols: int = 4
):
    """
    Grid of solution snapshots comparing adaptive vs non-adaptive PINN.

    Each subplot at epoch ep_hist_pred[i] shows:
      • True u(x)      (black solid)
      • Adaptive û(x) (blue dashed)
      • Non‐adaptive û(x) (red dash‐dot)

    Legend is placed in 3 columns above the grid.
    """
    import math
    x = x_test.flatten()
    y_true = u_test.flatten()

    # ensure we only loop over the common length
    n_snap = min(len(pred_hist_adapt),
                 len(pred_hist_nonadapt),
                 len(ep_hist_pred))
    if n_snap == 0:
        raise ValueError("No snapshots to plot!")

    # set up grid
    n_rows = math.ceil(n_snap / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4*n_cols, 3*n_rows),
                             squeeze=False)

    # plot each panel
    for idx in range(n_snap):
        ax = axes[idx//n_cols][idx % n_cols]
        ep = ep_hist_pred[idx]
        y_ad = pred_hist_adapt[idx].flatten()
        y_na = pred_hist_nonadapt[idx].flatten()

        # only label on the first panel; we'll pull these handles for the global legend
        if idx == 0:
            ax.plot(x, y_true, 'k-',  lw=1.5, label='True')
            ax.plot(x, y_ad,   'b--', lw=1.5, label='Adaptive')
            ax.plot(x, y_na,  'r-.', lw=1.5, label='Non-adaptive')
        else:
            ax.plot(x, y_true, 'k-',  lw=1.5)
            ax.plot(x, y_ad,   'b--', lw=1.5)
            ax.plot(x, y_na,  'r-.', lw=1.5)

        ax.set_title(f"Epoch {ep}")
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.grid(alpha=0.3)

    # delete any extra axes
    for j in range(n_snap, n_rows*n_cols):
        fig.delaxes(axes[j//n_cols][j % n_cols])

    # grab handles from the first axis
    handles, labels = axes[0][0].get_legend_handles_labels()

    # place a 3-column legend above all subplots
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        fontsize=14
    )
    # leave a bit less room at top for the legend
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def plot_continuous_collocation_error_evolution(
    pos_hist: list[np.ndarray],
    ep_hist: list[int],
    pred_hist: list[np.ndarray],
    res_hist: list[np.ndarray],
    u_test_fn: callable  # function that gives u_true(x)
):
    """
    Plots a continuous heatmap of the evolution of two error metrics over a common spatial grid.
    
    For each epoch, the collocation error and PDE residual (both taken at the collocation nodes)
    are interpolated onto a common x-grid and then plotted as two heatmaps:
      - Top: absolute PDE residual error.
      - Bottom: solution prediction error.
    
    Parameters
    ----------
    pos_hist : list of np.ndarray
        Node positions at each epoch.
    ep_hist : list of int
        Epoch numbers corresponding to pos_hist.
    pred_hist : list of np.ndarray
        Model predictions at the collocation nodes per epoch.
    res_hist : list of np.ndarray
        PDE residuals at the collocation nodes per epoch.
    u_test_fn : callable
        Function returning u_true(x) for a given array x.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Combine all positions to get a global common grid
    all_x = np.concatenate([p.flatten() for p in pos_hist])
    x_min, x_max = all_x.min(), all_x.max()    
    x_grid = np.linspace(x_min, x_max, 500)

    # Prepare arrays for the two error metrics, one row per epoch.
    residual_field = []
    sol_error_field = []
    
    for p, pred, res in zip(pos_hist, pred_hist, res_hist):
        x_i = p.flatten()
        # Interpolate PDE residual (absolute value) onto common grid.
        r_i = np.abs(res.flatten())
        r_interp = np.interp(x_grid, x_i, r_i, left=np.nan, right=np.nan)
        residual_field.append(r_interp)
        
        # Compute solution error: compare prediction at collocation nodes with u_true.
        # Here we interpolate the prediction onto the common grid.
        u_pred_i = np.interp(x_grid, x_i, pred.flatten(), left=np.nan, right=np.nan)
        u_true_grid = u_test_fn(x_grid)
        sol_err = np.abs(u_pred_i - u_true_grid)
        sol_error_field.append(sol_err)
    
    residual_field = np.array(residual_field)
    sol_error_field = np.array(sol_error_field)
    epochs = np.array(ep_hist)
    
    # Plot the continuous heatmaps.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    
    # Plot residual field (top)
    im1 = ax1.imshow(
        residual_field,
        aspect='auto',
        extent=[x_min, x_max, epochs[-1], epochs[0]],
        cmap='magma'
    )
    ax1.set_title("|PDE residual| evolution")
    ax1.set_ylabel("Epoch")
    fig.colorbar(im1, ax=ax1, label="|r|")
    
    # Plot solution error field (bottom)
    im2 = ax2.imshow(
        sol_error_field,
        aspect='auto',
        extent=[x_min, x_max, epochs[-1], epochs[0]],
        cmap='magma'
    )
    ax2.set_title("Solution error evolution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Epoch")
    fig.colorbar(im2, ax=ax2, label="|u_pred - u_true|")
    
    plt.tight_layout()
    plt.show()

def plot_discrete_collocation_error_evolution(
    pos_hist: list[np.ndarray],
    ep_hist: list[int],
    pred_hist: list[np.ndarray],
    res_hist: list[np.ndarray],
    u_test_fn: callable  # function that gives u_true(x)
):
    """
    Discrete scatter-heatmap of collocation node positions over training epochs,
    colored by (1) |PDE residual| and (2) |solution error|.

    Parameters
    ----------
    pos_hist : list of np.ndarray
        Node positions at each epoch (shape varies).
    ep_hist : list of int
        Epochs corresponding to pos_hist.
    pred_hist : list of np.ndarray
        Predictions at pos_hist per epoch.
    res_hist : list of np.ndarray
        Residuals at pos_hist per epoch.
    u_test_fn : callable
        Function that returns u_true(x) for arbitrary x.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    for x_i, ep, u_pred_i, r_i in zip(pos_hist, ep_hist, pred_hist, res_hist):
        x_i = x_i.flatten()
        u_true_i = u_test_fn(x_i)
        err_i = np.abs(u_pred_i.flatten() - u_true_i)
        r_abs = np.abs(r_i.flatten())

        # Scatter plots (epoch on y-axis, x on x-axis, color by value)
        ax1.scatter(x_i, np.full_like(x_i, ep), c=r_abs, s=20, cmap='magma', alpha=0.8)
        ax2.scatter(x_i, np.full_like(x_i, ep), c=err_i, s=20, cmap='magma', alpha=0.8)

    for ax, title, label in zip(
        (ax1, ax2),
        ("|PDE residual|", "Solution error"),
        ("|r|", "|uₚᵣₑd − u_true|")
    ):
        ax.set_xlabel('x')
        ax.set_ylabel('Epoch')
        ax.set_title(title)
        ax.grid(ls='--', alpha=0.3)

    fig.colorbar(ax1.collections[0], ax=ax1, label='|r|')
    fig.colorbar(ax2.collections[0], ax=ax2, label='|uₚᵣₑd − u_true|')

    plt.tight_layout()
    plt.show()

def plot_collocation_error_evolution(
    pos_hist, ep_hist,
    model,
    pde_residual_fn,
    x_test, u_test,
    device='cpu'
):
    """
    Scatter‐heatmap of collocation nodes over epochs,
    colored by (1) |PDE residual| and (2) |solution error|.
    
    Parameters
    ----------
    pos_hist : list of 1D np.ndarray
        Node‐positions at each recorded epoch.
    ep_hist : list of ints
        Corresponding epochs.
    model : nn.Module
        Your trained PINN (should be in eval() mode).
    pde_residual_fn : callable
        Residual function: res = pde_residual_fn(model, x_tensor).
    x_test, u_test : np.ndarray
        Dense test grid and true solution on it.
    device : str
        'cpu' or 'cuda'.
    """
    model.eval()
    # build interpolant for analytic u
    interp_u = interp1d(x_test.flatten(), u_test.flatten(), kind='cubic',
                        fill_value='extrapolate')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18,10))
    
    for pts, epoch in zip(pos_hist, ep_hist):
        # prepare tensor for residual
        x_dom = pts[:,None]
        x_t = torch.tensor(x_dom, dtype=torch.float32, device=device, requires_grad=True)
        # compute abs‐residual
        with torch.enable_grad():
            r = pde_residual_fn(model, x_t).abs().detach().cpu().numpy().flatten()
        # compute solution error
        with torch.no_grad():
            u_pred = model(torch.tensor(x_dom, dtype=torch.float32, device=device)).cpu().numpy().flatten()
        u_true = interp_u(pts)
        sol_err = np.abs(u_pred - u_true)
        
        # scatter for residual
        sc1 = ax1.scatter(pts, np.full_like(pts, epoch),
                          c=r, s=20, cmap='magma', alpha=0.8)
        # scatter for sol‐error
        sc2 = ax2.scatter(pts, np.full_like(pts, epoch),
                          c=sol_err, s=20, cmap='magma', alpha=0.8)
    
    # labels & colorbars
    for ax, title in zip((ax1, ax2), ("|PDE residual|", "Solution error")):
        ax.set_xlabel('x')
        ax.set_ylabel('Epoch')
        ax.set_title(title)
        ax.grid(ls='--', alpha=0.3)
    fig.colorbar(sc1, ax=ax1, label='|r|')
    fig.colorbar(sc2, ax=ax2, label='|uₚᵣₑd–u_true|')
    
    plt.tight_layout()
    plt.show()


# --- Plotting utils (unchanged) ---
def plot_training_and_solution(loss_hist, l2_hist, t_test, u_test, model):
    model.eval()
    with torch.no_grad():
        u_pred = model(torch.tensor(t_test, dtype=torch.float32, device=device)).cpu().numpy()
    plt.figure(figsize=(18,6))
    ax1 = plt.subplot(1,3,1)
    ax1.plot(loss_hist); ax1.set_yscale('log'); ax1.set_title('Training Loss', fontsize=14)
    ax2 = plt.subplot(1,3,2)
    ax2.plot(l2_hist); ax2.set_yscale('log'); ax2.set_title('Test Loss', fontsize=14)
    ax3 = plt.subplot(1,3,3)
    ax3.plot(t_test,u_test,'k-',label='True')
    ax3.plot(t_test,u_pred,'r--',label='Pred'); ax3.set_title('Solution')
    ax3.legend(); plt.tight_layout(); plt.show()

def plot_test_error(l2_hist_ad, l2_hist_na, epochs_ad, epochs_na):
    plt.figure(figsize=(10,5))
    plt.plot(epochs_ad, l2_hist_ad, 'b-', label='Adaptive PINN')
    plt.plot(epochs_na, l2_hist_na, 'r--', label='Non-adaptive PINN')
    plt.yscale('log')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('L2 Error', fontsize=14)
    plt.legend()
    plt.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_collocation_evolution(pos_hist, ep_hist):
    plt.figure(figsize=(18,6))
    for pts,epoch in zip(pos_hist,ep_hist):
        plt.scatter(pts, np.full_like(pts,epoch),s=5,alpha=0.6)
    plt.xlabel('x', fontsize=14); plt.ylabel('Epoch', fontsize=14); plt.title('Collocation Evolution',  fontsize=16)
    plt.tight_layout(); plt.show()


def plot_adaptation_density(
    model, x_dom_before, x_dom_after,
    pde_residual_fn, device='cpu',
    smoothing_window=5, density_bins=30, epoch=None
):
    """
    Visualize PDE residual and collocation-point densities before/after adaptation.

    - Smoothed line of |PDE residual| over pre-adaptation points.
    - Filled density plots for x_dom_before (black) and x_dom_after (red).

    Parameters:
    - model: PINN (in eval mode).
    - x_dom_before: np.ndarray (n_dom,1)
    - x_dom_after:  np.ndarray (n_dom,1)
    - pde_residual_fn: function(model, x_tensor) -> residual tensor
    - device: 'cpu' or 'cuda'
    - smoothing_window: int for moving-average smoothing
    - density_bins: int number of histogram bins
    """
    # Prepare and sort
    x = x_dom_before.flatten()
    idx = np.argsort(x)
    x_sorted = x[idx]

    # Compute residual
    x_t = torch.tensor(x_sorted[:, None], dtype=torch.float32, device=device).requires_grad_(True)
    r_t = pde_residual_fn(model, x_t).abs()       # no torch.no_grad here
    r = r_t.detach().cpu().numpy().flatten()

    # Smooth
    if smoothing_window > 1:
        kern = np.ones(smoothing_window) / smoothing_window
        r = np.convolve(r, kern, mode='same')

    # Densities
    bins = np.linspace(0.0, 1.0, density_bins+1)
    d_before, _ = np.histogram(x_dom_before.flatten(), bins=bins, density=True)
    d_after,  _ = np.histogram(x_dom_after.flatten(),  bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(x_sorted, r, color='black', lw=2, label='Smoothed |PDE residual|')
    ax1.set_xlabel('x'); ax1.set_ylabel('Residual', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.fill_between(centers, d_before, color='black', alpha=0.3, label='Density before')
    ax2.fill_between(centers, d_after,  color='red',   alpha=0.3, label='Density after')
    ax2.set_ylabel('Density')
    ax2.set_ylim(0.25, None)
    ax2.tick_params(axis='y', color='tab:gray')

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')

    ax1.grid(ls='--', alpha=0.3)
    plt.title(f"Adaptive Node Movement at epoch: {epoch}")
    plt.tight_layout()
    # save plot
    plt.savefig(f"plots/node_movement_at-{epoch}.png", dpi=300)
    plt.show()




def plot_spatial_loss_evolution(pde_grid, l2_grid, x_eval_grid, ep_hist):
    pde_arr = np.array(pde_grid)
    l2_arr = np.array(l2_grid)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    im1 = ax1.imshow(
        np.flipud(pde_arr),
        aspect='auto',
        extent=[x_eval_grid.min(), x_eval_grid.max(), ep_hist[0], ep_hist[-1]],
        cmap='magma',
        # norm=plt.matplotlib.colors.LogNorm(vmin=1e-40, vmax=1e1)
        norm=plt.matplotlib.colors.LogNorm()

    )
    ax1.set_title("PDE Residual Loss |r(x)|²", fontsize=16)
    ax1.set_ylabel("Epoch", fontsize=14)
    fig.colorbar(im1, ax=ax1, label="PDE Loss")

    im2 = ax2.imshow(
        np.flipud(l2_arr),
        aspect='auto',
        extent=[x_eval_grid.min(), x_eval_grid.max(), ep_hist[0], ep_hist[-1]],
        cmap='magma',
        # norm=plt.matplotlib.colors.LogNorm(vmin=1e-40, vmax=1e1)
        norm=plt.matplotlib.colors.LogNorm()
    )
    ax2.set_title("Test L2 Loss |u_pred(x) - u_true(x)|²", fontsize=16)
    ax2.set_xlabel("x", fontsize=14)
    ax2.set_ylabel("Epoch", fontsize=14)
    fig.colorbar(im2, ax=ax2, label="L2 Error")

    plt.tight_layout()
    plt.show()