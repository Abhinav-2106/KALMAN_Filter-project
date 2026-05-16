print("Program started")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf

# ------------------------------------------------------------------
# Pull 6 months of daily AAPL closes and flatten to a 1-D array
# ------------------------------------------------------------------
stock_symbol = "JPM"
stock_data = yf.download(stock_symbol, period="6mo", interval="1d")
close_prices = stock_data["Close"].dropna().to_numpy().flatten()

# Make sure the plots folder exists before we try to save anything
os.makedirs("plots", exist_ok=True)

np.random.seed(42)
mean_price   = np.mean(close_prices)
noise_levels = [0.01, 0.03, 0.05]   # 1 %, 3 %, 5 % noise


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def moving_average(data, window):
    """Simple rolling mean — NaNs fill the warm-up period."""
    return pd.Series(data).rolling(window=window).mean().values


def run_kalman_filter(observed_prices, q_value, r_value):
    """
    Two-state Kalman filter (position + velocity).
    Returns the filtered price series and the raw innovations.
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # Start from the first observation; assume zero velocity
    kf.x = np.array([float(observed_prices[0]), 0.0])

    # x_{k+1} = F * x_k  (constant-velocity model)
    kf.F = np.array([[1, 1],
                     [0, 1]])

    # We only observe the position, not the velocity
    kf.H = np.array([[1, 0]])

    kf.P *= 100                          # fairly uncertain about the initial state
    kf.Q = np.eye(2) * q_value          # process noise (same for both states)
    kf.R = np.array([[r_value]])         # measurement noise

    filtered_prices = []
    innovations     = []

    for price in observed_prices:
        kf.predict()

        # Innovation = how far the new measurement is from our prediction
        innovation = float(price - (kf.H @ kf.x)[0])
        innovations.append(innovation)

        kf.update(price)
        filtered_prices.append(float(kf.x[0]))

    return np.array(filtered_prices), np.array(innovations)


# ------------------------------------------------------------------
# Main experiment loop — one run per noise level
# ------------------------------------------------------------------

# Q and R grids to search over (R is defined relative to sigma later)
q_values = [0.001, 0.01, 0.1, 1.0]

for noise_percent in noise_levels:

    sigma = noise_percent * mean_price
    noise = np.random.normal(loc=0, scale=sigma, size=len(close_prices))
    noisy_prices = close_prices + noise

    # R candidates expressed as multiples of the noise variance
    r_values = [
        0.1 * sigma**2,
        0.5 * sigma**2,
        sigma**2,
        2.0 * sigma**2,
    ]

    # Grid search: find the (Q, R) pair with the lowest MSE
    mse_matrix = np.zeros((len(q_values), len(r_values)))

    best_mse            = float("inf")
    best_filtered_prices = None
    best_innovations    = None
    best_q              = None
    best_r              = None

    for i, q in enumerate(q_values):
        for j, r in enumerate(r_values):

            filtered_prices, innovations = run_kalman_filter(noisy_prices, q, r)
            mse = mean_squared_error(close_prices, filtered_prices)
            mse_matrix[i, j] = mse

            if mse < best_mse:
                best_mse             = mse
                best_filtered_prices = filtered_prices
                best_innovations     = innovations
                best_q               = q
                best_r               = r

    # Grab the best results for this noise level
    filtered_prices = best_filtered_prices
    innovations     = best_innovations

    kalman_mse  = mean_squared_error(close_prices, filtered_prices)
    kalman_corr = pearsonr(close_prices, filtered_prices)[0]

    # Best moving-average window (5 / 10 / 20 days)
    best_ma_mse = float("inf")
    best_window = None

    for window in [5, 10, 20]:
        ma        = moving_average(noisy_prices, window)
        valid_idx = ~np.isnan(ma)
        mse       = mean_squared_error(close_prices[valid_idx], ma[valid_idx])

        if mse < best_ma_mse:
            best_ma_mse = mse
            best_window = window

    # Print a tidy summary for this noise level
    tag = f"{int(noise_percent * 100)}%"
    print(f"\n=== Noise Level: {tag} ===")
    print(f"  Best Q              : {best_q}")
    print(f"  Best R              : {best_r:.4f}")
    print(f"  Kalman MSE          : {kalman_mse:.4f}")
    print(f"  Kalman Correlation  : {kalman_corr:.4f}")
    print(f"  Best MA window      : {best_window} days  (MSE: {best_ma_mse:.4f})")
    # ---- Plot: True vs Noisy vs Filtered ------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        close_prices,
        label="True Prices",
        linewidth=2
    )

    plt.plot(
        noisy_prices,
        label="Noisy Prices",
        alpha=0.6
    )

    plt.plot(
        filtered_prices,
        label="Kalman Filter Estimate",
        linewidth=2
    )

    plt.title(f"Kalman Filtering — {tag} Noise")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        f"plots/price_comparison_{int(noise_percent * 100)}.png",
        bbox_inches="tight"
    )

    plt.close()
    # ---- Plot 1: Innovation sequence --------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(innovations, color="steelblue", linewidth=0.9)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    plt.title(f"Innovation Sequence — {tag} Noise")
    plt.xlabel("Time Step")
    plt.ylabel("Innovation (residual)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"plots/innovations_{int(noise_percent * 100)}.png", bbox_inches="tight")
    plt.close()

    # ---- Plot 2: Autocorrelation of innovations ---------------------
    plt.figure(figsize=(10, 4))
    plot_acf(innovations, lags=20)
    plt.title(f"Innovation ACF — {tag} Noise")
    plt.tight_layout()
    plt.savefig(f"plots/acf_{int(noise_percent * 100)}.png", bbox_inches="tight")
    plt.close()

    # ---- Plot 3: MSE heat-map (Q vs R sensitivity) ------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mse_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=[round(r, 2) for r in r_values],
        yticklabels=q_values,
        cmap="YlOrRd",
    )
    plt.title(f"MSE Sensitivity — {tag} Noise")
    plt.xlabel("R Values")
    plt.ylabel("Q Values")
    plt.tight_layout()
    plt.savefig(f"plots/sensitivity_{int(noise_percent * 100)}.png", bbox_inches="tight")
    plt.close()

print("\nFinished running all experiments.")