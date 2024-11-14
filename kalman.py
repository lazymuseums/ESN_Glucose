import numpy as np
import matplotlib.pyplot as plt


def kalman_filter(signal, process_variance, measurement_variance):
    # Initialize Kalman filter parameters
    n = len(signal)
    x_hat = np.zeros(n)  # estimated state
    P = np.zeros(n)  # state covariance matrix
    x_hat_minus = np.zeros(n)  # predicted state
    P_minus = np.zeros(n)  # predicted state covariance matrix
    K = np.zeros(n)  # Kalman gain

    # Initial conditions
    x_hat[0] = signal[0]
    P[0] = 1.0

    for k in range(1, n):
        # prediction
        x_hat_minus[k] = x_hat[k - 1]
        P_minus[k] = P[k - 1] + process_variance

        # update
        K[k] = P_minus[k] / (P_minus[k] + measurement_variance)
        x_hat[k] = x_hat_minus[k] + K[k] * (signal[k] - x_hat_minus[k])
        P[k] = (1 - K[k]) * P_minus[k]

    return x_hat


