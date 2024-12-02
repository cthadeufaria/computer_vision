import numpy as np
import matplotlib.pyplot as plt

# UKF Parameters
alpha = 1e-3  # Scaling factor
beta = 2      # Optimal for Gaussian distributions
kappa = 0     # Secondary scaling parameter

# Process noise covariance
# Q = np.diag([0.5**2, 0.05**2])  # Linear and angular velocity noise variances
Q = np.diag([0.5**2, 0.5**2, 0.05**2])

# Measurement noise covariance
R = np.diag([0.5**2, 0.05**2])  # Noise in velocity measurements

# Initial state and covariance
x = np.array([3, 2, 0])  # Initial position (x, y) and orientation (theta)
P = np.eye(3) * 0.1  # Initial covariance

# Ground truth data
t = np.linspace(0, 10, 100)  # Time vector
v_true = 1.0 + 0.2 * np.sin(0.5 * t)  # True linear velocity
w_true = 0.1 + 0.05 * np.cos(0.5 * t)  # True angular velocity

# Simulated noisy measurements
v_meas = v_true + np.random.normal(0, np.sqrt(Q[0, 0]), len(t))
w_meas = w_true + np.random.normal(0, np.sqrt(Q[1, 1]), len(t))

# Unscented Kalman Filter Implementation
def generate_sigma_points(x, P, alpha, beta, kappa):
    n = len(x)
    lambda_ = alpha**2 * (n + kappa) - n
    sigma_points = np.zeros((2 * n + 1, n))
    Wm = np.zeros(2 * n + 1)  # Weights for mean
    Wc = np.zeros(2 * n + 1)  # Weights for covariance
    sqrt_P = np.linalg.cholesky((n + lambda_) * P)

    sigma_points[0] = x
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    for i in range(n):
        sigma_points[i + 1] = x + sqrt_P[:, i]
        sigma_points[i + 1 + n] = x - sqrt_P[:, i]
        Wm[i + 1] = Wm[i + 1 + n] = 1 / (2 * (n + lambda_))
        Wc[i + 1] = Wc[i + 1 + n] = Wm[i + 1]

    return sigma_points, Wm, Wc

def predict_sigma_points(sigma_points, dt, v, w):
    sigma_points_pred = np.zeros_like(sigma_points)
    for i, sp in enumerate(sigma_points):
        x, y, theta = sp
        x_pred = x + v * np.cos(theta) * dt
        y_pred = y + v * np.sin(theta) * dt
        theta_pred = theta + w * dt
        sigma_points_pred[i] = [x_pred, y_pred, theta_pred]
    return sigma_points_pred

def ukf_update(x, P, z, v, w, Q, R, dt):
    # Generate sigma points
    sigma_points, Wm, Wc = generate_sigma_points(x, P, alpha, beta, kappa)

    # Predict sigma points
    sigma_points_pred = predict_sigma_points(sigma_points, dt, v, w)

    # Predicted state mean
    x_pred = np.sum(Wm[:, None] * sigma_points_pred, axis=0)

    # Predicted state covariance
    # P_pred = Q + sum(Wc[i] * np.outer(sigma_points_pred[i] - x_pred,
    #                                    sigma_points_pred[i] - x_pred) for i in range(len(Wc)))
    
    # Predicted state covariance
    P_pred = sum(Wc[i] * np.outer(sigma_points_pred[i] - x_pred,
                                sigma_points_pred[i] - x_pred) for i in range(len(Wc)))
    P_pred += Q  # Add process noise

    # Predicted measurements
    z_pred = np.array([v, w])  # Predicted measurement
    S = R + P_pred[:2, :2]  # Measurement covariance

    # Kalman gain
    K = P_pred[:, :2] @ np.linalg.inv(S)

    # Update state and covariance
    y = z - z_pred  # Measurement residual
    x_update = x_pred + K @ y
    P_update = P_pred - K @ S @ K.T

    return x_update, P_update

# UKF Loop
dt = t[1] - t[0]  # Time step
estimates = []
variances = []
for i in range(len(t)):
    z = np.array([v_meas[i], w_meas[i]])
    x, P = ukf_update(x, P, z, v_meas[i], w_meas[i], Q, R, dt)
    estimates.append(x)
    variances.append(np.diag(P))

# Convert to arrays
estimates = np.array(estimates)
variances = np.array(variances)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, estimates[:, 0], label="Estimated x")
plt.plot(t, 3 + v_true * np.cos(w_true * t), label="True x")
plt.fill_between(t, estimates[:, 0] - variances[:, 0], estimates[:, 0] + variances[:, 0], alpha=0.2)
plt.legend()
plt.title("Position (x) Estimation")

plt.subplot(3, 1, 2)
plt.plot(t, estimates[:, 1], label="Estimated y")
plt.plot(t, 2 + v_true * np.sin(w_true * t), label="True y")
plt.fill_between(t, estimates[:, 1] - variances[:, 1], estimates[:, 1] + variances[:, 1], alpha=0.2)
plt.legend()
plt.title("Position (y) Estimation")

plt.subplot(3, 1, 3)
plt.plot(t, estimates[:, 2], label="Estimated θ")
plt.plot(t, w_true * t, label="True θ")
plt.fill_between(t, estimates[:, 2] - variances[:, 2], estimates[:, 2] + variances[:, 2], alpha=0.2)
plt.legend()
plt.title("Orientation (θ) Estimation")

plt.tight_layout()
plt.show()
