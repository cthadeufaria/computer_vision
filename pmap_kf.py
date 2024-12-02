import numpy as np
import matplotlib.pyplot as plt

# Time and simulation parameters
dt = 0.1  # Time step
steps = 100  # Number of steps

# Initial state [x, y, theta]
x_true = np.array([3, 2, 0])  # Ground truth
x_est = np.array([3, 2, 0])  # Kalman Filter estimate
P = np.eye(3) * 0.1  # Initial covariance

# Process noise covariance
Q = np.diag([0.5**2, 0.5**2, 0.05**2])

# Measurement noise covariance
R = np.diag([0.5**2, 0.05**2])  # Noise in v and omega measurements

# Ground truth motion data
v_true = 1.0 + 0.2 * np.sin(0.2 * np.arange(steps) * dt)
w_true = 0.1 + 0.05 * np.cos(0.2 * np.arange(steps) * dt)

# Simulated noisy measurements
v_meas = v_true + np.random.normal(0, np.sqrt(R[0, 0]), steps)
w_meas = w_true + np.random.normal(0, np.sqrt(R[1, 1]), steps)

# Storage for plotting
true_trajectory = []
estimated_trajectory = []
variances = []

# Kalman Filter loop
for k in range(steps):
    # Ground truth update
    x_true[0] += v_true[k] * np.cos(x_true[2]) * dt
    x_true[1] += v_true[k] * np.sin(x_true[2]) * dt
    x_true[2] += w_true[k] * dt
    true_trajectory.append(x_true.copy())

    # Prediction step
    A = np.eye(3)  # State transition is identity for small dt
    B = np.array([[np.cos(x_est[2]) * dt, 0],
                  [np.sin(x_est[2]) * dt, 0],
                  [0, dt]])  # Control input model
    u = np.array([v_meas[k], w_meas[k]])  # Control inputs

    x_pred = A @ x_est + B @ u
    P_pred = A @ P @ A.T + Q

    # Measurement update
    z = u  # Measurements directly observed as v, omega
    H = np.array([[0, 0, 0],  # v doesn't directly map to state
                  [0, 0, 1]])  # omega maps to theta
    y = z - H @ x_pred  # Measurement residual
    S = H @ P_pred @ H.T + R  # Measurement covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

    x_est = x_pred + K @ y
    P = (np.eye(3) - K @ H) @ P_pred

    # Store results
    estimated_trajectory.append(x_est.copy())
    variances.append(np.diag(P).copy())

# Convert to arrays
true_trajectory = np.array(true_trajectory)
estimated_trajectory = np.array(estimated_trajectory)
variances = np.array(variances)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(range(steps), true_trajectory[:, 0], label="True x")
plt.plot(range(steps), estimated_trajectory[:, 0], label="Estimated x")
plt.fill_between(range(steps),
                 estimated_trajectory[:, 0] - np.sqrt(variances[:, 0]),
                 estimated_trajectory[:, 0] + np.sqrt(variances[:, 0]), alpha=0.2)
plt.legend()
plt.title("Position x")

plt.subplot(3, 1, 2)
plt.plot(range(steps), true_trajectory[:, 1], label="True y")
plt.plot(range(steps), estimated_trajectory[:, 1], label="Estimated y")
plt.fill_between(range(steps),
                 estimated_trajectory[:, 1] - np.sqrt(variances[:, 1]),
                 estimated_trajectory[:, 1] + np.sqrt(variances[:, 1]), alpha=0.2)
plt.legend()
plt.title("Position y")

plt.subplot(3, 1, 3)
plt.plot(range(steps), true_trajectory[:, 2], label="True θ")
plt.plot(range(steps), estimated_trajectory[:, 2], label="Estimated θ")
plt.fill_between(range(steps),
                 estimated_trajectory[:, 2] - np.sqrt(variances[:, 2]),
                 estimated_trajectory[:, 2] + np.sqrt(variances[:, 2]), alpha=0.2)
plt.legend()
plt.title("Orientation θ")

plt.tight_layout()
plt.show()
