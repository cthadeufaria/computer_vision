import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


##### 1 #####
# Load data
data = np.loadtxt("./data/txt/data_slam.txt")
poses = data[:, :3]  # x, y, theta (robot pose)
lidar_data = data[:, 6:]  # LIDAR measurements (31 columns)

# Parameters
angles = np.linspace(-30, 30, 61) * np.pi / 180  # LIDAR angles in radians

def detect_corners(lidar_ranges, angles, threshold=0.1):
    points = np.array([
        lidar_ranges * np.cos(angles),  # x
        lidar_ranges * np.sin(angles)   # y
    ]).T
    
    corners = []
    # for i in range(1, len(points) - 1):
    #     d1 = np.linalg.norm(points[i] - points[i - 1])
    #     d2 = np.linalg.norm(points[i] - points[i + 1])
    #     if abs(d1 - d2) > threshold:  # Corner detection
    #         corners.append(points[i])
    # return np.array(corners)
    corners.append(max(np.sqrt(np.abs(points[:,0]**2) + np.abs(points[:,1]**2))))

# Detect corners and map them
global_corners = []
for i in range(len(lidar_data)):
    robot_x, robot_y, robot_theta = poses[i]
    ranges = lidar_data[i]
    corners = detect_corners(ranges, angles)
    
    # Transform to global coordinates
    for corner in corners:
        x, y = corner
        x_global = robot_x + x * np.cos(robot_theta) - y * np.sin(robot_theta)
        y_global = robot_y + x * np.sin(robot_theta) + y * np.cos(robot_theta)
        global_corners.append([x_global, y_global])

global_corners = np.array(global_corners)

# Visualize map
plt.scatter(global_corners[:, 0], global_corners[:, 1], s=10, c='blue', label='Detected Corners')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid()
plt.title("Room Map with Detected Corners")
plt.show()


##### 2 #####
# Load data
data = np.loadtxt("./data/txt/data_slam.txt")
exact_poses = data[:, :3]  # Exact (x, y, theta)
pose_variations = data[:, 3:6]  # Variations (delta_x, delta_y, delta_theta)

# Motion model
def motion_model(state, dt, control):
    x, y, theta = state
    delta_d, delta_theta = control
    x += delta_d * np.cos(theta)
    y += delta_d * np.sin(theta)
    theta += delta_theta
    return np.array([x, y, theta])

# Initialize UKF
dt = 1  # Time step (assume 1 unit between updates)
points = MerweScaledSigmaPoints(3, alpha=0.1, beta=2.0, kappa=0)
ukf = UnscentedKalmanFilter(dim_x=3, dim_z=3, fx=motion_model, hx=lambda x: x, dt=dt, points=points)

# Initial state
ukf.x = exact_poses[0]
ukf.P = np.eye(3) * 0.1  # Covariance matrix

# Process and measurement noise
ukf.Q = np.diag([0.01, 0.01, 0.001])  # Process noise
ukf.R = np.diag([0.05, 0.05, 0.01])  # Measurement noise

# UKF Estimation
ukf_estimates = []
for i, control in enumerate(pose_variations):
    ukf.predict(control=control[:2])  # Use translational and rotational variations
    ukf_estimates.append(ukf.x)

ukf_estimates = np.array(ukf_estimates)

# Compute Error
error_x = exact_poses[:, 0] - ukf_estimates[:, 0]
error_y = exact_poses[:, 1] - ukf_estimates[:, 1]
error_theta = exact_poses[:, 2] - ukf_estimates[:, 2]
error = np.sqrt(error_x**2 + error_y**2)

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(exact_poses[:, 0], exact_poses[:, 1], label="Exact Path", color="blue")
plt.plot(ukf_estimates[:, 0], ukf_estimates[:, 1], label="UKF Path", color="orange", linestyle="dashed")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid()
plt.title("Exact vs. UKF Estimated Path")
plt.show()

print(f"Mean Pose Error: {np.mean(error):.3f} m")
print(f"Max Pose Error: {np.max(error):.3f} m")


##### 3 #####
# Load data
data = np.loadtxt("./data/txt/data_slam.txt")
poses = data[:, :3]  # Ground truth poses (x, y, theta)
pose_variations = data[:, 3:6]  # Variations (delta_x, delta_y, delta_theta)
lidar_data = data[:, 6:]  # LIDAR measurements (31 columns)

# Parameters
angles = np.linspace(-30, 30, 61) * np.pi / 180  # LIDAR angles in radians
corner_threshold = 0.2  # Threshold for corner matching
state = np.array([poses[0, 0], poses[0, 1], poses[0, 2]])  # Initial robot pose
corners = []  # List of detected corners

def motion_model(state, control):
    x, y, theta = state
    delta_d, delta_theta = control
    x += delta_d * np.cos(theta)
    y += delta_d * np.sin(theta)
    theta += delta_theta
    return np.array([x, y, theta])

def detect_corners(lidar_ranges, angles, threshold=0.2):
    points = np.array([
        lidar_ranges * np.cos(angles),
        lidar_ranges * np.sin(angles)
    ]).T
    corners = []
    for i in range(1, len(points) - 1):
        d1 = np.linalg.norm(points[i] - points[i - 1])
        d2 = np.linalg.norm(points[i] - points[i + 1])
        if abs(d1 - d2) > threshold:
            corners.append(points[i])
    return np.array(corners)

def global_transform(state, local_corners):
    x, y, theta = state
    global_corners = []
    for corner in local_corners:
        cx, cy = corner
        gx = x + cx * np.cos(theta) - cy * np.sin(theta)
        gy = y + cx * np.sin(theta) + cy * np.cos(theta)
        global_corners.append([gx, gy])
    return np.array(global_corners)

# SLAM loop
for i in range(len(pose_variations)):
    # Motion Update
    control = pose_variations[i, :2]
    state = motion_model(state, control)
    
    # Detect corners
    local_corners = detect_corners(lidar_data[i], angles)
    global_corners = global_transform(state, local_corners)
    
    # Corner Validation
    for gc in global_corners:
        if all(np.linalg.norm(np.array(c) - gc) > corner_threshold for c in corners):
            corners.append(gc)

# Visualization
corners = np.array(corners)
plt.scatter(corners[:, 0], corners[:, 1], color='red', label="Detected Corners")
plt.plot(poses[:, 0], poses[:, 1], label="Exact Path", color="blue")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid()
plt.title("SLAM Results: Corners and Trajectory")
plt.show()
