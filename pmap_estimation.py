import numpy as np
from matplotlib import pyplot as plt


def question_1():
    k_list = np.linspace(0, 100, 100)
    x0 = 0
    xk = [x0]

    for k in k_list:
        wk = np.random.normal(0, np.sqrt(1))
        x_next = xk[-1] + wk
        xk.append(x_next)

    plt.plot(xk)
    plt.show()


def realizations(a_values):
    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters
    sigma_w2 = 0.01  # Variance of white noise
    sigma_w = np.sqrt(sigma_w2)  # Standard deviation of white noise
    num_steps = 500  # Number of steps for each realization

    # Initial state X0
    X0 = np.array([0.0, 0.0])

    # Function to generate realizations of X_k and corresponding x_k, w_k for given `a`
    def generate_realizations(a, num_steps):
        A = np.array([[0, 1], [a / 2, a / 2]])
        B = np.array([0, 1 - a])
        
        # Initialize arrays to store the process values
        X_k = np.zeros((num_steps, 2))
        x_k = np.zeros(num_steps)
        w_k = np.random.normal(0, sigma_w, num_steps)
        
        # Initial state
        X_k[0] = X0
        
        # Generate realizations of X_k
        for k in range(1, num_steps):
            X_k[k] = A @ X_k[k-1] + B * w_k[k-1]
            x_k[k] = X_k[k, 0]  # The first component is x_k
            
        return X_k[:, 0], w_k  # Returning x_k and w_k

    # Generate realizations for a = 0.1 and a = 0.95
    results = {a: generate_realizations(a, num_steps) for a in a_values}

    # Plotting the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Realizations of $x_k$ and $w_k$ for Different Values of $a$")

    for i, a in enumerate(a_values):
        x_k, w_k = results[a]
        
        # Plot x_k
        axes[i, 0].plot(x_k, label=f"$x_k$ (a = {a})")
        axes[i, 0].set_title(f"Realization of $x_k$ for a = {a}")
        axes[i, 0].set_xlabel("k")
        axes[i, 0].set_ylabel("$x_k$")
        axes[i, 0].legend()
        
        # Plot w_k
        axes[i, 1].plot(w_k, label=f"$w_k$ (a = {a})")
        axes[i, 1].set_title(f"Realization of $w_k$ for a = {a}")
        axes[i, 1].set_xlabel("k")
        axes[i, 1].set_ylabel("$w_k$")
        axes[i, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return results


# Question 2.d
realizations([0.1, 0.95])

# Question  3.a
X = realizations([0.9])[0.9][0]
print('X50 =', X[49])
print('X100 =', X[99])
print('X500 =', X[499])