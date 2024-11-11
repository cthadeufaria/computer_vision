import numpy as np
from matplotlib import pyplot as plt


k_list = np.linspace(0, 100, 100)
x0 = 0
xk = [x0]

for k in k_list:
    wk = np.random.normal(0, np.sqrt(1))
    x_next = xk[-1] + wk
    xk.append(x_next)

plt.plot(xk)
plt.show()


