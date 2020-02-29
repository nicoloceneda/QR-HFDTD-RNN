""" LOGISTIC SIGMOID FUNCTION
    -------------------------
    Graph of the logistic sigmoid function
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# 1. PLOT THE LOGISTIC SIGMOID FUNCTION
# -------------------------------------------------------------------------------


# Define the logistic sigmoid function

def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))


# Plot the logistic sigmoid function

z = np.arange(-10, 10, 0.01)
phi_z = sigmoid(z)

fig, ax = plt.subplots()
ax.plot(z, phi_z, color="blue")
ax.axhline(0.0, linestyle='--', linewidth=0.5, color='k')
ax.axhline(0.5, linestyle='--', linewidth=0.5, color='k')
ax.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
ax.axvline(0.0, linestyle='--', linewidth=0.5, color='k')
ax.set_xlabel("Z")
ax.set_ylabel(r"$\phi(Z)$", fontsize=11)
ax.set_title("Logistic Sigmoid Function")
fig.tight_layout()
plt.savefig('images/z_logistic_sigmoid_function.png')


# Plot the logistic sigmoid function and the cost function




def cost_1(z):

    return - np.log(sigmoid(z))


def cost_0(z):

    return - np.log(1 - sigmoid(z))


z = np.arange(-10, 10, 0.01)

phi_z = sigmoid(z)
c1 = cost_1(z)
c0 = cost_0(z)

fig, ax = plt.subplots()
ax.plot(phi_z, c1, color="blue", label="J(w) if y=1")
ax.plot(phi_z, c0, color="red", label="J(w) if y=0")
ax.set_xlabel(r"$\phi(Z)$", fontsize=11)
ax.set_ylabel("J(w)")
ax.set_title("Logistic Sigmoid Function")
ax.legend(loc="upper center")
ax.set_xlim([0, 1])
ax.set_ylim([0, 5.1])

fig.tight_layout()
plt.savefig('z_logistic_and_cost_functions.png')



import numpy as np
import matplotlib.pyplot as plt

def fun(x, yi, xi):

    return - yi/ xi * x + yi

x_axis = np.arange(0.2, 10, 0.2)
y_axis = np.arange(10, 0.2, -0.2)

fig, ax = plt.subplots()
for xi, yi in zip(x_axis, y_axis):

        x = np.arange(0, xi, 0.01)
        ax.plot(x, fun(x, yi, xi), color="blue")


