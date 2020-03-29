""" THEORY REVIEW
    -------------
    Generate some of the illustrations used in the theory review.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 29 March 2020
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


# Import the libraries

import os
import numpy as np
import matplotlib.pyplot as plt


# Create the directory to store the images:

if not os.path.isdir('images_appendix'):

    os.mkdir('images_appendix')


# -------------------------------------------------------------------------------
# 1. APPENDIX B: LOGISTIC SIGMOID FUNCTION
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
plt.savefig('images_appendix/z_logistic_sigmoid_function.png')


# -------------------------------------------------------------------------------
# 2. APPENDIX C: LOGISTIC COST FUNCTION
# -------------------------------------------------------------------------------


# Define the logistic cost function

def cost_1(z):

    return - np.log(sigmoid(z))


def cost_0(z):

    return - np.log(1 - sigmoid(z))


# Plot the logistic cost function

z = np.arange(-10, 10, 0.01)
phi_z = sigmoid(z)
c1 = cost_1(z)
c0 = cost_0(z)

fig, ax = plt.subplots()
ax.plot(phi_z, c1, color="blue", label="J(w) if y=1")
ax.plot(phi_z, c0, color="red", label="J(w) if y=0")
ax.set_xlabel(r"$\phi(Z)$", fontsize=11)
ax.set_ylabel("J(w)")
ax.set_title("Logistic Cost Function")
ax.legend(loc="upper center")
ax.set_xlim([0, 1])
ax.set_ylim([0, 5.1])
plt.savefig('images_appendix/z_logistic_cost_function.png')
