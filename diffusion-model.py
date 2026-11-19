#!/usr/bin/env python
# coding: utf-8

# # A 1D Diffusion Model

# For the CSDMS roadshow we develop a one-dimensional numerical model for diffusion.
# It assumes a constant diffusivity. 
# It uses a regular grid. 
# It has a step function for an initial profile.
# It has fixed boundary conditions.
# _We add these comments up here for good FAIR practices, also saves us some inline_

# The diffusion equation (_using some LaTeX to make it look nice here_):
# 
# $$ \frac{\partial z}{\partial t} = D\frac{\partial^2 z}{\partial x^2} $$

# where z is an elevation, and D being diffusitivity, by some curvature (the derivative).
# 
# Looking a little nicer (using _dollar signs_ around variables):
# Here, $z$ is elevation, $t$ is time, $x$ is the spatial dimension, and $D$ is a transport-rate coefficient, sometimes referred to as hillslope diffusivity---with dimensions of length squared per time.

# The discretized verison of the diffusion equation well solve with our model is:
# 
# $$ z^{t+1}_x = z^t_x + {D \Delta t \over \Delta x^2} (z^t_{x+1} - 2z^t_x + z^t_{x-1}) $$

# This is the Forward-in-time, centered-in-space (FTCS scheme that you can refer to in textbooks or literature.

# We'll use two libraries, NumPy (for arrays) and Matplotlib (for plots), that aren't a part of the core Python distribution

# In[ ]:


import numpy as np # follow standard idioms is best practice
import matplotlib.pyplot as plt


# Start by setting two fixed model parameters---the diffusivity and the size of the model domain.

# In[ ]:


# intentionally poor variable names...
D = 100 # diffusivity
Lx = 300 # model domain
# long names are no longer prohibited (jupyter allows for tab completion), also avoid CamelCase and capitializations


# Next, set up the grid usin a NumPy array:

# In[ ]:


dx = 0.5 # grid spacing
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x) # number of nodes


# Set the initial conditions for the model:
# The elevation $z$ is a step function with a high value to the left, a low value on the right, and a cliff in the middle of the domain. 

# In[ ]:


z = np.zeros_like(x) # create an empty array, same size of other array
z_left = 500
z_right = 0
z[x <= (Lx /2)] = z_left
z[x > (Lx/2)] = z_right # slicing ("subscripting") and setting z to the determined values


# Plot the initial profile.

# In[ ]:


plt.figure()
plt.plot(x, z, "k");
plt.xlabel("x");
plt.ylabel("z");
plt.title("Initial hillslope profile");


# Let's set up time stepping for our model:
# Calculate a stable time step from a stability criterion.

# In[ ]:


nt = 5000 # number of time steps
dt = 0.5 * dx**2 / D # time step


# We're now ready to run the model! 
# Loop over the time steps of the model, solving the diffusion equation using the FTCS scheme described above.
# Note the use of array operations on the elevation $z$. 
# The boundary conditions remain fixed in each time step.

# In[ ]:


for t in range(0, nt):
	z[1:-1] += D * dt / dx ** 2 * (z[:-2] - 2*z[1:-1] + z[2:])


# For this above loop, it's a relatively straight foward code example of the discretized equation above. 

# Plot the result:

# In[ ]:


plt.figure()
plt.plot(x, z, "b");
plt.xlabel("x");
plt.ylabel("z");
plt.title("Final hillslope profile");


# In[ ]:




