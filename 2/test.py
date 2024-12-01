import sympy as sp
import sympy
import numpy as np
import matplotlib.pyplot as plt

#Question 7 A
# Define symbolic variables
x, y = sympy.symbols('x y')
fxy = sp.exp(-x**2+2*x-y**2+x*y)
#find the derivative
fxy_diffx = sp.diff(fxy, x)
fxy_diffy = sp.diff(fxy, y)
local_max = sp.solve([fxy_diffx, fxy_diffy], [x,y], dict=True)

print("The local maximums are : ", local_max)


#Question 7 B
fxy_num = sp.lambdify((x, y) , fxy) # numeric equation in numpy

x_val = np.linspace(-3,3,300)
y_val= np.linspace(-3,3,300)
x_mesh, y_mesh = np.meshgrid(x_val,y_val)

f_mesh = fxy_num(x_mesh, y_mesh)

quadcontset = plt.contourf(x_mesh, y_mesh,f_mesh)
plt.colorbar(label="Value of f(x,y) ", orientation="horizontal")
x_maxval = local_max[0][x]
y_maxval = local_max[0][y]

plt.scatter(x_maxval, y_maxval)
plt.text(x_maxval, y_maxval, "Maximum value of f(x,y)", va="bottom", ha="center")

plt.xlabel("x")
plt.ylabel("y")
plt.title("contour plot")

plt.show()