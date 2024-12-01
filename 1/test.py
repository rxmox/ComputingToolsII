import numpy as np
import matplotlib.pyplot as plt

# Question 12 
# x1 = x0 = 0
# for 101 points

n = np.linspace(1,100,100)
x = np.zeros(101)

for i in range(2,101):
    x[i] = np.sin(x[i-1]) - 0.3 * x[i-2] + 1

# plot for 1-100
yaxis = np.delete(x,0)
xaxis = n
print(x)

plt.plot(xaxis, yaxis)
plt.xlabel('n')
plt.ylabel('xn')
plt.title('Plot of Recursion Equation')
plt.grid()
plt.show()
