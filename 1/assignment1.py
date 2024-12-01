import numpy as np
import matplotlib.pyplot as plt

# Question 12
def recursion(n):
    x = np.zeros(n)
    x[0] = x[1] = 0

    for i in range(2, n):
        x[i] = np.sin(x[i-1]) - 0.3 *x[i-2]  + 1

    return x

n_values = np.arange(0, 101)
x_values = recursion(101)

plt.plot(n_values, x_values)
plt.xlabel('n')
plt.ylabel('x_n')
plt.grid(True)
plt.show()


# # Question 11
# k = np.array([[4, 1, 1], [2,1,13], [3,0,-1]])
# result = np.array([[3], [2], [11]])
# Q = np.linalg.solve(k,result)
# print(Q)

# # Question 10
# x = np.array([[1,2,3],[0,7,7],[1,2,1]])
# y = np.array([[2,2,3],[7,6,0],[1,2,1]])
# Q = np.matmul(np.linalg.inv(x), (y + np.square(x)))
# print(Q)

# # Question 9
# u = np.array([-3,4,-2])
# v = np.array([2,-5,-4])
# w = np.array([1,-1,-1])

# Q = (np.dot(u,v))**2 + np.absolute(np.cross(np.cross(u,v), w))
# print(Q)


# # Question 8
# k = []
# for i in range(-50, 31, 1):
#     k.append(i)
# x = np.array(k)
# y = 3*np.square(x) + 2
# print(y)

# # Question 7
# a = np.array([[1,2,-3],[4,8,8],[2,2,4]])
# b = np.array([[5,5,-3],[4,8,8],[2,2,4]])
# ab = np.array([[1,2,-3,5,5,-3],[4,8,8,4,8,8],[2,2,4,2,2,4],[0,0,0,1,2,-3],[0,0,0,4,8,8],[0,0,0,2,2,4]])
# result = np.array([[1],[0],[0],[0],[0],[0]])
# x = np.linalg.solve(ab,result)
# print(x)
# # to double check
# print(np.dot(ab,x))

# #  Question 6
# #  Assumed y = x + xT * x + x^3
# x = np.array([[1,2,-3],[4,8,8],[2,2,4]])
# y = x + np.matmul(np.transpose(x),x) + np.power(x,3)
# print(y)


# # Question 5
# x = -4 + 1j
# y = 3j
# z = np.array([x**y , x*y**2, np.exp(np.sqrt(x))])
# print(f'Answer to Q5: {np.linalg.norm(z)}')

# # Question 5a
# m = []
# p = []
# for i in range(len(z)):
#     print(np.imag(z[i]))
#     m.append(np.sqrt(np.real(z[i])**2 + np.imag(z[i])**2))
#     p.append(np.arctan(np.imag(z[i]) / np.sqrt(np.real(z[i]))))
# print(m)
# print(p)



# # Question 4
# x = np.linspace(0, 400, 500)
# y = np.tanh(x)
# plt.plot(x,y)
# plt.grid(True)
# plt.xlabel('x')
# plt.ylabel("y = tanh(x)")
# plt.show()


# # Question 3
# x = np.sqrt(3)
# y = 0.3*x**2 + np.sqrt(x)
# z = np.sqrt(np.exp(1)) + x - np.log(x) - np.log10(x)
# v = np.sqrt(np.tanh(x*y*z))
# print(v)


# Question 2

# results = []
# for j in range(1, 4, 1):
#     for k in range(-3, 5, 1):
#         x = np.sqrt(j) * k**2 * np.sin(0.1 * (k - j)**2)
#         results.append(x)
  
# print(sum(results))



# # Question 1

# results = []
# for k in range(-3, 5, 1):
#     x = k**2 * np.sin(0.1 * k**2)
#     results.append(x)
# print(sum(results))