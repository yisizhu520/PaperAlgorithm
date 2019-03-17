import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19950223)

N = 200
x1 = np.random.rand(N) * 0.5
y1 = np.random.rand(N) * 0.5 + 0.5
plt.scatter(x1, y1, marker='x', c='red', label='tiger', alpha=0.5)

x2 = np.random.rand(N) * 0.5 + 0.5
y2 = np.random.rand(N) * 0.5 + 0.5
plt.scatter(x2, y2, marker='^', c='green', label='pandas', alpha=0.5)

N = 400
x3 = np.random.rand(N) * 1
y3 = np.random.rand(N) * 0.5
plt.scatter(x3, y3, marker='+', c='blue', label='cat', alpha=0.5)

N = 20

# TODO 在这儿修改半径大小
def get_random_circle_size(max_size=1000):
    return [np.random.rand(N) * 1000 + 500, 0.2 / 6000 * (max_size + 500)]


area, unit = get_random_circle_size()
x1 = np.random.rand(N) * (0.5 - unit * 2) + unit
y1 = np.random.rand(N) * (0.5 - unit * 2) + 0.5 + unit
# c = np.sqrt(area1)
plt.scatter(x1, y1, s=area, c='red', label='tiger', alpha=0.5)

area, unit = get_random_circle_size()
x2 = np.random.rand(N) * (0.5 - unit * 2) + 0.5 + unit
y2 = np.random.rand(N) * (0.5 - unit * 2) + 0.5 + unit
r = np.sqrt(x2 * x2 + y2 * y2)
plt.scatter(x2, y2, s=area, c='green', label='pandas', alpha=0.5)

N = 40
area, unit = get_random_circle_size()
x3 = np.random.rand(N) * (1 - unit * 2) + unit
y3 = np.random.rand(N) * (0.5 - unit * 2) + unit
plt.scatter(x3, y3, marker='+', c='blue', label='cat', alpha=0.5)
plt.scatter(x3, y3, s=area, c='blue', label='cat', alpha=0.5)

x = np.linspace(0, 1)

plt.plot(np.linspace(0, 1), np.linspace(0.5, 0.5))
plt.plot([0.5, 0.5], [0.5, 1])
# x = np.linspace(0, 1)
# y = np.sqrt(abs(0.6 * 0.6 - (x-0.5) * (x-0.5)))
# plt.plot(x, y, c='blue')
#
# x = np.linspace(0, 0.5)
# y = 1 - np.sqrt(abs(0.5 * 0.5 - (x) * (x)))
# plt.plot(x, y, c='red')
#
# x = np.linspace(0.5, 1)
# y = 1-np.sqrt(abs(0.5 * 0.5 - (1-x) * (1-x)))
# plt.plot(x, y, c='green')

plt.show()
