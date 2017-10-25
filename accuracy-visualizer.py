import matplotlib.pyplot as plt
import numpy as np

arr = np.loadtxt("log.txt")
print(arr.shape)

step, train, valid, cost = np.split(arr, 4, axis=1)
step = np.reshape(step, [200])
train = np.reshape(train, [200])
valid = np.reshape(valid, [200])
cost = np.reshape(cost, [200])

plt.plot(step, train, "r-", step, valid, "b-", step, cost, "g-")
plt.show()
