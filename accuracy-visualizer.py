import matplotlib.pyplot as plt
import numpy as np

arr = np.loadtxt("log.txt")
print(arr.shape)

step, train, valid, cost = np.split(arr, 4, axis=1)
step = np.reshape(step, [100])
train = np.reshape(train, [100])
valid = np.reshape(valid, [100])
# cost = np.reshape(cost, [100])

plt.plot(step, train, "r-", step, valid, "b-")
plt.show()
