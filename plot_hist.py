import numpy as np
import matplotlib.pyplot as plt
import sys

fname = sys.argv[1]
hist = np.load(fname + ".npy")

plt.figure()
plt.plot(hist)
# plt.title("")
plt.xlabel("Epoch")
plt.ylabel("Score")


plt.savefig(fname + ".png")
plt.show()

print("Mean score: " + str(np.mean(hist)))
print("Mean score (last 30 epochs): " + str(np.mean(hist[-30:])))
print("SD score: " + str(np.std(hist)))
print("SD score (last 30 epochs): " + str(np.std(hist[-30:])))
print("High score: " + str(np.max(hist)))
print("High score epoch: " + str(np.argmax(hist) + 1))
