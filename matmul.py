import numpy as np
import time


size=2048
np.show_config()
x = np.random.randn(size, size).astype(np.float32)
y = np.random.randn(size, size).astype(np.float32)
start = time.time_ns()
z = np.dot(x, y)
end = time.time_ns() - start

print(end / 1000000, " ms")
print(2 * (size **3.) / end , " GFlops/s")
