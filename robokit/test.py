import numpy as np

# 3차원 배열 생성 (모양: [3, 4, 2])
arr = np.arange(3 * 4 * 2).reshape(3, 4, 2)
print("arr:\n", arr)


print("arr[..., 0]:\n", arr[..., 0])

