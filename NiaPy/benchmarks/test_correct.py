import numpy as np
import math

sol = [4,4,4,4]

a = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],\
                [3,7,3,7],[2,9,2,9],[5,5,3,3],\
                [8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])

c = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])

m = 5
val = 0
for i in range(m):

    val += math.pow(np.inner(sol-a[i],np.transpose(sol-a[i])) + c[i], -1)

print(-val)