import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0,1.0,2.0])

y = step_function(x)

print(y) 