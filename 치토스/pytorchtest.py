from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
image = Image.open('test1.png')
testval = np.array(image)
# testval = torch.Tensor(image)

plt.imshow(testval)