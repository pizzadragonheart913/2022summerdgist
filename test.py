import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from PIL import Image


testval = Image.open('test.png')
plt.imshow(testval)
print(testval.size)
a = model.predict(np.reshape(testval, (1, 28, 28)))