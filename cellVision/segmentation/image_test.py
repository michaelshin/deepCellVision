import numpy
from PIL import Image
from segmentation import segment
import matplotlib.pyplot as plt
img = Image.open('/home/michael/Desktop/Stock-Dock-House.jpg').convert('L')
arr = numpy.array(img)

plt.imshow(segment(arr))
plt.show()
