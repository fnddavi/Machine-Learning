from PIL import Image
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

img_pil_b4 = Image.open("T23KMQ_20250604T131239_B04_20m.jp2")
img_pil_b3 = Image.open("T23KMQ_20250604T131239_B03_20m.jp2")
img_pil_b2 = Image.open("T23KMQ_20250604T131239_B02_20m.jp2")

img_b4 = np.array(img_pil_b4)
img_b3 = np.array(img_pil_b3)
img_b2 = np.array(img_pil_b2)

img_b432 = np.zeros((5490, 5490, 3), dtype=np.float32)
img_b432[:,:, 0] = np.where(img_b4 > 8000, 1.0, img_b4 / 8000) 
img_b432[:,:, 1] = np.where(img_b3 > 8000, 1.0, img_b3 / 8000)
img_b432[:,:, 2] = np.where(img_b2 > 8000, 1.0, img_b2 / 8000)

 # Show the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(img_b432)
plt.show()