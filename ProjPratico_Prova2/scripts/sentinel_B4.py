from PIL import Image
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

img_pil = Image.open("T23KMQ_20250604T131239_B04_20m.jp2")

print(img_pil.size)
print(img_pil.mode)
print(img_pil.format)

img = np.array(img_pil)

print(img[1000,1000])
img2 = np.where(img > 5000, 1.0, img / 5000)
print(img2[1000,1000])
 # Show the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(img2, cmap="gray")
plt.show()