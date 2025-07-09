from PIL import Image
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

img_pil = Image.open("T23KMQ_20250604T131239_SCL_20m.jp2")

print(img_pil.size)
print(img_pil.mode)
print(img_pil.format)

img = np.array(img_pil)

scl_palette = [
            '#000000', '#ff0000', '#2f2f2f', '#643200', '#00a000', '#ffe65a',
            '#0000ff', '#808080', '#c0c0c0', '#ffffff', '#64c8ff', '#ff96ff'
        ]
cmap = ListedColormap(scl_palette)
bounds = list(range(len(scl_palette) + 1))
print(bounds)
norm = BoundaryNorm(bounds, cmap.N)

 # Show the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap=cmap, norm=norm)
plt.show()