from data.dataset import Dataset, TestDataset, inverse_normalize
from torch.utils import data as data_
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


dataset = Dataset("/root/Desktop/simple-faster-rcnn-pytorch/VOC2012/VOC2012")
print(len(dataset))

img, bbox, label, scale = dataset[1]
img = (img - img.min())/(img.max()-img.min())
img = img.transpose(1, 2, 0)
print(img.min())
print(img.max())
fig, ax = plt.subplots(1)

ax.imshow(img)

bbox_scaled = [coord for coord in bbox[1]]
ymin, xmin, ymax, xmax = bbox_scaled
rect_width = xmax - xmin
rect_height = ymax - ymin
rect = patches.Rectangle((xmin, ymin), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

ax.text(ymin, xmin, label[1], color='r', verticalalignment='bottom')

plt.savefig('saved_image.png')
plt.show()

