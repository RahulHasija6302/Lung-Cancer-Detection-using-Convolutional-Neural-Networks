//Step 1: Data Exploration

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Set the path to the dataset folder
data_path = "/path/to/dataset"

# Load the dataset
images = []
labels = []
for label in os.listdir(data_path):
    label_path = os.path.join(data_path, label)
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        labels.append(int(label))

# Display sample images from the dataset
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title("Label: {}".format(labels[i]))
    ax.axis("off")
plt.show()
