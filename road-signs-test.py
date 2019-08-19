import tensorflow as tf
import skimage
from skimage.color import rgb2gray
import numpy as np
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import random
import os

def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/Users/rmb0047/Desktop/tensorflow-test"
train_data_dir = os.path.join(ROOT_PATH, "Data/Training")
test_data_dir = os.path.join(ROOT_PATH, "Data/Testing")

images, labels = load_data(train_data_dir)

images_array = np.array(images)
labels_array = np.array(labels)

#plt.hist(labels, 62)
#plt.show()

#traffic_signs = [300, 2250, 3650, 400]
#for i in range(len(traffic_signs)):
#    plt.subplot(1, 4, i+1)
#    plt.axis('off')
#    plt.imshow(images[traffic_signs[i]])
#    plt.subplots_adjust(wspace=0.5)
#    plt.show()
#    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
#                                                  images[traffic_signs[i]].min(), 
#                                                  images[traffic_signs[i]].max()))

#unique_labels = set(labels)
#plt.figure(figsize=(15, 15))
#i = 1
#for label in unique_labels:
  #  image = images[labels.index(label)]
  #  plt.subplot(8, 8, i)
  #  plt.axis('off')
  #  plt.title("Label {0} ({1})".format(label, labels.count(label)))
  #  i += 1
  #  plt.imshow(image)

#plt.show()


# Resize images
#images32 = [transform.resize(image, (28, 28)) for image in images]
#images32 = np.array(images32)
#traffic_signs = [300, 2250, 3650, 4000]
#for i in range(len(traffic_signs)):
#    plt.subplot(1, 4, i+1)
#    plt.axis('off')
#    plt.imshow(images32[traffic_signs[i]])
#    plt.subplots_adjust(wspace=0.5)
#    plt.show()
#    print("shape: {0}, min: {1}, max: {2}".format(images32[traffic_signs[i]].shape, 
#                                                  images32[traffic_signs[i]].min(), 
#                                                  images32[traffic_signs[i]].max()))


