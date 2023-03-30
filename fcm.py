import mat73
from matplotlib import pyplot as plt
import numpy as np
import cv2

data_dict = mat73.loadmat('assignmentSegmentBrain.mat')

image = data_dict["imageData"]
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = (image * 256).astype(np.uint8)
imageMask = data_dict["imageMask"]

k = 4 


pixels = np.float32(image.reshape((-1,1)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))


fig, axs = plt.subplots(1, 2 )
axs[0].imshow(image)
axs[1].imshow(segmented_image)
plt.show()
