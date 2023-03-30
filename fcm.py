import mat73
from matplotlib import pyplot as plt
import numpy as np
from class_means import class_means
from cost import J_fun
import cv2

def update_memberships(neighbourhood, pixels, centers, segments, q):
    """ Return the new memberships assuming the centers

    Args:
        neighbourhood (The Neighbourhood defined bythe Gauusian): 
        image (The Image given): 
        centers (THe Centers provided by K-means): 
        segments (Number of segments): 
        q (The Fuzzy number): 
    """

    M = image.size

    distance = np.zeros(M, segments)
    for i in range(segments):
        distance[:, i] = image**2  - 2 * centers[i] * image + centers[i] ** 2

    power = 1 / (q - 1)
    reverse_d = ( 1 / distance ) ** (power) 
    sumD = np.sum(reverse_d, axis = 1)

    memberships = np.zeros(M, segments)

    for i in range(segments):
        memberships[:, i] = reverse_d[:, i] / sumD
     
    return memberships

data_dict = mat73.loadmat('assignmentsegmentbrain.mat')

image = data_dict["imagedata"]
image = cv2.cvtcolor(image, cv2.color_rgb2gray)
image = (image * 256).astype(np.uint8)
imagemask = data_dict["imagemask"]

k = 4 
q = 1.6

pixels = np.float32(image.reshape((-1,1)))
criteria = (cv2.term_criteria_eps + cv2.term_criteria_max_iter, 100, 0.85)
retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

uInit = np.zeros((pixels.shape[0],centers.shape[0]))
for i in range(len(pixels.shape[0])):
    uInit[i,labels[i]] = 1

maxIters = 100 
u = uInit
J = 0

for i in range(maxIters):
    u = update_memberships(pixels,centers,q)
    centers = class_means(u,pixels,q)
    J = J_fun(u,pixels,centers,q)
    print(i,J)

labels = np.argmax(u,axis = 0)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((image.shape))


fig, axs = plt.subplots(1, 2 )
axs[0].imshow(image,cmap='gray')
axs[1].imshow(segmented_image,cmap='gray')
plt.show()