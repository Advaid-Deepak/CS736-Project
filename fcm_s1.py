import mat73
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sig
import cv2

def J_fun(memberships,pixels,centres,q,avg_pixels,alpha):
    return np.sum(np.power(memberships,q).T@np.square(pixels-centers.T) + alpha*np.power(memberships,q).T@np.square(avg_pixels-centers.T))

def class_means(memberships,pixels,q,avg_pixels,alpha) :

    powered_Membership = memberships ** q
    c = powered_Membership.T@(pixels+alpha*avg_pixels)
    c = c.T/((1+alpha)*np.sum(powered_Membership,axis = 0))
    return c.T

def update_memberships(pixels, centers, segments, q,avg_pixels,alpha):
    """ Return the new memberships assuming the centers

    Args:
        neighbourhood (The Neighbourhood defined bythe Gauusian): 
        pixels (The pixels given): 
        centers (THe Centers provided by K-means): 
        segments (Number of segments): 
        q (The Fuzzy number): 
    """

    M = pixels.size

    distance = np.zeros((M, segments))
    for i in range(segments):
        distance[:, i] = (pixels**2  - 2 * centers[i] * pixels + centers[i] ** 2).flatten()
        distance[:, i] += alpha*(avg_pixels**2  - 2 * centers[i] * avg_pixels + centers[i] ** 2).flatten()

    power = 1 / (q - 1)
    reverse_d = ( 1 / distance ) ** (power) 
    sumD = np.sum(reverse_d, axis = 1)

    memberships = np.zeros((M, segments))

    for i in range(segments):
        memberships[:, i] = reverse_d[:, i] / sumD
     
    return memberships

# data_dict = mat73.loadmat('assignmentSegmentBrain.mat')

# image = data_dict["imageData"]
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = (image * 256).astype(np.uint8)
# imagemask = data_dict["imageMask"]

image  = cv2.imread('brain_mri.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
avg_img = np.zeros(image.shape)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        sum = 0 
        count = 0
        if i != 0 :
            sum += image[i-1][j]
            count += 1
        if j != 0 :
            sum += image[i][j-1]
            count += 1
        if i != image.shape[0]-1:
            sum += image[i+1][j]
            count += 1
        if j != image.shape[1]-1:
            sum += image[i][j+1]
            count += 1
        avg_img[i,j] = sum/count
 

k = 4 
q = 4

pixels = np.float32(image.reshape((-1,1)))
avg_pixels = np.float32(avg_img.reshape((-1,1)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
savedLabels = np.copy(labels)
savedCenters = np.copy(centers)

uInit = np.zeros((pixels.shape[0],centers.shape[0]))
for i in range(pixels.shape[0]):
    uInit[i,labels[i]] = 1

maxIters = 100 
u = uInit
J = 0
alpha = 0.2


for i in range(maxIters):
    u = update_memberships(pixels,centers,k,q,avg_pixels,alpha)
    centers = class_means(u,pixels,q,avg_pixels,alpha)
    J = J_fun(u,pixels,centers,q,avg_pixels,alpha)
    # print(i,J)

print(u.shape)
labels = np.argmax(u,axis = 1)
print(labels.shape)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((image.shape))

savedCenters = np.uint8(savedCenters)
kmeans_segmented_data = savedCenters[savedLabels.flatten()]
kmeans_segmented_data = kmeans_segmented_data.reshape((image.shape))

fig, axs = plt.subplots(1, 3 )
axs[0].imshow(image,cmap='gray')
axs[1].imshow(segmented_image,cmap='gray')
axs[2].imshow(kmeans_segmented_data, cmap = 'gray')
plt.show()
