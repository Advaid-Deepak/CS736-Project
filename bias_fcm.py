import mat73
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import cv2

def J_fun(n, m, memberships,pixels,neighbourhood, bias, q, segments,centers):
    distance = np.zeros(memberships.shape)
    b = bias.reshape((n, m))

    t1 = convolve2d(b, neighbourhood, "same")
    t1 = t1.reshape((-1, 1))
    t2 = convolve2d(b ** 2, neighbourhood, "same")
    t2 = t1.reshape((-1, 1))
    t3 = np.zeros(pixels.shape)

    w = sum(sum(neighbourhood))

    for i in range(segments):
        distance[:, i] = ((pixels**2) * w  - 2 * (centers[i] * pixels)*t1  + (centers[i] ** 2)*t2 ).flatten()

    for i in range(segments):
        t3 = t3 + ((memberships[:,i]**q)*distance[:, i]).reshape((-1, 1))

    t3 = t3.reshape((n, m))

    J = sum(sum(convolve2d(t3, neighbourhood, "same")))

    return J



def class_means(n, m, memberships,pixels,neighbourhood, bias, q, segments, imageMask) :

    numer = convolve2d(bias.reshape((n, m)), neighbourhood, "same").reshape(pixels.shape)
    denom = convolve2d(bias.reshape((n, m)) ** 2, neighbourhood, "same").reshape(pixels.shape)

    centers = np.zeros((segments, 1))

    for k in range(segments):
        powered_Membership = memberships[:, k] ** q
        powered_Membership = powered_Membership.reshape((-1,1))
        num = sum(powered_Membership * (pixels * numer * imageMask))
        den = sum(powered_Membership * (denom * imageMask))
        centers[k] = num / den
    return centers

def gauss(size, sigma = 0.5):
    m = ( size - 1 ) / 2
    y,x = np.ogrid[-m:m+1 , -m:m+1]
    h = np.exp(-(y*y + x*x) / (2*sigma*sigma))
    s = h.sum()
    if s != 0:
        h = h / s
    return h

def update_bias(n, m, neighbourhood, pixels, memberships, centers, segments, q, imageMask):
    
    numer = np.zeros (pixels.shape)
    denom = np.zeros (pixels.shape)
    image = pixels.reshape((n, m))
    for k in range(segments):
        numer = numer + ((memberships[:, k] ** (q))*centers[k]).reshape(pixels.shape)
        denom = denom + ((memberships[:, k] ** (q))*( centers[k]**2 )).reshape(pixels.shape)


    num = convolve2d(image*(numer.reshape((n,m))), neighbourhood, "same").reshape(pixels.shape)
    den = convolve2d(denom.reshape((n, m)), neighbourhood, "same").reshape(pixels.shape)
    
    bias = num/den
    
    bias[np.isnan(bias)] = 0
    bias[np.isinf(bias)] = 0

    bias = bias * imageMask
    return bias
    



def update_memberships(n, m, neighbourhood, pixels, centers, bias, segments, q, imageMask):
    """ Return the new memberships assuming the centers

    Args:
        neighbourhood (The Neighbourhood defined bythe Gauusian): 
        pixels (The pixels given): 
        centers (THe Centers provided by K-means): 
        segments (Number of segments): 
        q (The Fuzzy number): 
    """

    M = pixels.size
    image = pixels.reshape((n, m))

    t1 = convolve2d(bias.reshape(n, m), neighbourhood, "same").reshape(pixels.shape)
    t2 = convolve2d(bias.reshape(n, m) ** 2, neighbourhood, "same").reshape(pixels.shape)

    w = sum(sum(neighbourhood))

    distance = np.zeros((M, segments))
    for i in range(segments):
        distance[:, i] = ((pixels**2) * w  - 2 * (centers[i] * pixels)*t1  + (centers[i] ** 2)*t2 ).flatten()

    distance = distance * imageMask

    distance[distance < 0] = 0
    p =  1 / (q - 1)
    reverse_d = ( 1 / distance ) ** (p) 
    sumD = np.nansum(reverse_d, axis = 1).reshape((-1,1))

    memberships = np.zeros((M, segments))

    memberships = reverse_d / sumD

    # for i in range(segments):
    #     temp = reverse_d[:, i].reshape((-1, 1))
    #     temp = temp / sumD
    #     temp[imageMask == 0] = 0
    #     memberships[:, i] = temp.reshape((-1))
    memberships[np.isnan(memberships)] = 0
    memberships[np.isinf(memberships)] = 0

    memberships = memberships * imageMask
    return memberships




data_dict = mat73.loadmat('assignmentSegmentBrain.mat')

image = data_dict["imageData"]
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = (image * 256).astype(np.uint8)
image_rows, image_cols = image.shape
imagemask = data_dict["imageMask"]
image = imagemask * image
imagemask = imagemask.reshape((-1, 1))


k = 3
q = 1.1

pixels = np.float32(image.reshape((-1,1)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
savedLabels = np.copy(labels)
savedCenters = np.copy(centers)

uInit = np.zeros((pixels.shape[0],centers.shape[0]))
for i in range(pixels.shape[0]):
    uInit[i,labels[i]] = 1


bInit = np.ones(pixels.shape) * imagemask
neighbourhood = gauss(10)
print(neighbourhood)

maxIters = 20 
u = uInit
J = 0
bias = bInit

for i in range(maxIters):
    u = update_memberships(image_rows, image_cols, neighbourhood,pixels,centers,bias, k,q, imagemask)
    centers = class_means(image_rows, image_cols, u,pixels,neighbourhood, bias, q, k, imagemask)
    bias = update_bias(image_rows, image_cols, neighbourhood, pixels, u, centers, k, q, imagemask)
    J = J_fun(image_rows, image_cols, u,pixels,neighbourhood, bias, q, k,centers)
    print(i,J)
    
labels = np.argmax(u,axis = 1)
intcenters = np.uint8(centers)
segmented_data = intcenters[labels.flatten()]
segmented_image = segmented_data.reshape((image.shape))

retval, labels, centers = cv2.kmeans(pixels, k+1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
kmeans_segmented_data = centers[labels.flatten()]
kmeans_segmented_data = kmeans_segmented_data.reshape((image.shape))

biasRemoved = np.zeros(pixels.shape)
for j in range(k):
    biasRemoved = biasRemoved + u[:, j].reshape((-1,1)) * centers[j]
biasRemoved = biasRemoved * imagemask


fig, axs = plt.subplots(3, 2 )
axs[0][0].imshow(image,cmap='gray')
axs[0][1].imshow(segmented_image)
axs[1][0].imshow(kmeans_segmented_data)
axs[1][1].imshow(bias.reshape(image.shape), cmap = 'gray')
axs[2][0].imshow(biasRemoved.reshape(image.shape), cmap = 'gray')
axs[2][1].imshow((pixels - biasRemoved*bias).reshape(image.shape), cmap='gray')


plt.show()

