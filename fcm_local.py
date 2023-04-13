# Change the cost function 
import mat73
from matplotlib import pyplot as plt
import numpy as np
import cv2

def J_fun(image_rows, image_cols, pixels, memberships,  centers, segments, q):
    M = pixels.size
    image = pixels.reshape((image_rows, image_cols))
    fuzzy_factor = np.zeros((M, segments))

    distance = np.zeros((M, segments))
    for k in range(segments):
        distance[:, k] = (pixels**2  - 2 * centers[k] * pixels + centers[k] ** 2).flatten()
        distanceImage = distance[:, k].reshape((image_rows, image_cols))
        membershipImage = memberships[:, k].reshape((image_rows, image_cols))
        fuzzy_factor_k = np.zeros((image_rows, image_cols))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                sum = 0
                if i > 0 :
                    sum += 1 / (dis(image[i][j], image[i-1][j]) + 1) * ((1 - membershipImage[i-1][j]) ** q ) * (distanceImage[i-1][j])
                if j > 0 :
                    sum += 1 / (dis(image[i][j], image[i][j-1]) + 1) * ((1 - membershipImage[i][j-1]) ** q ) * (distanceImage[i][j-1])
                if i < image.shape[0]-1:
                    sum += 1 / (dis(image[i][j], image[i+1][j]) + 1) * ((1 - membershipImage[i+1][j]) ** q ) * (distanceImage[i+1][j])
                if j < image.shape[1]-1:
                    sum += 1 / (dis(image[i][j], image[i][j+1]) + 1) * ((1 - membershipImage[i][j+1]) ** q ) * (distanceImage[i][j+1])
                fuzzy_factor_k[i][j] = sum
        fuzzy_factor[:, k] = fuzzy_factor_k.reshape((-1))
    return np.sum(fuzzy_factor + distance*memberships)


def class_means(memberships,pixels,q) :

    powered_Membership = memberships ** q
    c = powered_Membership.T@pixels
    c = c.T/np.sum(powered_Membership,axis = 0)
    return c.T

def dis(a, b):
    return a**2 - 2*a*b + b**2

def update_memberships(image_rows, image_cols, pixels, memberships,  centers, segments, q):
    """ Return the new memberships assuming the centers

    Args:
        neighbourhood (The Neighbourhood defined bythe Gauusian): 
        pixels (The pixels given): 
        centers (THe Centers provided by K-means): 
        segments (Number of segments): 
        q (The Fuzzy number): 
    """

    M = pixels.size
    image = pixels.reshape((image_rows, image_cols))
    fuzzy_factor = np.zeros((M, segments))

    distance = np.zeros((M, segments))
    for k in range(segments):
        distance[:, k] = (pixels**2  - 2 * centers[k] * pixels + centers[k] ** 2).flatten()
        distanceImage = distance[:, k].reshape((image_rows, image_cols))
        membershipImage = memberships[:, k].reshape((image_rows, image_cols))
        fuzzy_factor_k = np.zeros((image_rows, image_cols))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                sum = 0
                if i > 0 :
                    sum += 1 / (dis(image[i][j], image[i-1][j]) + 1) * ((1 - membershipImage[i-1][j]) ** q ) * (distanceImage[i-1][j])
                if j > 0 :
                    sum += 1 / (dis(image[i][j], image[i][j-1]) + 1) * ((1 - membershipImage[i][j-1]) ** q ) * (distanceImage[i][j-1])
                if i < image.shape[0]-1:
                    sum += 1 / (dis(image[i][j], image[i+1][j]) + 1) * ((1 - membershipImage[i+1][j]) ** q ) * (distanceImage[i+1][j])
                if j < image.shape[1]-1:
                    sum += 1 / (dis(image[i][j], image[i][j+1]) + 1) * ((1 - membershipImage[i][j+1]) ** q ) * (distanceImage[i][j+1])
                fuzzy_factor_k[i][j] = sum
        fuzzy_factor[:, k] = fuzzy_factor_k.reshape((-1))


    power = 1 / (q - 1)
    reverse_d = ( 1 / (distance + fuzzy_factor)) ** (power) 
    sumD = np.sum(reverse_d, axis = 1)

    memberships = np.zeros((M, segments))

    for i in range(segments):
        memberships[:, i] = reverse_d[:, i] / sumD
     
    return memberships


def c_means(image, imagemask, k, q = 1.6, iter = 20):
    image_rows, image_cols = image.shape

    pixels = np.float32(image.reshape((-1,1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    savedLabels = np.copy(labels)
    savedCenters = np.copy(centers)

    uInit = np.zeros((pixels.shape[0],centers.shape[0]))
    for i in range(pixels.shape[0]):
        uInit[i,labels[i]] = 1

    u = uInit
    J = 0
    cost = []

    for i in range(iter):
        u = update_memberships(image_rows, image_cols, pixels,u, centers,k,q)
        centers = class_means(u,pixels,q)
        J = J_fun(image_rows, image_cols, pixels,u, centers,k,q)
        cost.append(J)
        print(f"Iteration {i}: {J}")

    labels = np.argmax(u,axis = 1)
    
    if np.all(centers >= 0) and np.all(centers <= 1):
        centers = centers * 255
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    if np.all(savedCenters >= 0) and np.all(savedCenters <= 1):
        savedCenters = savedCenters * 255
    savedCenters = np.uint8(savedCenters)
    kmeans_segmented_data = savedCenters[savedLabels.flatten()]
    kmeans_segmented_data = kmeans_segmented_data.reshape((image.shape))

    fig, axs = plt.subplots(1, 3 )
    axs[0].imshow(image,cmap='gray')
    axs[0].set_title("original")
    axs[1].imshow(segmented_image,cmap='gray')
    axs[0].set_title("c_means")
    axs[2].imshow(kmeans_segmented_data, cmap = 'gray')
    axs[0].set_title("k_means")
    plt.show()

    return segmented_image, cost





if __name__ == "__main__":
    # data_dict = mat73.loadmat('assignmentSegmentBrain.mat')
    
    # image = data_dict["imageData"]
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = (image * 256).astype(np.uint8)
    # imagemask = data_dict["imageMask"]

    image  = cv2.imread('brain_mri.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_rows, image_cols = image.shape

    k = 4 
    q = 4

    pixels = np.float32(image.reshape((-1,1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    savedLabels = np.copy(labels)
    savedCenters = np.copy(centers)

    uInit = np.zeros((pixels.shape[0],centers.shape[0]))
    for i in range(pixels.shape[0]):
        uInit[i,labels[i]] = 1

    maxIters = 10 
    u = uInit
    J = 0

    for i in range(maxIters):
        print(pixels.shape)
        print(centers.shape)
        u = update_memberships(image_rows, image_cols, pixels,u, centers,k,q)
        centers = class_means(u,pixels,q)
        J = J_fun(image_rows, image_cols, pixels,u, centers,k,q)
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

