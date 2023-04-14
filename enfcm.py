import mat73
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sig
import cv2

def J_fun(u,y,c,q,gamma):
    d = y-c.T
    ret = np.sum(np.power(u,q)*(np.square(d)*gamma))
    return ret

def class_means(memberships,pixels,q,gamma) :
    pixels = pixels.reshape((-1,1))
    powered_Membership = memberships ** q
    c = powered_Membership.T@(pixels*gamma)
    c = c/(powered_Membership.T@gamma)
    return c

def update_memberships(pixels, centers, segments, q):
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

    distance[distance <= 0] = 1e-10
    power = 1 / (q - 1)
    reverse_d = ( 1 / distance ) ** (power) 
    sumD = np.sum(reverse_d, axis = 1)

    memberships = np.zeros((M, segments))

    for i in range(segments):
        memberships[:, i] = reverse_d[:, i] / sumD
     
    return memberships

def c_means(image, imagemask,k, q = 1.6, iter = 20):

    image = image*imagemask
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
     

    alpha = 0.2

    zeta = (image + alpha*avg_img)/(1+alpha)
    pixels = np.float32(zeta.reshape((-1,1)))
    values,inverse,counts = np.unique(pixels,return_inverse=True,return_counts=True)
    avg_pixels = np.float32(avg_img.reshape((-1,1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    savedLabels = np.copy(labels)
    gt = np.copy(labels).flatten()
    savedCenters = np.copy(centers)

    # uInit = np.zeros((pixels.shape[0],centers.shape[0]))
    # for i in range(pixels.shape[0]):
    #     uInit[i,labels[i]] = 1

    uInit = np.random.rand(values.shape[0],centers.shape[0])
    uInit = uInit/uInit.sum(axis=1)[:,None]
    u = uInit
    J = 0
    counts = counts.reshape((-1,1))
    cost = []

    for i in range(iter):
        centers = class_means(u,values,q,counts)
        u = update_memberships(values,centers,k,q)
        J = J_fun(u,values.reshape((-1,1)),centers.reshape((-1,1)),q,counts.reshape((-1,1)))
        cost.append(J)
        print(f"iteration {i}: { J }")


    labels = np.argmax(u,axis = 1)
    seg = np.copy(labels[inverse]).flatten()
    if np.all(centers >= 0) and np.all(centers <= 1):
        centers = centers * 255
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_data = segmented_data[inverse]
    segmented_image = segmented_data.reshape((image.shape))

    if np.all(savedCenters >= 0) and np.all(savedCenters <= 1):
        savedCenters = savedCenters * 255
    savedCenters = np.uint8(savedCenters)
    kmeans_segmented_data = savedCenters[savedLabels.flatten()]
    kmeans_segmented_data = kmeans_segmented_data.reshape((image.shape))
    dice = np.zeros(k)
    for i in range(k):
        dice[i] = np.sum(seg[gt==i]==i)*2.0 / (np.sum(seg[seg==i]==i) + np.sum(gt[gt==i]==i))
    print("dice_accuracy",np.mean(dice))
    fig, axs = plt.subplots(1, 3 )
    axs[0].imshow(image,cmap='gray')
    axs[0].set_title("original")
    axs[1].imshow(segmented_image,cmap='gray')
    axs[1].set_title("enfcm")
    axs[2].imshow(kmeans_segmented_data, cmap = 'gray')
    axs[2].set_title("kmeans")
    plt.show()

    return segmented_image, cost


if __name__ == "__main__":
    
    data_dict = mat73.loadmat('assignmentSegmentBrain.mat')

    image = data_dict["imageData"]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (image * 256).astype(np.uint8)
    imagemask = data_dict["imageMask"]

    # image  = cv2.imread('brain_mri.jpeg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    q = 2
    alpha = 0.2

    # fig, axs = plt.subplots(1, 2 )
    # axs[0].imshow(image,cmap='gray')
    # axs[1].imshow(avg_img,cmap='gray')
    # plt.show()

    zeta = (image + alpha*avg_img)/(1+alpha)
    pixels = np.float32(zeta.reshape((-1,1)))
    values,inverse,counts = np.unique(pixels,return_inverse=True,return_counts=True)
    avg_pixels = np.float32(avg_img.reshape((-1,1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    savedLabels = np.copy(labels)
    savedCenters = np.copy(centers)

    # uInit = np.zeros((pixels.shape[0],centers.shape[0]))
    # for i in range(pixels.shape[0]):
    #     uInit[i,labels[i]] = 1

    uInit = np.random.rand(values.shape[0],centers.shape[0])
    uInit = uInit/uInit.sum(axis=1)[:,None]
    maxIters = 20
    u = uInit
    J = 0
    counts = counts.reshape((-1,1))
    for i in range(maxIters):
        centers = class_means(u,values,q,counts)
        u = update_memberships(values,centers,k,q)
        J = J_fun(u,values.reshape((-1,1)),centers.reshape((-1,1)),q,counts.reshape((-1,1)))


        if i % 1 == 0:
            labels = np.argmax(u,axis = 1)
            centers = np.uint8(centers)

            segmented_data = centers[labels.flatten()]
            segmented_data = segmented_data[inverse]
            segmented_image = segmented_data.reshape((image.shape))
            savedCenters = np.uint8(savedCenters)
            kmeans_segmented_data = savedCenters[savedLabels.flatten()]
            kmeans_segmented_data = kmeans_segmented_data.reshape((image.shape))

            fig, axs = plt.subplots(1, 3 )
            axs[0].imshow(image,cmap='gray')
            axs[1].imshow(segmented_image,cmap='gray')
            axs[2].imshow(kmeans_segmented_data, cmap = 'gray')
            plt.show()

        
        # t = input()
        print(i,J)
