import enfcm
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    iter = 100
    finalCost = []
    image  = cv2.imread("./brain_mri.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255
    imagemask = np.ones(image.shape)

    for alpha in np.linspace(0, 1, 100):
        result, cost = enfcm.c_means(image, imagemask, 4, iter=iter, alpha = alpha, show_image=False )
        finalCost.append(cost[-1])

    plt.plot(finalCost)
    plt.title("cost vs alpha values")
    plt.xlabel("alpha Value")
    plt.ylabel("cost after 100 iterations")
    plt.savefig("./results/alpha_enfcm.png")
    plt.show()
