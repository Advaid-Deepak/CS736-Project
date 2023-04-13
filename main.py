import argparse
import os
import sys
import cv2
import mat73
import numpy as np
import matplotlib.pyplot as plt
import fcm
import fgfcm
import enfcm
import bias_fcm
import fcm_local
import fcm_s1


def validate(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError(f"{f} does not exist")
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--k", type = int)
    parser.add_argument("--method", choices=["fcm","fgfcm", "enfcm", "bias_fcm", "fcm_local", "fcm_s1"])
    parser.add_argument("--image", type=validate)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--q", default=1.6)
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    segments = args.k
    method = args.method
    image_file = args.image
    q = args.q
    save = args.save

    extension = os.path.splitext(os.path.basename( image_file ))[1]
    if extension in [".png", ".jpeg", ".jpg"]:
        image  = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255
        imagemask = np.ones(image.shape)

    elif extension in [".mat"]:
        data_dict = mat73.loadmat(image_file)
        image = data_dict["imageData"]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = (image * 256).astype(np.uint8)
        imagemask = data_dict["imageMask"]

    else:
        sys.exit("ERROR: File Format not Supported")



    if method == 'fcm':
        result, cost = fcm.c_means(image, imagemask, segments, q)
    elif method == "fgfcm":
        result, cost = fgfcm.c_means(image, imagemask, segments, q)
    elif method == 'enfcm':
        result, cost = enfcm.c_means(image, imagemask, segments, q)
    elif method == 'bias_fcm':
        result, cost = bias_fcm.c_means(image, imagemask, segments, q)
    elif method == 'fcm_local':
        result, cost = fcm_local.c_means(image, imagemask, segments, q)
    elif method == 'fcm_s1':
        result, cost = fcm_s1.c_means(image, imagemask, segments, q)
    else:
        sys.exit("ERROR: method not supported")

    plt.imshow(result)
    plt.title("result")
    if save:
        plt.savefig(f"./results/{os.path.splitext(os.path.basename(image_file))[0]}_{method}.png")
    plt.show()

    plt.plot(cost)
    plt.title("cost")
    if save:
        plt.savefig(f"./results/{os.path.splitext(os.path.basename(image_file))[0]}_{method}_cost.png")
    plt.show()


