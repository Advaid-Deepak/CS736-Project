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


    parser.add_argument("--k", type = int, help="number of segments in the image")
    parser.add_argument("--method", choices=["fcm","fgfcm", "enfcm", "bias_fcm", "fcm_local", "fcm_s1"], help="choose your flavor of fcm")
    parser.add_argument("--image", type=validate, help="path to the image file")
    # parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--q", default=1.6, type=float, help="fuzziness factor")
    parser.add_argument("--save", action="store_true", help="files will be saved in results folder with name {file_name}_{method}.png")
    parser.add_arguments("--iter", default = 20, type = int, help="Number of iterations")

    args = parser.parse_args()
    segments = args.k
    method = args.method
    image_file = args.image
    q = args.q
    save = args.save
    iter = args.iter

    if q <= 1:
        sys.exit("ERROR: q should be greater than 1, 1 is just k_means, will give error")

    if iter <= 0:
        sys.exit("ERROR: number of iterations should be greater than 0")

    extension = os.path.splitext(os.path.basename( image_file ))[1]
    if extension in [".png", ".jpeg", ".jpg"]:
        image  = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255
        imagemask = np.ones(image.shape)

    elif extension in [".mat"]:
        data_dict = mat73.loadmat(image_file)
        image = data_dict["imageData"]
        image = (image * 255).astype(np.uint8)
        image = image/255
        imagemask = data_dict["imageMask"]

    else:
        sys.exit("ERROR: File Format not Supported")



    if method == 'fcm':
        result, cost = fcm.c_means(image, imagemask, segments, q, iter)
    elif method == "fgfcm":
        result, cost = fgfcm.c_means(image, imagemask, segments, q, iter)
    elif method == 'enfcm':
        result, cost = enfcm.c_means(image, imagemask, segments, q, iter)
    elif method == 'bias_fcm':
        result, cost = bias_fcm.c_means(image, imagemask, segments, q, iter)
    elif method == 'fcm_local':
        result, cost = fcm_local.c_means(image, imagemask, segments, q, iter)
    elif method == 'fcm_s1':
        result, cost = fcm_s1.c_means(image, imagemask, segments, q, iter)
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



