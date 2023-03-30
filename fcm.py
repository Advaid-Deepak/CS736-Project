import mat73
from matplotlib import pyplot as plt

data_dict = mat73.loadmat('assignmentSegmentBrain.mat')

image = data_dict["imageData"]
imageMask = data_dict["imageMask"]

plt.imshow(image, interpolation='nearest')
plt.show()

k = 3 




