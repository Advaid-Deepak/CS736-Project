from matplotlib import pyplot as plt
import numpy as np

# dice_values = [0.8457,0.8387,0.8447,0.8473,0.464,0.844]
# method = ['fcm','fcm_s1','enfcm','fgfcm','bias_fcm','fcm_local']
dice_values = [0.8457,0.8387,0.8447,0.8473,0.844]
method = ['fcm','fcm_s1','enfcm','fgfcm','fcm_local']
plt.ylim(0.83,0.85)
plt.bar(method,dice_values)
plt.xlabel('method')
plt.ylabel('dice accuracy')
plt.title('Dice Accuracy vs Method')
plt.show()